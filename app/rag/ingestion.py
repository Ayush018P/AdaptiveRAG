import os
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.workspace import Workspace
from app.models.knowledge_base import KnowledgeBase
from app.rag.loaders import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import embedding_service
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
import faiss
from rank_bm25 import BM25Okapi
import numpy as np

def get_or_create_default_kb(workspace_id: str, db: Session) -> KnowledgeBase:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == workspace_id).order_by(KnowledgeBase.created_at.asc()).first()
    if not kb:
        kb_id = f"default_{workspace_id}"
        kb = KnowledgeBase(
            id=kb_id,
            name="Default Knowledge Base",
            description="Default workspace knowledge base",
            workspace_id=workspace_id
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
    return kb

def get_kb_key(workspace_id: str, kb_id: str = None) -> str:
    return f"{workspace_id}:{kb_id}" if kb_id else workspace_id

def ingest_document(workspace_id: str, file_path: str, filename: str, doc_type: str, db: Session, kb_id: str = None):
    # Ensure KB belongs to workspace or resolve default
    if not kb_id:
        kb = get_or_create_default_kb(workspace_id, db)
        kb_id = kb.id
    else:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == workspace_id).first()
        if not kb:
            raise ValueError(f"Knowledge base {kb_id} not found in workspace {workspace_id}")

    # 1. Save document to DB first
    doc = Document(
        filename=filename,
        doc_type=doc_type,
        source_path=file_path,
        workspace_id=workspace_id,
        kb_id=kb_id,
        status="PENDING",
        error_message=None
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    try:
        doc.status = "PROCESSING"
        doc.error_message = None
        db.commit()
        
        # 2. Load text
        text = load_document(file_path, doc_type)
        
        # 3. Chunk text
        raw_chunks = chunk_text(text)
        
        chunk_models = []
        for c in raw_chunks:
            chunk = Chunk(
                document_id=doc.id,
                text=c['text'],
                chunk_index=c['chunk_index'],
                page_number=c['page_number'],
                workspace_id=workspace_id,
                kb_id=kb_id
            )
            chunk_models.append(chunk)
            
        db.add_all(chunk_models)
        db.commit()
        
        for c in chunk_models:
            db.refresh(c)
            
        # 4. Embed chunks
        if chunk_models:
            texts = [c.text for c in chunk_models]
            ids = [c.id for c in chunk_models]
            
            vectors = embedding_service.embed_documents(texts)
            
            # 5. Add to Vector Store & BM25 for both KB index and workspace index
            kb_key = get_kb_key(workspace_id, kb_id)
            faiss_store.add_vectors(kb_key, ids, vectors)
            bm25_store.fit(kb_key, ids, texts)

            faiss_store.add_vectors(workspace_id, ids, vectors)
            bm25_store.fit(workspace_id, ids, texts)
            
        doc.status = "READY"
        doc.error_message = None
        db.commit()
        return doc
        
    except Exception as e:
        # Rollback chunks and set failed
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        doc.status = "FAILED"
        doc.error_message = str(e)
        db.commit()
        # Also rebuild indices to remove any partial insertion
        _rebuild_kb_index(workspace_id, kb_id, db)
        _rebuild_workspace_index(workspace_id, db)
        raise e

def _rebuild_kb_index(workspace_id: str, kb_id: str, db: Session):
    kb_key = get_kb_key(workspace_id, kb_id)
    chunks = db.query(Chunk).filter(Chunk.workspace_id == workspace_id, Chunk.kb_id == kb_id).all()
    if not chunks:
        faiss_store.reset(kb_key)
        bm25_store.reset(kb_key)
        return

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]

    vectors = embedding_service.embed_documents(texts)
    
    new_faiss = faiss.IndexFlatL2(faiss_store.dimension)
    vectors_np = np.array(vectors).astype('float32')
    new_faiss.add(vectors_np)
    
    tokenized_corpus = [text.lower().split() for text in texts]
    new_bm25 = BM25Okapi(tokenized_corpus)
    
    faiss_store.replace_index(kb_key, new_faiss, ids.copy())
    bm25_store.replace_index(kb_key, tokenized_corpus, ids.copy(), new_bm25)

def _rebuild_workspace_index(workspace_id: str, db: Session):
    chunks = db.query(Chunk).filter(Chunk.workspace_id == workspace_id).all()
    if not chunks:
        faiss_store.reset(workspace_id)
        bm25_store.reset(workspace_id)
        return

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]

    vectors = embedding_service.embed_documents(texts)
    
    # Safely rebuild FAISS
    new_faiss = faiss.IndexFlatL2(faiss_store.dimension)
    vectors_np = np.array(vectors).astype('float32')
    new_faiss.add(vectors_np)
    
    # Safely rebuild BM25
    tokenized_corpus = [text.lower().split() for text in texts]
    new_bm25 = BM25Okapi(tokenized_corpus)
    
    faiss_store.replace_index(workspace_id, new_faiss, ids.copy())
    bm25_store.replace_index(workspace_id, tokenized_corpus, ids.copy(), new_bm25)

def reindex_knowledge_base(workspace_id: str, kb_id: str, db: Session):
    # Verify KB belongs to workspace
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == workspace_id).first()
    if not kb:
        raise ValueError("Knowledge base not found or unauthorized")
    _rebuild_kb_index(workspace_id, kb_id, db)

def reindex_workspace(workspace_id: str, db: Session):
    _rebuild_workspace_index(workspace_id, db)
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == workspace_id).all()
    for kb in kbs:
        _rebuild_kb_index(workspace_id, kb.id, db)

def reindex_all(db: Session):
    workspaces = db.query(Workspace).all()
    for ws in workspaces:
        reindex_workspace(ws.id, db)

def delete_document(workspace_id: str, document_id: int, db: Session):
    # Ensure document belongs to workspace
    doc = db.query(Document).filter(Document.id == document_id, Document.workspace_id == workspace_id).first()
    if not doc:
        return False
        
    kb_id = doc.kb_id

    # Remove file on disk if exists
    if doc.source_path and os.path.exists(doc.source_path):
        try:
            os.remove(doc.source_path)
        except OSError:
            pass

    # Delete chunks
    db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    
    # Delete document
    db.delete(doc)
    db.commit()
    
    # Reindex KB and workspace to remove stale chunks
    if kb_id:
        _rebuild_kb_index(workspace_id, kb_id, db)
    _rebuild_workspace_index(workspace_id, db)
    return True

def retry_ingest_document(workspace_id: str, document_id: int, db: Session):
    doc = db.query(Document).filter(Document.id == document_id, Document.workspace_id == workspace_id).first()
    if not doc:
        return False
        
    if not doc.source_path or not os.path.exists(doc.source_path):
        doc.status = "FAILED"
        doc.error_message = "Source file missing from disk"
        db.commit()
        return False

    kb_id = doc.kb_id
    if not kb_id:
        kb = get_or_create_default_kb(workspace_id, db)
        kb_id = kb.id
        doc.kb_id = kb_id

    # Reset chunks first
    db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    doc.status = "PROCESSING"
    doc.error_message = None
    db.commit()

    try:
        text = load_document(doc.source_path, doc.doc_type)
        raw_chunks = chunk_text(text)
        chunk_models = []
        for c in raw_chunks:
            chunk = Chunk(
                document_id=doc.id,
                text=c['text'],
                chunk_index=c['chunk_index'],
                page_number=c['page_number'],
                workspace_id=workspace_id,
                kb_id=kb_id
            )
            chunk_models.append(chunk)

        db.add_all(chunk_models)
        db.commit()

        for c in chunk_models:
            db.refresh(c)

        if chunk_models:
            texts = [c.text for c in chunk_models]
            ids = [c.id for c in chunk_models]
            vectors = embedding_service.embed_documents(texts)
            
            kb_key = get_kb_key(workspace_id, kb_id)
            faiss_store.add_vectors(kb_key, ids, vectors)
            bm25_store.fit(kb_key, ids, texts)

            faiss_store.add_vectors(workspace_id, ids, vectors)
            bm25_store.fit(workspace_id, ids, texts)

        doc.status = "READY"
        doc.error_message = None
        db.commit()
        return True
    except Exception as e:
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        doc.status = "FAILED"
        doc.error_message = str(e)
        db.commit()
        if kb_id:
            _rebuild_kb_index(workspace_id, kb_id, db)
        _rebuild_workspace_index(workspace_id, db)
        raise e

def delete_knowledge_base_cascade(workspace_id: str, kb_id: str, db: Session) -> bool:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == workspace_id).first()
    if not kb:
        return False

    docs = db.query(Document).filter(Document.workspace_id == workspace_id, Document.kb_id == kb_id).all()
    for d in docs:
        if d.source_path and os.path.exists(d.source_path):
            try:
                os.remove(d.source_path)
            except OSError:
                pass

    from app.models.conversation import Conversation
    from app.models.query import QueryLog

    # Set kb_id to None for conversations and queries scoped to this KB to preserve history and foreign keys
    db.query(Conversation).filter(Conversation.workspace_id == workspace_id, Conversation.kb_id == kb_id).update({Conversation.kb_id: None})
    db.query(QueryLog).filter(QueryLog.workspace_id == workspace_id, QueryLog.kb_id == kb_id).update({QueryLog.kb_id: None})

    db.query(Chunk).filter(Chunk.workspace_id == workspace_id, Chunk.kb_id == kb_id).delete()
    db.query(Document).filter(Document.workspace_id == workspace_id, Document.kb_id == kb_id).delete()
    db.delete(kb)
    db.commit()

    kb_key = get_kb_key(workspace_id, kb_id)
    faiss_store.reset(kb_key)
    bm25_store.reset(kb_key)

    _rebuild_workspace_index(workspace_id, db)
    return True
