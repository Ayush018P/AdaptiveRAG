import os
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.knowledge_base import KnowledgeBase
from app.rag.loaders import load_document
from app.rag.chunker import chunk_text
from app.rag.embeddings import embedding_service
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
from app.rag.ingestion import (
    ingest_document,
    reindex_workspace,
    reindex_knowledge_base,
    delete_document,
    retry_ingest_document,
    delete_knowledge_base_cascade,
    get_or_create_default_kb,
    get_kb_key,
    _rebuild_kb_index,
    _rebuild_workspace_index
)
from app.core.auth import get_current_user, validate_workspace

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def _render_documents_page(request: Request, ws_id: str, db: Session, active_kb_id: str = None, message: str = None, error: str = None):
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id).order_by(KnowledgeBase.created_at.asc()).all()
    
    kbs_data = []
    for kb in kbs:
        kb_docs = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb.id).all()
        doc_count = len(kb_docs)
        ready_count = sum(1 for d in kb_docs if d.status == "READY")
        failed_count = sum(1 for d in kb_docs if d.status == "FAILED")
        processing_count = sum(1 for d in kb_docs if d.status == "PROCESSING")
        chunk_count = sum(len(d.chunks) for d in kb_docs)
        kbs_data.append({
            "kb": kb,
            "id": kb.id,
            "name": kb.name,
            "description": kb.description or "",
            "created_at": kb.created_at,
            "doc_count": doc_count,
            "ready_count": ready_count,
            "failed_count": failed_count,
            "processing_count": processing_count,
            "chunk_count": chunk_count
        })

    active_kb_data = None
    docs = []
    if kbs_data:
        if active_kb_id:
            active_kb_data = next((k for k in kbs_data if k["id"] == active_kb_id), kbs_data[0])
        else:
            active_kb_data = kbs_data[0]

        docs = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == active_kb_data["id"]).order_by(Document.id.desc()).all()
    else:
        docs = db.query(Document).filter(Document.workspace_id == ws_id).order_by(Document.id.desc()).all()

    return templates.TemplateResponse("documents.html", {
        "request": request,
        "knowledge_bases": kbs_data,
        "active_kb": active_kb_data,
        "documents": docs,
        "workspace_id": ws_id,
        "message": message,
        "error": error
    })

@router.get("/documents")
async def documents_page(request: Request, workspace_id: str = None, kb_id: str = None, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id)

@router.post("/knowledge-bases")
async def create_kb(request: Request, name: str = Form(...), description: str = Form(""), workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    clean_name = name.strip() if name else ""

    if not clean_name or len(clean_name) > 100:
        return _render_documents_page(request, ws_id, db, error="Knowledge base name must be between 1 and 100 characters.")

    existing = db.query(KnowledgeBase).filter(
        KnowledgeBase.workspace_id == ws_id,
        KnowledgeBase.name.ilike(clean_name)
    ).first()
    if existing:
        return _render_documents_page(request, ws_id, db, active_kb_id=existing.id, error=f"A knowledge base named '{clean_name}' already exists in this workspace.")

    new_kb = KnowledgeBase(
        name=clean_name,
        description=description.strip() if description else None,
        workspace_id=ws_id
    )
    db.add(new_kb)
    db.commit()
    db.refresh(new_kb)

    return _render_documents_page(request, ws_id, db, active_kb_id=new_kb.id, message=f"Knowledge base '{clean_name}' created successfully.")

@router.post("/knowledge-bases/{kb_id}/rename")
async def rename_kb(request: Request, kb_id: str, name: str = Form(...), description: str = Form(""), workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == ws_id).first()
    if not kb:
        return _render_documents_page(request, ws_id, db, error="Knowledge base not found or unauthorized.")

    clean_name = name.strip() if name else ""
    if not clean_name or len(clean_name) > 100:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Knowledge base name must be between 1 and 100 characters.")

    existing = db.query(KnowledgeBase).filter(
        KnowledgeBase.workspace_id == ws_id,
        KnowledgeBase.name.ilike(clean_name),
        KnowledgeBase.id != kb_id
    ).first()
    if existing:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Another knowledge base named '{clean_name}' already exists.")

    kb.name = clean_name
    kb.description = description.strip() if description else None
    db.commit()

    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message=f"Knowledge base renamed to '{clean_name}'.")

@router.post("/knowledge-bases/{kb_id}/delete")
async def delete_kb(request: Request, kb_id: str, confirm: str = Form(None), workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    if not confirm or confirm.lower() != "true":
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Explicit confirmation required to delete a knowledge base.")

    success = delete_knowledge_base_cascade(ws_id, kb_id, db)
    if not success:
        return _render_documents_page(request, ws_id, db, error="Knowledge base not found or unauthorized.")

    return _render_documents_page(request, ws_id, db, message="Knowledge base deleted successfully.")

@router.post("/knowledge-bases/{kb_id}/reindex")
async def reindex_kb(request: Request, kb_id: str, workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    try:
        reindex_knowledge_base(ws_id, kb_id, db)
    except Exception as e:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Reindexing failed: {str(e)}")

    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message="Knowledge base reindexed successfully.")

@router.post("/documents/upload")
async def upload_document(request: Request, workspace_id: str = Form(None), kb_id: str = Form(None), file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)

    # Determine target KB
    if kb_id:
        target_kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == ws_id).first()
        if not target_kb:
            return _render_documents_page(request, ws_id, db, error="Target knowledge base not found or unauthorized.")
    else:
        target_kb = get_or_create_default_kb(ws_id, db)
        kb_id = target_kb.id

    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Unsupported format: {ext}. Only PDF, TXT, and MD are supported.")

    content = bytearray()
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="File size exceeds the 10MB limit.")
        content.extend(chunk)

    if size == 0:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="File is empty.")

    if ext == ".pdf" and not content.startswith(b"%PDF-"):
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Malformed PDF file.")
    elif ext in [".txt", ".md"]:
        try:
            content.decode('utf-8')
        except UnicodeDecodeError:
            return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Text file must be valid UTF-8.")

    # Prevent duplicate uploads in this KB: replace safely if exists
    existing = db.query(Document).filter(
        Document.workspace_id == ws_id,
        Document.kb_id == kb_id,
        Document.filename == filename
    ).first()
    if existing:
        delete_document(ws_id, existing.id, db)

    os.makedirs(f"data/documents/{ws_id}", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = f"data/documents/{ws_id}/{file_id}{ext}"
    
    with open(file_path, "wb") as f:
        f.write(content)
        
    try:
        ingest_document(ws_id, file_path, filename, ext, db, kb_id=kb_id)
    except Exception as e:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Ingestion failed: {str(e)}")
    
    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message="Uploaded successfully.")

@router.post("/documents/{doc_id}/replace")
async def replace_doc(request: Request, doc_id: int, workspace_id: str = Form(None), file: UploadFile = File(...), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    doc = db.query(Document).filter(Document.id == doc_id, Document.workspace_id == ws_id).first()
    if not doc:
        return _render_documents_page(request, ws_id, db, error="Document not found or unauthorized.")

    kb_id = doc.kb_id
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Unsupported format: {ext}. Only PDF, TXT, and MD are supported.")

    content = bytearray()
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="File size exceeds the 10MB limit.")
        content.extend(chunk)

    if size == 0:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="File is empty.")

    if ext == ".pdf" and not content.startswith(b"%PDF-"):
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Malformed PDF file.")
    elif ext in [".txt", ".md"]:
        try:
            content.decode('utf-8')
        except UnicodeDecodeError:
            return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Text file must be valid UTF-8.")

    # Safe physical cleanup of previous file
    if doc.source_path and os.path.exists(doc.source_path):
        try:
            os.remove(doc.source_path)
        except OSError:
            pass

    # Clear old chunks
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

    os.makedirs(f"data/documents/{ws_id}", exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = f"data/documents/{ws_id}/{file_id}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)

    doc.filename = filename
    doc.doc_type = ext
    doc.source_path = file_path
    doc.ingestion_timestamp = datetime.utcnow()
    doc.status = "PROCESSING"
    doc.error_message = None
    db.commit()

    try:
        text = load_document(file_path, ext)
        raw_chunks = chunk_text(text)
        chunk_models = []
        for c in raw_chunks:
            chunk = Chunk(
                document_id=doc.id,
                text=c['text'],
                chunk_index=c['chunk_index'],
                page_number=c['page_number'],
                workspace_id=ws_id,
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
            if kb_id:
                faiss_store.add_vectors(get_kb_key(ws_id, kb_id), ids, vectors)
                bm25_store.fit(get_kb_key(ws_id, kb_id), ids, texts)
            faiss_store.add_vectors(ws_id, ids, vectors)
            bm25_store.fit(ws_id, ids, texts)

        doc.status = "READY"
        doc.error_message = None
        db.commit()
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message=f"Document replaced with '{filename}' successfully.")
    except Exception as e:
        db.query(Chunk).filter(Chunk.document_id == doc.id).delete()
        doc.status = "FAILED"
        doc.error_message = str(e)
        db.commit()
        if kb_id:
            _rebuild_kb_index(ws_id, kb_id, db)
        _rebuild_workspace_index(ws_id, db)
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Replacement failed: {str(e)}")

@router.post("/documents/{doc_id}/retry")
async def retry_doc(request: Request, doc_id: int, workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    doc = db.query(Document).filter(Document.id == doc_id, Document.workspace_id == ws_id).first()
    if not doc:
        return _render_documents_page(request, ws_id, db, error="Document not found or unauthorized.")

    kb_id = doc.kb_id
    try:
        success = retry_ingest_document(ws_id, doc_id, db)
        if not success:
            return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error="Retry failed: source file missing from disk.")
    except Exception as e:
        return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, error=f"Retry failed: {str(e)}")

    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message="Ingestion retried successfully.")

@router.post("/documents/reindex")
async def reindex_documents(request: Request, workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    reindex_workspace(ws_id, db)
    return _render_documents_page(request, ws_id, db, message="Reindexed successfully.")

@router.post("/documents/{doc_id}/delete")
async def delete_doc(request: Request, doc_id: int, workspace_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    doc = db.query(Document).filter(Document.id == doc_id, Document.workspace_id == ws_id).first()
    kb_id = doc.kb_id if doc else None
    success = delete_document(ws_id, doc_id, db)
    if not success:
        return _render_documents_page(request, ws_id, db, error="Document not found or unauthorized.")
    return _render_documents_page(request, ws_id, db, active_kb_id=kb_id, message="Deleted successfully.")
