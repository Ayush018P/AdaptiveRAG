import os
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.user import User
from app.models.workspace import Workspace
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.chunk import Chunk
from app.core.config_manager import SessionLocal as TestingSessionLocal
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
from app.rag.hybrid_retriever import hybrid_retriever
from app.rag.ingestion import ingest_document, reindex_knowledge_base, delete_document, retry_ingest_document

client = TestClient(app)

def create_authenticated_user(username: str, password: str = "password123"):
    tc = TestClient(app)
    tc.post("/register", data={"username": username, "password": password})
    db = TestingSessionLocal()
    user = db.query(User).filter(User.username == username).first()
    ws_id = user.workspace_id
    db.close()
    return tc, user, ws_id

def test_kb_creation_listing_and_renaming():
    tc, user, ws_id = create_authenticated_user(f"kb_user_{uuid.uuid4().hex[:6]}")
    
    # 1. Create Knowledge Base with valid name
    r = tc.post("/knowledge-bases", data={"name": "Engineering Docs", "description": "Internal engineering specs", "workspace_id": ws_id})
    assert r.status_code == 200
    assert "Engineering Docs" in r.text
    assert "created successfully" in r.text

    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Engineering Docs").first()
    assert kb is not None
    assert kb.description == "Internal engineering specs"
    kb_id = kb.id
    db.close()

    # 2. Reject empty name
    r_empty = tc.post("/knowledge-bases", data={"name": "   ", "description": "", "workspace_id": ws_id})
    assert "between 1 and 100 characters" in r_empty.text

    # 3. Reject duplicate name
    r_dup = tc.post("/knowledge-bases", data={"name": "engineering docs", "description": "", "workspace_id": ws_id})
    assert "already exists in this workspace" in r_dup.text

    # 4. List KBs via GET /documents
    r_list = tc.get(f"/documents?workspace_id={ws_id}")
    assert r_list.status_code == 200
    assert "Engineering Docs" in r_list.text

    # 5. Rename Knowledge Base
    r_rename = tc.post(f"/knowledge-bases/{kb_id}/rename", data={"name": "Architecture Docs", "description": "Updated", "workspace_id": ws_id})
    assert r_rename.status_code == 200
    assert "Architecture Docs" in r_rename.text
    assert "renamed to &#39;Architecture Docs&#39;" in r_rename.text or "renamed" in r_rename.text

    db = TestingSessionLocal()
    renamed_kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    assert renamed_kb.name == "Architecture Docs"
    db.close()

def test_kb_deletion_and_cascade_cleanup():
    tc, user, ws_id = create_authenticated_user(f"kb_del_{uuid.uuid4().hex[:6]}")
    
    # Create KB
    tc.post("/knowledge-bases", data={"name": "Temporary KB", "description": "To be deleted", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Temporary KB").first()
    kb_id = kb.id
    db.close()

    # Upload document into Temporary KB
    test_content = b"Temporary content for deletion test."
    files = {"file": ("temp_doc.txt", test_content, "text/plain")}
    r_up = tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files=files)
    assert r_up.status_code == 200

    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb_id).first()
    assert doc is not None
    doc_path = doc.source_path
    doc_id = doc.id
    assert os.path.exists(doc_path)
    chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
    assert len(chunks) > 0
    db.close()

    # Index has vectors
    kb_key = f"{ws_id}:{kb_id}"
    assert kb_key in faiss_store.indices

    # Try delete without confirmation
    r_no_conf = tc.post(f"/knowledge-bases/{kb_id}/delete", data={"workspace_id": ws_id})
    assert "Explicit confirmation required" in r_no_conf.text

    # Delete with explicit confirmation
    r_del = tc.post(f"/knowledge-bases/{kb_id}/delete", data={"workspace_id": ws_id, "confirm": "true"})
    assert "deleted successfully" in r_del.text

    # Verify cascading cleanup
    db = TestingSessionLocal()
    assert db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first() is None
    assert db.query(Document).filter(Document.id == doc_id).first() is None
    assert db.query(Chunk).filter(Chunk.document_id == doc_id).count() == 0
    db.close()

    # Physical file removed
    assert not os.path.exists(doc_path)

    # Retrieval index reset
    assert kb_key not in faiss_store.indices or faiss_store.indices[kb_key].ntotal == 0

def test_document_upload_and_scoped_indexing():
    tc, user, ws_id = create_authenticated_user(f"kb_up_{uuid.uuid4().hex[:6]}")
    
    # Create KB
    tc.post("/knowledge-bases", data={"name": "Research Notes", "description": "", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Research Notes").first()
    kb_id = kb.id
    db.close()

    # Upload supported file
    content = b"AdaptiveRAG optimizes retrieval dynamically using closed loop evaluation."
    files = {"file": ("research.txt", content, "text/plain")}
    r = tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files=files)
    assert r.status_code == 200
    assert "Uploaded successfully" in r.text

    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb_id, Document.filename == "research.txt").first()
    assert doc is not None
    assert doc.status == "READY"
    assert doc.error_message is None
    assert len(doc.chunks) > 0
    db.close()

def test_document_replace_and_delete_consistency():
    tc, user, ws_id = create_authenticated_user(f"kb_repl_{uuid.uuid4().hex[:6]}")
    
    tc.post("/knowledge-bases", data={"name": "KB Replace Test", "description": "", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "KB Replace Test").first()
    kb_id = kb.id
    db.close()

    # Upload initial doc
    files = {"file": ("changelog.txt", b"Version 1.0 Alpha release with legacy features.", "text/plain")}
    tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files=files)

    # Verify retrievable
    res = hybrid_retriever.retrieve(ws_id, "Alpha release", kb_id=kb_id)
    assert len(res["hybrid_results"]) > 0

    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb_id).first()
    doc_id = doc.id
    old_chunks = [c.text for c in doc.chunks]
    db.close()

    # Explicitly Replace Document
    new_files = {"file": ("changelog_v2.txt", b"Version 2.0 Quantum release with zero legacy code.", "text/plain")}
    r_replace = tc.post(f"/documents/{doc_id}/replace", data={"workspace_id": ws_id}, files=new_files)
    assert r_replace.status_code == 200
    assert "Document replaced" in r_replace.text

    # Verify old content is no longer in retrievable chunks
    db = TestingSessionLocal()
    doc_after = db.query(Document).filter(Document.id == doc_id).first()
    assert doc_after.filename == "changelog_v2.txt"
    assert "Alpha release" not in [c.text for c in doc_after.chunks][0]
    assert "Quantum release" in [c.text for c in doc_after.chunks][0]
    db.close()

    # Delete Document
    r_del = tc.post(f"/documents/{doc_id}/delete", data={"workspace_id": ws_id})
    assert "Deleted successfully" in r_del.text

    # Subsequent retrieval must return empty
    res_after = hybrid_retriever.retrieve(ws_id, "Quantum release", kb_id=kb_id)
    assert len(res_after["hybrid_results"]) == 0

def test_retry_failed_ingestion():
    tc, user, ws_id = create_authenticated_user(f"kb_retry_{uuid.uuid4().hex[:6]}")
    
    tc.post("/knowledge-bases", data={"name": "Retry KB", "description": "", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Retry KB").first()
    kb_id = kb.id
    db.close()

    # Ingest document while mocking embedding failure
    files = {"file": ("retry_test.txt", b"Important document that will fail embedding initially.", "text/plain")}
    with patch("app.rag.ingestion.embedding_service.embed_documents", side_effect=RuntimeError("GPU out of memory")):
        r = tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files=files)
        assert "Ingestion failed" in r.text

    db = TestingSessionLocal()
    failed_doc = db.query(Document).filter(Document.workspace_id == ws_id, Document.filename == "retry_test.txt").first()
    assert failed_doc is not None
    assert failed_doc.status == "FAILED"
    assert "GPU out of memory" in failed_doc.error_message
    doc_id = failed_doc.id
    db.close()

    # Retry ingestion without the mock failure
    r_retry = tc.post(f"/documents/{doc_id}/retry", data={"workspace_id": ws_id})
    assert r_retry.status_code == 200
    assert "retried successfully" in r_retry.text

    db = TestingSessionLocal()
    recovered_doc = db.query(Document).filter(Document.id == doc_id).first()
    assert recovered_doc.status == "READY"
    assert recovered_doc.error_message is None
    assert len(recovered_doc.chunks) > 0
    db.close()

def test_reindex_only_intended_knowledge_base():
    tc, user, ws_id = create_authenticated_user(f"kb_reindex_{uuid.uuid4().hex[:6]}")
    
    tc.post("/knowledge-bases", data={"name": "KB Alpha", "description": "", "workspace_id": ws_id})
    tc.post("/knowledge-bases", data={"name": "KB Beta", "description": "", "workspace_id": ws_id})
    
    db = TestingSessionLocal()
    kb_a = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "KB Alpha").first()
    kb_b = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "KB Beta").first()
    kb_a_id = kb_a.id
    kb_b_id = kb_b.id
    db.close()

    tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_a_id}, files={"file": ("a.txt", b"Content for alpha knowledge base.", "text/plain")})
    tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_b_id}, files={"file": ("b.txt", b"Content for beta knowledge base.", "text/plain")})

    key_a = f"{ws_id}:{kb_a_id}"
    key_b = f"{ws_id}:{kb_b_id}"

    # Record Beta's index object
    beta_index_before = faiss_store.indices.get(key_b)

    # Reindex only KB Alpha
    r_reindex = tc.post(f"/knowledge-bases/{kb_a_id}/reindex", data={"workspace_id": ws_id})
    assert r_reindex.status_code == 200
    assert "Knowledge base reindexed successfully" in r_reindex.text

    # KB Beta's index instance in memory must remain strictly identical (untouched!)
    beta_index_after = faiss_store.indices.get(key_b)
    assert beta_index_before is beta_index_after

def test_empty_kb_never_falls_back_to_global():
    tc, user, ws_id = create_authenticated_user(f"kb_empty_{uuid.uuid4().hex[:6]}")
    
    # Create empty KB
    tc.post("/knowledge-bases", data={"name": "Empty KB", "description": "", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Empty KB").first()
    empty_kb_id = kb.id
    db.close()

    # Search in empty KB
    res = hybrid_retriever.retrieve(ws_id, "some common query", kb_id=empty_kb_id)
    assert res["hybrid_results"] == []
    assert res["dense_results"] == []
    assert res["bm25_results"] == []

def test_cross_workspace_kb_authorization_prevention():
    tc1, user1, ws1 = create_authenticated_user(f"user_sec_1_{uuid.uuid4().hex[:6]}")
    tc2, user2, ws2 = create_authenticated_user(f"user_sec_2_{uuid.uuid4().hex[:6]}")

    # User 1 creates a KB
    tc1.post("/knowledge-bases", data={"name": "Confidential KB", "description": "", "workspace_id": ws1})
    db = TestingSessionLocal()
    kb1 = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws1).first()
    kb1_id = kb1.id
    db.close()

    # User 2 tries to rename User 1's KB
    r_rename = tc2.post(f"/knowledge-bases/{kb1_id}/rename", data={"name": "Hacked KB", "workspace_id": ws2})
    assert "Knowledge base not found or unauthorized" in r_rename.text

    # User 2 tries to delete User 1's KB
    r_del = tc2.post(f"/knowledge-bases/{kb1_id}/delete", data={"workspace_id": ws2, "confirm": "true"})
    assert "Knowledge base not found or unauthorized" in r_del.text

    # User 2 tries to upload to User 1's KB
    files = {"file": ("exploit.txt", b"malicious content", "text/plain")}
    r_upload = tc2.post("/documents/upload", data={"workspace_id": ws2, "kb_id": kb1_id}, files=files)
    assert "Target knowledge base not found or unauthorized" in r_upload.text

def test_malformed_and_oversized_file_handling():
    tc, user, ws_id = create_authenticated_user(f"kb_val_{uuid.uuid4().hex[:6]}")

    # 1. Malformed PDF
    bad_pdf = ("fake.pdf", b"NOT_A_PDF_HEADER", "application/pdf")
    r1 = tc.post("/documents/upload", data={"workspace_id": ws_id}, files={"file": bad_pdf})
    assert "Malformed PDF file." in r1.text

    # 2. Invalid UTF-8 text file
    bad_txt = ("bad.txt", b"\xff\xfe\x00\x00invalid_utf8", "text/plain")
    r2 = tc.post("/documents/upload", data={"workspace_id": ws_id}, files={"file": bad_txt})
    assert "Text file must be valid UTF-8." in r2.text

    # 3. Empty file
    empty_file = ("empty.txt", b"", "text/plain")
    r3 = tc.post("/documents/upload", data={"workspace_id": ws_id}, files={"file": empty_file})
    assert "File is empty." in r3.text

def test_legacy_data_migration_preservation():
    from scripts.migrate_knowledge_bases import migrate
    # Running migration should be safe and idempotent
    migrate()
    
    db = TestingSessionLocal()
    # Check that any existing documents in DB have a valid kb_id
    docs = db.query(Document).all()
    for d in docs:
        assert d.kb_id is not None
    db.close()

def test_reindex_failure_preserves_kb_index():
    tc, user, ws_id = create_authenticated_user(f"kb_fail_{uuid.uuid4().hex[:6]}")
    tc.post("/knowledge-bases", data={"name": "Safe Reindex KB", "description": "", "workspace_id": ws_id})
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == "Safe Reindex KB").first()
    kb_id = kb.id
    db.close()

    # Upload valid document
    tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files={"file": ("stable.txt", b"Stable knowledge base content.", "text/plain")})
    kb_key = f"{ws_id}:{kb_id}"
    assert kb_key in faiss_store.indices
    initial_count = faiss_store.indices[kb_key].ntotal
    assert initial_count > 0

    # Simulate failure during reindex embedding
    with patch("app.rag.ingestion.embedding_service.embed_documents", side_effect=ValueError("Simulated reindexing failure")):
        try:
            tc.post(f"/knowledge-bases/{kb_id}/reindex", data={"workspace_id": ws_id})
        except Exception:
            pass

    # The existing index must remain intact with its original vectors!
    assert kb_key in faiss_store.indices
    assert faiss_store.indices[kb_key].ntotal == initial_count

