import pytest
import os
from fastapi.testclient import TestClient
from app.main import app
from app.models.document import Document
from app.models.chunk import Chunk
from app.core.config_manager import SessionLocal as TestingSessionLocal
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store

client = TestClient(app)

import uuid as uuid_lib

def setup_user_and_workspace(username=None, password="password123"):
    if username is None:
        username = f"u_{uuid_lib.uuid4().hex[:8]}"
    client.post("/register", data={"username": username, "password": password})
    client.post("/login", data={"username": username, "password": password})
    db = TestingSessionLocal()
    from app.models.user import User
    user = db.query(User).filter(User.username == username).first()
    ws_id = user.workspace_id
    db.close()
    return ws_id

def setup_user2_and_workspace():
    return setup_user_and_workspace(username=f"u_{uuid_lib.uuid4().hex[:8]}", password="password456")

def test_valid_txt_upload():
    ws_id = setup_user_and_workspace()
    
    file_content = b"This is a valid text file."
    files = {"file": ("test.txt", file_content, "text/plain")}
    data = {"workspace_id": ws_id}
    
    r = client.post("/documents/upload", data=data, files=files)
    assert r.status_code == 200
    assert "Uploaded successfully" in r.text
    
    db = TestingSessionLocal()
    docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
    assert len(docs) == 1
    assert docs[0].status == "READY"
    db.close()

def test_unsupported_extension():
    ws_id = setup_user_and_workspace()
    
    file_content = b"Some content"
    files = {"file": ("test.exe", file_content, "text/plain")}
    data = {"workspace_id": ws_id}
    
    r = client.post("/documents/upload", data=data, files=files)
    assert r.status_code == 200
    assert "Unsupported format" in r.text
    
    db = TestingSessionLocal()
    docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
    assert len(docs) == 0
    db.close()

def test_empty_file():
    ws_id = setup_user_and_workspace()
    
    files = {"file": ("empty.txt", b"", "text/plain")}
    data = {"workspace_id": ws_id}
    
    r = client.post("/documents/upload", data=data, files=files)
    assert "File is empty" in r.text

def test_malformed_pdf():
    ws_id = setup_user_and_workspace()
    
    files = {"file": ("fake.pdf", b"Not a real PDF", "application/pdf")}
    data = {"workspace_id": ws_id}
    
    r = client.post("/documents/upload", data=data, files=files)
    assert "Malformed PDF file" in r.text

def test_duplicate_upload_replaces():
    ws_id = setup_user_and_workspace()
    
    files1 = {"file": ("test.txt", b"First version.", "text/plain")}
    data = {"workspace_id": ws_id}
    client.post("/documents/upload", data=data, files=files1)
    
    db = TestingSessionLocal()
    docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
    assert len(docs) == 1
    doc_id_1 = docs[0].id
    db.close()
    
    files2 = {"file": ("test.txt", b"Second version replacement.", "text/plain")}
    client.post("/documents/upload", data=data, files=files2)
    
    db = TestingSessionLocal()
    docs = db.query(Document).filter(Document.workspace_id == ws_id).all()
    assert len(docs) == 1
    doc_id_2 = docs[0].id
    
    chunks = db.query(Chunk).filter(Chunk.workspace_id == ws_id).all()
    assert all(c.document_id == doc_id_2 for c in chunks)
    assert len(chunks) > 0
    assert "Second version replacement." in chunks[0].text
    db.close()

def test_cross_workspace_upload():
    ws_id1 = setup_user_and_workspace()
    ws_id2 = setup_user2_and_workspace()
    
    files = {"file": ("test.txt", b"Hacked", "text/plain")}
    data = {"workspace_id": ws_id1}
    r = client.post("/documents/upload", data=data, files=files)
    
    assert r.status_code == 403

def test_delete_document():
    ws_id = setup_user_and_workspace(username="u_del", password="password123")
    
    files = {"file": ("test.txt", b"To be deleted.", "text/plain")}
    data = {"workspace_id": ws_id}
    client.post("/documents/upload", data=data, files=files)
    
    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id).first()
    assert doc is not None
    db.close()
    
    r = client.post(f"/documents/{doc.id}/delete", data=data)
    assert "Deleted successfully" in r.text
    
    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id).first()
    assert doc is None
    chunks = db.query(Chunk).filter(Chunk.workspace_id == ws_id).all()
    assert len(chunks) == 0
    db.close()

def test_reindex_failure_preserves_index(monkeypatch):
    ws_id = setup_user_and_workspace()
    
    files = {"file": ("test.txt", b"Content.", "text/plain")}
    data = {"workspace_id": ws_id}
    client.post("/documents/upload", data=data, files=files)
    
    def mock_embed(*args, **kwargs):
        raise ValueError("Simulated embedding failure")
        
    import app.rag.ingestion as ing
    monkeypatch.setattr(ing.embedding_service, "embed_documents", mock_embed)
    
    try:
        client.post("/documents/reindex", data=data)
    except Exception:
        pass
    
    assert ws_id in faiss_store.indices
    assert faiss_store.indices[ws_id].ntotal > 0
    assert ws_id in bm25_store.corpus
    assert len(bm25_store.corpus[ws_id]) > 0

def test_size_limit_rejection():
    ws_id = setup_user_and_workspace()
    
    content = b"a" * (11 * 1024 * 1024)
    files = {"file": ("large.txt", content, "text/plain")}
    data = {"workspace_id": ws_id}
    
    r = client.post("/documents/upload", data=data, files=files)
    assert "exceeds the 10MB limit" in r.text
