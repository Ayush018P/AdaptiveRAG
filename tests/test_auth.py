import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.user import User
from app.models.workspace import Workspace
from app.core.config_manager import SessionLocal as TestingSessionLocal

client = TestClient(app)

def test_unauthenticated_access_redirects():
    # Attempting to access protected UI page redirects to login
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"

def test_user_registration_and_login():
    # Register
    response = client.post("/register", data={"username": "testuser", "password": "password123"}, follow_redirects=False)
    assert response.status_code == 302
    
    # Check that session was set (we should get a cookie)
    assert "session" in response.cookies
    
    # Now try to access homepage with the session
    response = client.get("/")
    assert response.status_code == 200
    assert b"Dashboard" in response.content
    assert b"AdaptiveRAG" in response.content

def test_cross_workspace_isolation():
    # Register user1
    client1 = TestClient(app)
    client1.post("/register", data={"username": "user1", "password": "password123"})
    
    # Register user2
    client2 = TestClient(app)
    client2.post("/register", data={"username": "user2", "password": "password123"})
    
    # user1 gets their own workspace
    db = TestingSessionLocal()
    u2 = db.query(User).filter(User.username == "user2").first()
    u2_workspace = u2.workspace_id
    
    # user1 tries to upload a document to user2's workspace
    # Should get 403 or fall back to their own workspace (in our implementation we raise 403)
    dummy_file = ("test.txt", b"dummy content", "text/plain")
    response = client1.post("/documents/upload", data={"workspace_id": u2_workspace}, files={"file": dummy_file})
    assert response.status_code == 403
    assert response.json()["detail"] == "Unauthorized workspace access"

    # User 1 tries to hit query API on user2's workspace
    response = client1.post("/query", data={"query": "test", "workspace_id": u2_workspace})
    assert response.status_code == 403
    assert response.json()["detail"] == "Unauthorized workspace access"

    # User 1 hits query API with their OWN workspace
    u1 = db.query(User).filter(User.username == "user1").first()
    u1_workspace = u1.workspace_id
    
    # Mock rag pipeline so it doesn't fail on missing chunks
    from unittest.mock import patch
    with patch("app.rag.pipeline.rag_pipeline.run") as mock_run:
        mock_run.return_value = {"answer": "mock answer", "context": [], "retrieved_chunks": []}
        with patch("app.evaluation.evaluator.evaluator.evaluate") as mock_eval:
            mock_eval.return_value = {
                "relevance_score": 1.0,
                "faithfulness_score": 1.0,
                "completeness_score": 1.0,
                "context_score": 1.0,
                "overall_score": 1.0,
                "failure_detected": False
            }
            response = client1.post("/query", data={"query": "test", "workspace_id": u1_workspace})
            assert response.status_code == 200

def test_invalid_or_missing_workspace():
    client1 = TestClient(app)
    client1.post("/register", data={"username": "user1", "password": "password123"})
    
    # Should automatically fallback to user's workspace
    response = client1.get("/documents")
    assert response.status_code == 200
