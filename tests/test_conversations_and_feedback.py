import os
import uuid
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models.user import User
from app.models.workspace import Workspace
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query import QueryLog
from app.models.conversation import Conversation
from app.models.feedback import QueryFeedback
from app.core.config_manager import SessionLocal as TestingSessionLocal, config_manager
from app.rag.generator import generator
from app.evaluation.evaluator import evaluator
from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine
from app.experiment.experiment_engine import experiment_engine

client = TestClient(app)

def create_authenticated_user(username: str, password: str = "password123"):
    tc = TestClient(app)
    tc.post("/register", data={"username": username, "password": password})
    db = TestingSessionLocal()
    user = db.query(User).filter(User.username == username).first()
    ws_id = user.workspace_id
    db.close()
    return tc, user, ws_id

def create_kb_for_workspace(tc: TestClient, ws_id: str, name: str = "Test KB"):
    r = tc.post("/knowledge-bases", data={"name": name, "description": "KB for testing", "workspace_id": ws_id})
    assert r.status_code == 200
    db = TestingSessionLocal()
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id, KnowledgeBase.name == name).first()
    kb_id = kb.id
    db.close()
    return kb_id

def upload_test_doc(tc: TestClient, ws_id: str, kb_id: str, filename: str, content: bytes, content_type: str = "text/plain"):
    files = {"file": (filename, content, content_type)}
    r = tc.post("/documents/upload", data={"workspace_id": ws_id, "kb_id": kb_id}, files=files)
    assert r.status_code == 200
    db = TestingSessionLocal()
    doc = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb_id, Document.filename == filename).first()
    assert doc is not None
    doc_id = doc.id
    db.close()
    return doc_id


# --------------------------------------------------------------------------
# 1. A new conversation can be created
# --------------------------------------------------------------------------
def test_create_new_conversation():
    tc, user, ws_id = create_authenticated_user(f"conv_u_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Conv KB")

    r = tc.post(
        "/conversations",
        data={"title": "System Architecture Discussion", "kb_id": kb_id, "workspace_id": ws_id},
        headers={"Accept": "application/json"}
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["title"] == "System Architecture Discussion"
    assert data["workspace_id"] == ws_id
    assert data["kb_id"] == kb_id

    # Verify in DB
    db = TestingSessionLocal()
    conv = db.query(Conversation).filter(Conversation.id == data["id"]).first()
    assert conv is not None
    assert conv.user_id == user.id
    assert conv.workspace_id == ws_id
    db.close()


# --------------------------------------------------------------------------
# 2. Multiple queries persist in the intended conversation
# --------------------------------------------------------------------------
def test_multiple_queries_persist_in_conversation():
    tc, user, ws_id = create_authenticated_user(f"multi_u_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Multi Turn KB")
    upload_test_doc(tc, ws_id, kb_id, "protocols.txt", b"Protocol Alpha uses TLS 1.3 encryption. Protocol Beta uses AES 256.")

    # Create conversation
    r = tc.post(
        "/conversations",
        data={"title": "Security Protocols", "kb_id": kb_id, "workspace_id": ws_id},
        headers={"Accept": "application/json"}
    )
    conv_id = r.json()["id"]

    # First turn in standard mode
    with patch.object(generator, "generate", return_value={"answer": "Protocol Alpha uses TLS 1.3 [1]", "success": True, "error": None}):
        r1 = tc.post(
            "/query",
            data={"query": "What does Protocol Alpha use?", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["conversation_id"] == conv_id

    # Second turn in standard mode
    with patch.object(generator, "generate", return_value={"answer": "Protocol Beta uses AES 256 [1]", "success": True, "error": None}):
        r2 = tc.post(
            "/query",
            data={"query": "What does Protocol Beta use?", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["conversation_id"] == conv_id

    # Check conversation detail endpoint
    rc = tc.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    assert rc.status_code == 200
    cdata = rc.json()
    assert len(cdata["queries"]) == 2
    assert cdata["queries"][0]["query"] == "What does Protocol Alpha use?"
    assert cdata["queries"][1]["query"] == "What does Protocol Beta use?"


# --------------------------------------------------------------------------
# 3. Conversation history survives a fresh database session
# --------------------------------------------------------------------------
def test_conversation_history_survives_fresh_database_session():
    tc, user, ws_id = create_authenticated_user(f"persist_u_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Persist KB")
    upload_test_doc(tc, ws_id, kb_id, "doc.txt", b"Persistent doc content")

    # Create conversation
    rc = tc.post(
        "/conversations",
        data={"title": "Persistence Test", "kb_id": kb_id, "workspace_id": ws_id},
        headers={"Accept": "application/json"}
    )
    conv_id = rc.json()["id"]

    with patch.object(generator, "generate", return_value={"answer": "Persisted answer", "success": True, "error": None}):
        tc.post(
            "/query",
            data={"query": "Persistent question", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )

    # Fresh session verification
    fresh_db = TestingSessionLocal()
    try:
        conv_fresh = fresh_db.query(Conversation).filter(Conversation.id == conv_id).first()
        assert conv_fresh is not None
        assert conv_fresh.title == "Persistence Test"
        assert len(conv_fresh.queries) == 1
        assert conv_fresh.queries[0].query == "Persistent question"
        assert conv_fresh.queries[0].answer == "Persisted answer"
    finally:
        fresh_db.close()


# --------------------------------------------------------------------------
# 4. Users cannot read another user's conversations (Cross-workspace isolation)
# --------------------------------------------------------------------------
def test_cross_workspace_conversation_isolation():
    tc_a, user_a, ws_a = create_authenticated_user(f"user_a_{uuid.uuid4().hex[:6]}")
    tc_b, user_b, ws_b = create_authenticated_user(f"user_b_{uuid.uuid4().hex[:6]}")

    # User A creates a conversation
    r = tc_a.post(
        "/conversations",
        data={"title": "User A Private Conversation", "workspace_id": ws_a},
        headers={"Accept": "application/json"}
    )
    conv_a_id = r.json()["id"]

    # User B attempts to access User A's conversation
    rb = tc_b.get(f"/conversations/{conv_a_id}", headers={"Accept": "application/json"})
    assert rb.status_code == 404

    # User B attempts to list conversations; User A's conversation must not appear
    rlist_b = tc_b.get("/conversations", headers={"Accept": "application/json"})
    assert rlist_b.status_code == 200
    ids = [c["id"] for c in rlist_b.json()["conversations"]]
    assert conv_a_id not in ids


# --------------------------------------------------------------------------
# 5. Rename and delete enforce ownership
# --------------------------------------------------------------------------
def test_rename_and_delete_enforce_ownership():
    tc_owner, user_owner, ws_id = create_authenticated_user(f"owner_{uuid.uuid4().hex[:6]}")
    
    # Second user in different workspace
    tc_other, user_other, ws_other = create_authenticated_user(f"other_{uuid.uuid4().hex[:6]}")

    # Create conversation
    rc = tc_owner.post(
        "/conversations",
        data={"title": "Original Title", "workspace_id": ws_id},
        headers={"Accept": "application/json"}
    )
    conv_id = rc.json()["id"]

    # Other user attempts to rename
    r_bad_rename = tc_other.post(
        f"/conversations/{conv_id}/rename",
        data={"title": "Hijacked Title"},
        headers={"Accept": "application/json"}
    )
    assert r_bad_rename.status_code == 404

    # Owner renames successfully
    r_good_rename = tc_owner.post(
        f"/conversations/{conv_id}/rename",
        data={"title": "Updated Official Title"},
        headers={"Accept": "application/json"}
    )
    assert r_good_rename.status_code == 200
    assert r_good_rename.json()["title"] == "Updated Official Title"

    # Other user attempts to delete
    r_bad_del = tc_other.post(
        f"/conversations/{conv_id}/delete",
        headers={"Accept": "application/json"}
    )
    assert r_bad_del.status_code == 404

    # Owner deletes successfully
    r_good_del = tc_owner.post(
        f"/conversations/{conv_id}/delete",
        headers={"Accept": "application/json"}
    )
    assert r_good_del.status_code == 200

    # Ensure deleted from DB
    db = TestingSessionLocal()
    assert db.query(Conversation).filter(Conversation.id == conv_id).first() is None
    db.close()


# --------------------------------------------------------------------------
# 6. Invalid or missing conversation identifiers are handled safely
# --------------------------------------------------------------------------
def test_invalid_or_missing_conversation_identifiers_handled_safely():
    tc, user, ws_id = create_authenticated_user(f"safe_u_{uuid.uuid4().hex[:6]}")

    # Non-existent conversation detail
    r1 = tc.get("/conversations/non-existent-uuid-123", headers={"Accept": "application/json"})
    assert r1.status_code == 404

    # Non-existent conversation query submission
    r2 = tc.post(
        "/query",
        data={"query": "test question", "conversation_id": "non-existent-uuid-123", "workspace_id": ws_id},
        headers={"Accept": "application/json"}
    )
    assert r2.status_code == 404


# --------------------------------------------------------------------------
# 7. Legacy query records remain accessible after migration
# --------------------------------------------------------------------------
def test_legacy_query_records_remain_accessible():
    tc, user, ws_id = create_authenticated_user(f"legacy_u_{uuid.uuid4().hex[:6]}")

    # Create a query directly without conversation_id (legacy style)
    db = TestingSessionLocal()
    legacy_query = QueryLog(
        query="What is legacy execution?",
        answer="Legacy answer without conversation",
        latency=0.1,
        retrieval_config="{}",
        workspace_id=ws_id,
        conversation_id=None,
        mode="standard",
        answer_status="SUCCESS"
    )
    db.add(legacy_query)
    db.commit()
    q_id = legacy_query.id
    db.close()

    # Verify history page renders it cleanly without errors
    rh = tc.get("/history")
    assert rh.status_code == 200
    assert "What is legacy execution?" in rh.text

    # Verify list_conversations works without error
    rc = tc.get("/conversations", headers={"Accept": "application/json"})
    assert rc.status_code == 200


# --------------------------------------------------------------------------
# 8. Historical mode and citation metadata are preserved
# --------------------------------------------------------------------------
def test_historical_mode_and_citation_metadata_preserved():
    tc, user, ws_id = create_authenticated_user(f"meta_u_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Meta KB")
    doc_id = upload_test_doc(tc, ws_id, kb_id, "spec.txt", b"Specification 1.0 details.")

    # Create conversation
    rc = tc.post("/conversations", data={"title": "Meta Test", "kb_id": kb_id, "workspace_id": ws_id}, headers={"Accept": "application/json"})
    conv_id = rc.json()["id"]

    with patch.object(generator, "generate", return_value={"answer": "Spec 1.0 answer [1]", "success": True, "error": None}):
        r = tc.post(
            "/query",
            data={"query": "Tell me about spec 1.0", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200

    # Retrieve conversation detail
    r_detail = tc.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    assert r_detail.status_code == 200
    queries = r_detail.json()["queries"]
    assert len(queries) == 1
    q = queries[0]
    assert q["mode"] == "standard"
    assert q["answer_status"] == "SUCCESS"
    assert len(q["citations"]) > 0
    assert q["citations"][0]["filename"] == "spec.txt"
    assert q["citations"][0]["source_available"] is True


# --------------------------------------------------------------------------
# 9. Deleted source documents are not falsely presented as available
# --------------------------------------------------------------------------
def test_deleted_source_documents_handled_safely():
    tc, user, ws_id = create_authenticated_user(f"del_src_u_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Delete Source KB")
    doc_id = upload_test_doc(tc, ws_id, kb_id, "to_delete.txt", b"Temporary text to be removed.")

    rc = tc.post("/conversations", data={"title": "Deleted Source Test", "kb_id": kb_id, "workspace_id": ws_id}, headers={"Accept": "application/json"})
    conv_id = rc.json()["id"]

    with patch.object(generator, "generate", return_value={"answer": "Temp answer [1]", "success": True, "error": None}):
        rq = tc.post(
            "/query",
            data={"query": "Temporary query", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert rq.status_code == 200

    # Delete the document
    rdel = tc.post(f"/documents/{doc_id}/delete", data={"workspace_id": ws_id})
    assert rdel.status_code == 200

    # Retrieve conversation detail: citations must show source_available == False
    r_detail = tc.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    assert r_detail.status_code == 200
    queries = r_detail.json()["queries"]
    assert len(queries) == 1
    assert len(queries[0]["citations"]) > 0
    assert queries[0]["citations"][0]["source_available"] is False


# --------------------------------------------------------------------------
# 10. Feedback is persisted against the correct answer
# --------------------------------------------------------------------------
def test_feedback_persisted_against_correct_answer():
    tc, user, ws_id = create_authenticated_user(f"fb_u_{uuid.uuid4().hex[:6]}")

    db = TestingSessionLocal()
    q = QueryLog(query="Question 1", answer="Answer 1", workspace_id=ws_id, user_id=user.id, mode="standard", answer_status="SUCCESS")
    db.add(q)
    db.commit()
    q_id = q.id
    db.close()

    # Submit feedback
    rf = tc.post(
        f"/query/{q_id}/feedback",
        data={"rating": "helpful", "comment": "Clear and accurate response."},
        headers={"Accept": "application/json"}
    )
    assert rf.status_code == 200
    res = rf.json()
    assert res["status"] == "ok"
    assert res["feedback"]["rating"] == "helpful"
    assert res["feedback"]["comment"] == "Clear and accurate response."

    # Check GET feedback
    r_get = tc.get(f"/query/{q_id}/feedback", headers={"Accept": "application/json"})
    assert r_get.status_code == 200
    assert r_get.json()["feedback"]["rating"] == "helpful"


# --------------------------------------------------------------------------
# 11. Invalid feedback values are rejected with 400
# --------------------------------------------------------------------------
def test_invalid_feedback_values_rejected_with_400():
    tc, user, ws_id = create_authenticated_user(f"fb_inv_u_{uuid.uuid4().hex[:6]}")

    db = TestingSessionLocal()
    q = QueryLog(query="Question 1", answer="Answer 1", workspace_id=ws_id, user_id=user.id, mode="standard")
    db.add(q)
    db.commit()
    q_id = q.id
    db.close()

    # Invalid rating
    r1 = tc.post(f"/query/{q_id}/feedback", data={"rating": "stellar"})
    assert r1.status_code == 400

    # Invalid reason
    r2 = tc.post(f"/query/{q_id}/feedback", data={"rating": "not_helpful", "reason": "unsupported_reason"})
    assert r2.status_code == 400


# --------------------------------------------------------------------------
# 12. Unauthorized feedback access is blocked (cross-workspace)
# --------------------------------------------------------------------------
def test_unauthorized_feedback_access_blocked():
    tc_a, user_a, ws_a = create_authenticated_user(f"fb_owner_{uuid.uuid4().hex[:6]}")
    tc_b, user_b, ws_b = create_authenticated_user(f"fb_intruder_{uuid.uuid4().hex[:6]}")

    db = TestingSessionLocal()
    q = QueryLog(query="Private query", answer="Private answer", workspace_id=ws_a, user_id=user_a.id, mode="standard")
    db.add(q)
    db.commit()
    q_id = q.id
    db.close()

    # User B tries to submit feedback for User A's query
    r = tc_b.post(f"/query/{q_id}/feedback", data={"rating": "helpful"})
    assert r.status_code == 404


# --------------------------------------------------------------------------
# 13. Duplicate feedback submissions follow the documented update policy
# --------------------------------------------------------------------------
def test_duplicate_feedback_submission_updates_existing_record():
    tc, user, ws_id = create_authenticated_user(f"fb_upd_u_{uuid.uuid4().hex[:6]}")

    db = TestingSessionLocal()
    q = QueryLog(query="Question 1", answer="Answer 1", workspace_id=ws_id, user_id=user.id, mode="standard")
    db.add(q)
    db.commit()
    q_id = q.id
    db.close()

    # 1st submission: Helpful
    tc.post(f"/query/{q_id}/feedback", data={"rating": "helpful", "comment": "Good"})

    # 2nd submission: Updated to Not helpful with reason
    r_upd = tc.post(
        f"/query/{q_id}/feedback",
        data={"rating": "not_helpful", "reason": "incorrect_answer", "comment": "Actually flawed"},
        headers={"Accept": "application/json"}
    )
    assert r_upd.status_code == 200
    assert r_upd.json()["feedback"]["rating"] == "not_helpful"
    assert r_upd.json()["feedback"]["reason"] == "incorrect_answer"

    # Confirm in database that ONLY 1 record exists for this (query, user)
    db = TestingSessionLocal()
    records = db.query(QueryFeedback).filter(QueryFeedback.query_id == q_id, QueryFeedback.user_id == user.id).all()
    assert len(records) == 1
    assert records[0].rating == "not_helpful"
    assert records[0].comment == "Actually flawed"
    db.close()


# --------------------------------------------------------------------------
# 14. Standard and Adaptive query workflows continue to work inside conversations
# --------------------------------------------------------------------------
def test_standard_and_adaptive_modes_in_conversation():
    tc, user, ws_id = create_authenticated_user(f"modes_conv_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Mixed Modes KB")
    upload_test_doc(tc, ws_id, kb_id, "doc.txt", b"Authentication protocols and TLS.")

    rc = tc.post("/conversations", data={"title": "Mixed Modes Thread", "kb_id": kb_id, "workspace_id": ws_id}, headers={"Accept": "application/json"})
    conv_id = rc.json()["id"]

    # Turn 1: Standard Mode
    with patch.object(generator, "generate", return_value={"answer": "Standard answer [1]", "success": True, "error": None}):
        r1 = tc.post(
            "/query",
            data={"query": "Standard question", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert r1.status_code == 200
        assert r1.json()["mode"] == "standard"

    # Turn 2: Adaptive Mode
    baseline_eval = {
        "overall_score": 0.45,
        "faithfulness_score": 0.40,
        "relevance_score": 0.50,
        "context_score": 0.45,
        "completeness_score": 0.45,
        "failure_detected": True,
        "explanation": "Low retrieval relevance detected"
    }

    with patch.object(generator, "generate", return_value={"answer": "Adaptive answer [1]", "success": True, "error": None}), \
         patch.object(evaluator, "evaluate", return_value=baseline_eval), \
         patch.object(diagnostic_engine, "diagnose", return_value={"failure_type": "LOW_RETRIEVAL_RELEVANCE", "confidence": 0.9, "evidence": "Low scores", "explanation": "Scores below threshold"}), \
         patch.object(policy_engine, "recommend", return_value={"action": "INCREASE_TOP_K", "candidate_config": {"top_k": 7}, "reason": "Retrieve more chunks"}), \
         patch.object(experiment_engine, "run_experiment", return_value={
             "experiment_id": 101,
             "decision": "ACCEPT",
             "baseline_score": 0.45,
             "candidate_score": 0.88,
             "score_delta": 0.43,
             "baseline_latency": 0.4,
             "candidate_latency": 0.5,
             "latency_delta": 0.1,
             "candidate_result": {"answer": "Improved adaptive answer [1]", "answer_status": "SUCCESS", "retrieved_chunks": [], "citations": []},
             "candidate_eval": {"overall_score": 0.88, "failure_detected": False, "faithfulness_score": 0.88, "relevance_score": 0.88, "context_score": 0.88, "completeness_score": 0.88}
         }):

        r2 = tc.post(
            "/query",
            data={"query": "Adaptive question", "conversation_id": conv_id, "workspace_id": ws_id, "mode": "adaptive"},
            headers={"Accept": "application/json"}
        )
        assert r2.status_code == 200
        assert r2.json()["mode"] == "adaptive"
        assert r2.json()["adaptive_outcome"] == "ADAPTATION_ACCEPTED"

    # Verify both turns in conversation detail
    r_detail = tc.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    assert r_detail.status_code == 200
    queries = r_detail.json()["queries"]
    assert len(queries) == 2
    assert queries[0]["mode"] == "standard"
    assert queries[1]["mode"] == "adaptive"
    assert queries[1]["evaluation"] is not None
    assert queries[1]["evaluation"]["overall_score"] == 0.88
