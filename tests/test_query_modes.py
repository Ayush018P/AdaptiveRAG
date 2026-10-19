import os
import uuid
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
from app.core.config_manager import SessionLocal as TestingSessionLocal, config_manager
from app.rag.hybrid_retriever import hybrid_retriever
from app.rag.pipeline import rag_pipeline
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
# 1. Standard mode returns grounded answer with verified citations
# --------------------------------------------------------------------------
def test_standard_mode_grounded_answer_and_citations():
    tc, user, ws_id = create_authenticated_user(f"std_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Standard Mode KB")
    
    upload_test_doc(
        tc, ws_id, kb_id, "policy.txt",
        b"Standard operating procedure requires encrypted database connections."
    )

    with patch.object(generator, "generate", return_value={
        "answer": "Encrypted database connections are required [1].",
        "success": True,
        "error": None,
        "llm_calls": 1
    }):
        r = tc.post(
            "/query",
            data={"query": "database connection policy", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200
        data = r.json()
        
        assert data["mode"] == "standard"
        assert data["answer_status"] == "SUCCESS"
        assert len(data["final"]["citations"]) > 0
        assert data["final"]["citations"][0]["filename"] == "policy.txt"
        assert "encrypted database connections" in data["final"]["citations"][0]["excerpt"].lower()
        
        # Verify adaptation block reflects standard mode without experiments
        assert data["adaptation"]["required"] is False
        assert data["adaptation"]["outcome"] == "STANDARD_MODE"
        assert data["adaptation"]["attempts"] == []
        assert data["final"]["evaluation"] is None


# --------------------------------------------------------------------------
# 2. Standard mode does NOT automatically launch adaptive experiments
# --------------------------------------------------------------------------
def test_standard_mode_does_not_launch_adaptive_experiments():
    tc, user, ws_id = create_authenticated_user(f"std_noexp_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "No Exp KB")
    
    upload_test_doc(tc, ws_id, kb_id, "sample.txt", b"System telemetry guidelines.")

    initial_config = config_manager.get_active_config(ws_id)

    with patch.object(generator, "generate", return_value={"answer": "Telemetry info [1]", "success": True, "error": None}), \
         patch.object(evaluator, "evaluate") as mock_eval, \
         patch.object(diagnostic_engine, "diagnose") as mock_diag, \
         patch.object(experiment_engine, "run_experiment") as mock_exp:
        
        r = tc.post("/query", data={"query": "telemetry", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"})
        assert r.status_code == 200
        
        # In Standard mode, evaluation, diagnosis, and experiments must NOT be called
        assert mock_eval.call_count == 0
        assert mock_diag.call_count == 0
        assert mock_exp.call_count == 0

        # Configuration must remain untouched
        assert config_manager.get_active_config(ws_id) == initial_config


# --------------------------------------------------------------------------
# 3. Adaptive mode invokes adaptive workflow on failure detection
# --------------------------------------------------------------------------
def test_adaptive_mode_invokes_adaptive_workflow():
    tc, user, ws_id = create_authenticated_user(f"adap_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Adaptive KB")
    
    upload_test_doc(tc, ws_id, kb_id, "doc.txt", b"Authentication protocols documentation.")

    baseline_eval = {
        "overall_score": 0.45,
        "faithfulness_score": 0.40,
        "relevance_score": 0.50,
        "context_score": 0.45,
        "completeness_score": 0.45,
        "failure_detected": True,
        "explanation": "Low retrieval relevance detected"
    }

    with patch.object(generator, "generate", return_value={"answer": "Auth answer [1]", "success": True, "error": None}), \
         patch.object(evaluator, "evaluate", return_value=baseline_eval) as mock_eval, \
         patch.object(diagnostic_engine, "diagnose", return_value={"failure_type": "LOW_RETRIEVAL_RELEVANCE", "confidence": 0.9, "evidence": "Low chunk scores", "explanation": "Scores below threshold"}) as mock_diag, \
         patch.object(policy_engine, "recommend", return_value={"action": "INCREASE_TOP_K", "candidate_config": {"top_k": 7}, "reason": "Retrieve more chunks to improve context"}) as mock_policy, \
         patch.object(experiment_engine, "run_experiment", return_value={
             "experiment_id": 99,
             "decision": "ACCEPT",
             "baseline_score": 0.45,
             "candidate_score": 0.85,
             "score_delta": 0.40,
             "baseline_latency": 0.5,
             "candidate_latency": 0.6,
             "latency_delta": 0.1,
             "candidate_result": {"answer": "Improved auth answer [1]", "answer_status": "SUCCESS", "retrieved_chunks": [], "citations": []},
             "candidate_eval": {"overall_score": 0.85, "failure_detected": False, "faithfulness_score": 0.85, "relevance_score": 0.85, "context_score": 0.85, "completeness_score": 0.85}
         }) as mock_exp:

        r = tc.post(
            "/query",
            data={"query": "authentication", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200
        data = r.json()
        
        assert data["mode"] == "adaptive"
        assert mock_eval.call_count >= 1
        assert mock_diag.call_count >= 1
        assert mock_exp.call_count >= 1
        assert data["adaptive_outcome"] == "ADAPTATION_ACCEPTED"
        assert len(data["adaptation"]["attempts"]) > 0


# --------------------------------------------------------------------------
# 4. Invalid mode values are rejected with 400
# --------------------------------------------------------------------------
def test_invalid_mode_values_rejected_with_400():
    tc, user, ws_id = create_authenticated_user(f"inv_user_{uuid.uuid4().hex[:6]}")
    
    # Reject "turbo"
    r1 = tc.post("/query", data={"query": "test query", "workspace_id": ws_id, "mode": "turbo"})
    assert r1.status_code == 400
    assert "Invalid mode 'turbo'" in r1.text or "Supported modes are: standard, adaptive" in r1.text

    # Reject "hyper"
    r2 = tc.post("/query", data={"query": "test query", "workspace_id": ws_id, "mode": "hyper"})
    assert r2.status_code == 400
    assert "Invalid mode 'hyper'" in r2.text


# --------------------------------------------------------------------------
# 5. Omitted mode defaults to "standard" for backward compatibility
# --------------------------------------------------------------------------
def test_omitted_mode_defaults_to_standard():
    tc, user, ws_id = create_authenticated_user(f"def_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Default Mode KB")
    upload_test_doc(tc, ws_id, kb_id, "def_doc.txt", b"Default mode verification content.")

    with patch.object(generator, "generate", return_value={"answer": "Default response [1]", "success": True, "error": None}):
        # Client omits mode completely (legacy request)
        r = tc.post(
            "/query",
            data={"query": "verification", "kb_id": kb_id, "workspace_id": ws_id},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "standard"
        assert data["adaptive_outcome"] == "STANDARD_MODE"

        # Check DB persistence of mode
        db = TestingSessionLocal()
        latest_log = db.query(QueryLog).filter(QueryLog.workspace_id == ws_id).order_by(QueryLog.id.desc()).first()
        assert latest_log is not None
        assert latest_log.mode == "standard"
        db.close()


# --------------------------------------------------------------------------
# 6. Both modes preserve workspace and knowledge-base isolation
# --------------------------------------------------------------------------
def test_workspace_and_kb_isolation_in_both_modes():
    tc_a, user_a, ws_a = create_authenticated_user(f"iso_a_{uuid.uuid4().hex[:6]}")
    kb_a = create_kb_for_workspace(tc_a, ws_a, "KB Alpha")

    tc_b, user_b, ws_b = create_authenticated_user(f"iso_b_{uuid.uuid4().hex[:6]}")
    kb_b = create_kb_for_workspace(tc_b, ws_b, "KB Beta")

    # Tenant B tries to query Tenant A's workspace in Standard mode -> 403
    r_std_hack = tc_b.post("/query", data={"query": "test", "workspace_id": ws_a, "mode": "standard"})
    assert r_std_hack.status_code == 403

    # Tenant B tries to query Tenant A's workspace in Adaptive mode -> 403
    r_adap_hack = tc_b.post("/query", data={"query": "test", "workspace_id": ws_a, "mode": "adaptive"})
    assert r_adap_hack.status_code == 403

    # Tenant B tries to pass Tenant A's KB in their own workspace -> 404
    r_kb_hack_std = tc_b.post("/query", data={"query": "test", "workspace_id": ws_b, "kb_id": kb_a, "mode": "standard"})
    assert r_kb_hack_std.status_code == 404

    r_kb_hack_adap = tc_b.post("/query", data={"query": "test", "workspace_id": ws_b, "kb_id": kb_a, "mode": "adaptive"})
    assert r_kb_hack_adap.status_code == 404


# --------------------------------------------------------------------------
# 7. Empty knowledge base state works in both modes
# --------------------------------------------------------------------------
def test_empty_kb_state_in_both_modes():
    tc, user, ws_id = create_authenticated_user(f"empty_modes_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Empty KB Modes")

    # Standard mode on empty KB
    r_std = tc.post("/query", data={"query": "search empty", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"})
    assert r_std.status_code == 200
    assert "Target Knowledge Base is Empty" in r_std.text or "Empty Knowledge Base" in r_std.text
    assert "Mode: Standard" in r_std.text

    # Adaptive mode on empty KB
    r_adap = tc.post("/query", data={"query": "search empty", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"})
    assert r_adap.status_code == 200
    assert "Target Knowledge Base is Empty" in r_adap.text or "Empty Knowledge Base" in r_adap.text
    assert "Mode: Adaptive" in r_adap.text
    assert "No Intervention Triggered" in r_adap.text


# --------------------------------------------------------------------------
# 8. No relevant chunks state works in both modes
# --------------------------------------------------------------------------
def test_no_relevant_chunks_in_both_modes():
    tc, user, ws_id = create_authenticated_user(f"norelev_modes_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "No Relev KB")
    upload_test_doc(tc, ws_id, kb_id, "simple.txt", b"Simple facts.")

    with patch.object(hybrid_retriever, "retrieve", return_value={"hybrid_results": []}):
        # Standard mode
        r_std = tc.post("/query", data={"query": "irrelevant quantum", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"})
        assert r_std.status_code == 200
        assert "Insufficient Retrieved Evidence" in r_std.text or "No Relevant Chunks" in r_std.text
        assert "Mode: Standard" in r_std.text

        # Adaptive mode
        r_adap = tc.post("/query", data={"query": "irrelevant quantum", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"})
        assert r_adap.status_code == 200
        assert "Insufficient Retrieved Evidence" in r_adap.text or "No Relevant Chunks" in r_adap.text
        assert "Mode: Adaptive" in r_adap.text
        assert "No Intervention Triggered" in r_adap.text


# --------------------------------------------------------------------------
# 9. Provider failure reported honestly in both modes without fake fallback
# --------------------------------------------------------------------------
def test_provider_failure_in_both_modes():
    tc, user, ws_id = create_authenticated_user(f"prov_fail_modes_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Prov Fail KB")
    upload_test_doc(tc, ws_id, kb_id, "service.txt", b"Microservice deployment checklist.")

    with patch.object(generator, "generate", return_value={
        "answer": "Failed to generate answer due to LLM error.",
        "success": False,
        "error": "HTTP 429 Too Many Requests: Rate limit exceeded",
        "llm_calls": 3
    }):
        # Standard mode
        r_std = tc.post("/query", data={"query": "checklist", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"})
        assert r_std.status_code == 200
        assert "Generation Provider Failure" in r_std.text or "LLM Provider Error" in r_std.text
        assert "HTTP 429" in r_std.text
        assert "Mode: Standard" in r_std.text

        # Adaptive mode
        r_adap = tc.post("/query", data={"query": "checklist", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"})
        assert r_adap.status_code == 200
        assert "Generation Provider Failure" in r_adap.text or "LLM Provider Error" in r_adap.text
        assert "HTTP 429" in r_adap.text
        assert "Mode: Adaptive" in r_adap.text


# --------------------------------------------------------------------------
# 10. Adaptive rejected experiment preserves baseline config
# --------------------------------------------------------------------------
def test_adaptive_rejected_experiment_preserves_baseline_config():
    tc, user, ws_id = create_authenticated_user(f"exp_rej_modes_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Rej Config KB")
    upload_test_doc(tc, ws_id, kb_id, "sample.txt", b"Sample documentation.")

    initial_config = config_manager.get_active_config(ws_id)
    baseline_top_k = initial_config["top_k"]

    baseline_eval = {
        "overall_score": 0.55,
        "faithfulness_score": 0.50,
        "relevance_score": 0.60,
        "context_score": 0.55,
        "completeness_score": 0.55,
        "failure_detected": True
    }
    candidate_eval = {
        "overall_score": 0.35,  # Worse
        "faithfulness_score": 0.30,
        "relevance_score": 0.40,
        "context_score": 0.35,
        "completeness_score": 0.35,
        "failure_detected": True
    }

    with patch.object(generator, "generate", return_value={"answer": "Ans [1]", "success": True, "error": None}), \
         patch.object(evaluator, "evaluate", side_effect=[baseline_eval, candidate_eval]):

        r = tc.post(
            "/query",
            data={"query": "test query", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200
        data = r.json()
        assert data["adaptive_outcome"] == "ADAPTATION_REJECTED"

        # Active configuration must remain baseline
        current_config = config_manager.get_active_config(ws_id)
        assert current_config["top_k"] == baseline_top_k


# --------------------------------------------------------------------------
# 11. Query mode persistence and history display
# --------------------------------------------------------------------------
def test_query_mode_persistence_and_history():
    tc, user, ws_id = create_authenticated_user(f"hist_mode_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Hist KB")
    upload_test_doc(tc, ws_id, kb_id, "history_doc.txt", b"History testing document.")

    with patch.object(generator, "generate", return_value={"answer": "History Ans [1]", "success": True, "error": None}):
        # Run standard query
        r1 = tc.post("/query", data={"query": "first query", "kb_id": kb_id, "workspace_id": ws_id, "mode": "standard"})
        assert r1.status_code == 200

        # Run adaptive query
        with patch.object(evaluator, "evaluate", return_value={"overall_score": 0.9, "failure_detected": False, "faithfulness_score": 0.9, "relevance_score": 0.9, "context_score": 0.9, "completeness_score": 0.9, "explanation": "Good"}):
            r2 = tc.post("/query", data={"query": "second query", "kb_id": kb_id, "workspace_id": ws_id, "mode": "adaptive"})
            assert r2.status_code == 200

    # Inspect GET /history
    r_hist = tc.get(f"/history?workspace_id={ws_id}")
    assert r_hist.status_code == 200
    assert "Standard" in r_hist.text
    assert "Adaptive" in r_hist.text


# --------------------------------------------------------------------------
# 12. Mode selector preserves selection in GET /query
# --------------------------------------------------------------------------
def test_mode_selector_preserves_selection_in_get_query():
    tc, user, ws_id = create_authenticated_user(f"sel_pres_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Selection KB")

    # Request with mode=adaptive
    r = tc.get(f"/query?kb_id={kb_id}&mode=adaptive")
    assert r.status_code == 200
    assert 'checked' in r.text
    assert 'id="modeAdaptive"' in r.text
    assert f'value="{kb_id}"' in r.text
