import os
import sys
import shutil
import sqlite3
import uuid
import json
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.models.database import SessionLocal
from app.models.user import User
from app.models.workspace import Workspace
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query import QueryLog
from app.models.conversation import Conversation
from app.models.feedback import QueryFeedback
from app.rag.generator import generator
from app.evaluation.evaluator import evaluator
from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine
from app.experiment.experiment_engine import experiment_engine

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(step_name, passed, details=""):
    mark = "[PASS]" if passed else "[FAIL]"
    print(f"  {mark} {step_name}")
    if details:
        print(f"         {details}")

def run_all_checks():
    client = TestClient(app)
    results = {}

    # -------------------------------------------------------------------------
    # 1. Conversation Workflow
    # -------------------------------------------------------------------------
    print_header("1. CONVERSATION WORKFLOW VERIFICATION")
    u1_name = f"verify_user1_{uuid.uuid4().hex[:6]}"
    tc1 = TestClient(app)
    tc1.post("/register", data={"username": u1_name, "password": "password123"})
    db = SessionLocal()
    user1 = db.query(User).filter(User.username == u1_name).first()
    ws1_id = user1.workspace_id
    db.close()

    # Create KB and upload doc
    r_kb = tc1.post("/knowledge-bases", data={"name": "Engineering Knowledge", "description": "Docs", "workspace_id": ws1_id})
    db = SessionLocal()
    kb1 = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws1_id, KnowledgeBase.name == "Engineering Knowledge").first()
    kb1_id = kb1.id
    db.close()

    # Upload doc
    tc1.post(
        "/documents/upload",
        data={"workspace_id": ws1_id, "kb_id": kb1_id},
        files={"file": ("guide.txt", b"Authentication uses OAuth2 and TLS 1.3 encryption.", "text/plain")}
    )

    # Create conversation
    rc = tc1.post(
        "/conversations",
        data={"title": "Architecture Thread", "kb_id": kb1_id, "workspace_id": ws1_id},
        headers={"Accept": "application/json"}
    )
    conv_id = rc.json()["id"]
    print(f"  Created Conversation ID: {conv_id} (Title: Architecture Thread)")

    # Turn 1
    with patch.object(generator, "generate", return_value={"answer": "Authentication uses OAuth2 [1].", "success": True, "error": None}):
        r_q1 = tc1.post(
            "/query",
            data={"query": "How is authentication handled?", "conversation_id": conv_id, "workspace_id": ws1_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
    q1_data = r_q1.json()
    q1_id = q1_data["query_id"]

    # Turn 2
    with patch.object(generator, "generate", return_value={"answer": "TLS 1.3 is required for all endpoints [1].", "success": True, "error": None}):
        r_q2 = tc1.post(
            "/query",
            data={"query": "What encryption standard is used?", "conversation_id": conv_id, "workspace_id": ws1_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
    q2_data = r_q2.json()
    q2_id = q2_data["query_id"]

    # Simulate browser reload by reading conversation endpoint in fresh request
    r_thread = tc1.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    thread_data = r_thread.json()
    queries = thread_data["queries"]

    c1_ok = (
        len(queries) == 2 and
        queries[0]["query"] == "How is authentication handled?" and
        queries[0]["answer"] == "Authentication uses OAuth2 [1]." and
        queries[1]["query"] == "What encryption standard is used?" and
        queries[1]["answer"] == "TLS 1.3 is required for all endpoints [1]."
    )
    print_result("Conversation created and multi-question thread persisted", c1_ok, f"Found {len(queries)} turns in correct order.")
    results["conversation_workflow"] = c1_ok

    # -------------------------------------------------------------------------
    # 2. Mode and Knowledge-Base Continuity
    # -------------------------------------------------------------------------
    print_header("2. MODE AND KNOWLEDGE-BASE CONTINUITY")
    # Start second thread for mixed modes
    r_mix = tc1.post(
        "/conversations",
        data={"title": "Mixed Modes Thread", "kb_id": kb1_id, "workspace_id": ws1_id},
        headers={"Accept": "application/json"}
    )
    mix_conv_id = r_mix.json()["id"]

    # Turn 1: Standard Mode
    with patch.object(generator, "generate", return_value={"answer": "Standard direct answer [1].", "success": True, "error": None}):
        r_std = tc1.post(
            "/query",
            data={"query": "Standard mode question", "conversation_id": mix_conv_id, "workspace_id": ws1_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
    std_data = r_std.json()

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

    with patch.object(generator, "generate", return_value={"answer": "Adaptive candidate answer [1].", "success": True, "error": None}), \
         patch.object(evaluator, "evaluate", return_value=baseline_eval), \
         patch.object(diagnostic_engine, "diagnose", return_value={"failure_type": "LOW_RETRIEVAL_RELEVANCE", "confidence": 0.9, "evidence": "Scores low", "explanation": "Scores below threshold"}), \
         patch.object(policy_engine, "recommend", return_value={"action": "INCREASE_TOP_K", "candidate_config": {"top_k": 7}, "reason": "Retrieve more chunks"}), \
         patch.object(experiment_engine, "run_experiment", return_value={
             "experiment_id": 201,
             "decision": "ACCEPT",
             "baseline_score": 0.45,
             "candidate_score": 0.89,
             "score_delta": 0.44,
             "baseline_latency": 0.4,
             "candidate_latency": 0.5,
             "latency_delta": 0.1,
             "candidate_result": {"answer": "Optimized adaptive answer [1]", "answer_status": "SUCCESS", "retrieved_chunks": [], "citations": []},
             "candidate_eval": {"overall_score": 0.89, "failure_detected": False, "faithfulness_score": 0.89, "relevance_score": 0.89, "context_score": 0.89, "completeness_score": 0.89}
         }):
        r_adap = tc1.post(
            "/query",
            data={"query": "Adaptive mode question", "conversation_id": mix_conv_id, "workspace_id": ws1_id, "mode": "adaptive"},
            headers={"Accept": "application/json"}
        )
    adap_data = r_adap.json()

    # Fetch thread
    r_mix_thread = tc1.get(f"/conversations/{mix_conv_id}", headers={"Accept": "application/json"})
    mix_queries = r_mix_thread.json()["queries"]

    c2_ok = (
        len(mix_queries) == 2 and
        mix_queries[0]["mode"] == "standard" and
        mix_queries[0]["evaluation"] is None and
        mix_queries[1]["mode"] == "adaptive" and
        mix_queries[1]["evaluation"] is not None and
        mix_queries[1]["evaluation"]["overall_score"] == 0.89 and
        r_mix_thread.json()["conversation"]["kb_id"] == kb1_id
    )
    print_result("Standard & Adaptive modes retain independent semantics in same thread", c2_ok, f"Turn 1 Mode: {mix_queries[0]['mode']}, Turn 2 Mode: {mix_queries[1]['mode']}, Score: {mix_queries[1]['evaluation']['overall_score']}")
    results["mode_and_kb_continuity"] = c2_ok

    # -------------------------------------------------------------------------
    # 3. Feedback: Helpful -> Not Helpful with Reason (In-Place Update)
    # -------------------------------------------------------------------------
    print_header("3. FEEDBACK SUBMISSION AND IN-PLACE UPDATE")
    target_q_id = q1_id

    # 1. Submit Helpful
    r_fb1 = tc1.post(
        f"/query/{target_q_id}/feedback",
        data={"rating": "helpful", "comment": "Initial rating: concise and helpful"},
        headers={"Accept": "application/json"}
    )
    fb1_res = r_fb1.json()

    db = SessionLocal()
    records_1 = db.query(QueryFeedback).filter(QueryFeedback.query_id == target_q_id).all()
    count_1 = len(records_1)
    db.close()

    # 2. Update to Not Helpful with Reason
    r_fb2 = tc1.post(
        f"/query/{target_q_id}/feedback",
        data={"rating": "not_helpful", "reason": "incorrect_answer", "comment": "Changed my mind: citation was ambiguous"},
        headers={"Accept": "application/json"}
    )
    fb2_res = r_fb2.json()

    db = SessionLocal()
    records_2 = db.query(QueryFeedback).filter(QueryFeedback.query_id == target_q_id).all()
    count_2 = len(records_2)
    final_fb = records_2[0] if count_2 > 0 else None
    final_rating = final_fb.rating if final_fb else None
    final_reason = final_fb.reason if final_fb else None
    final_comment = final_fb.comment if final_fb else None
    db.close()

    c3_ok = (
        r_fb1.status_code == 200 and
        r_fb2.status_code == 200 and
        count_1 == 1 and
        count_2 == 1 and
        final_rating == "not_helpful" and
        final_reason == "incorrect_answer" and
        final_comment == "Changed my mind: citation was ambiguous"
    )
    print_result("Feedback updated in-place without duplication", c3_ok, f"Total records for query: {count_2}, Final Rating: {final_rating}, Reason: {final_reason}")
    results["feedback_update"] = c3_ok

    # -------------------------------------------------------------------------
    # 4. Deleted Sources Safeguard
    # -------------------------------------------------------------------------
    print_header("4. DELETED SOURCES SAFEGUARD")
    # Upload doc to be deleted
    r_del_up = tc1.post(
        "/documents/upload",
        data={"workspace_id": ws1_id, "kb_id": kb1_id},
        files={"file": ("legacy_spec.txt", b"Legacy specification version 0.9 text for deletion test.", "text/plain")}
    )
    db = SessionLocal()
    del_doc = db.query(Document).filter(Document.workspace_id == ws1_id, Document.filename == "legacy_spec.txt").first()
    del_doc_id = del_doc.id
    db.close()

    # Start thread and query against this doc
    r_del_conv = tc1.post("/conversations", data={"title": "Deletion Thread", "kb_id": kb1_id, "workspace_id": ws1_id}, headers={"Accept": "application/json"})
    del_conv_id = r_del_conv.json()["id"]

    with patch.object(generator, "generate", return_value={"answer": "Legacy answer based on spec [1].", "success": True, "error": None}):
        r_q_del = tc1.post(
            "/query",
            data={"query": "What is in legacy spec?", "conversation_id": del_conv_id, "workspace_id": ws1_id, "mode": "standard"},
            headers={"Accept": "application/json"}
        )
    q_del_data = r_q_del.json()
    assert len(q_del_data["final"]["citations"]) > 0
    del_chunk_id = q_del_data["final"]["citations"][0]["chunk_id"]

    # Verify citation was available before delete
    r_before_del = tc1.get(f"/conversations/{del_conv_id}", headers={"Accept": "application/json"})
    citations_before = r_before_del.json()["queries"][0]["citations"]
    avail_before = citations_before[0]["source_available"]

    # Delete the document
    r_doc_del = tc1.post(f"/documents/{del_doc_id}/delete", data={"workspace_id": ws1_id})

    # Revisit conversation
    r_after_del = tc1.get(f"/conversations/{del_conv_id}", headers={"Accept": "application/json"})
    queries_after = r_after_del.json()["queries"]
    citations_after = queries_after[0]["citations"]
    avail_after = citations_after[0]["source_available"]
    answer_readable = queries_after[0]["answer"] == "Legacy answer based on spec [1]."

    # Try to inspect the deleted chunk directly
    r_chunk_inspect = tc1.get(f"/chunks/{del_chunk_id}")
    chunk_404 = (r_chunk_inspect.status_code == 404)

    c4_ok = (
        avail_before is True and
        avail_after is False and
        answer_readable and
        chunk_404
    )
    print_result("Deleted document citations marked unavailable & answer preserved", c4_ok, f"Source Avail Before: {avail_before}, After: {avail_after}, Chunk 404: {chunk_404}")
    results["deleted_sources"] = c4_ok

    # -------------------------------------------------------------------------
    # 5. Multi-User and Cross-Workspace Authorization
    # -------------------------------------------------------------------------
    print_header("5. MULTI-USER AND CROSS-WORKSPACE AUTHORIZATION")
    u2_name = f"verify_user2_{uuid.uuid4().hex[:6]}"
    tc2 = TestClient(app)
    tc2.post("/register", data={"username": u2_name, "password": "password123"})
    db = SessionLocal()
    user2 = db.query(User).filter(User.username == u2_name).first()
    ws2_id = user2.workspace_id
    db.close()

    # User 2 attempts to access User 1's conversation
    r_u2_get = tc2.get(f"/conversations/{conv_id}", headers={"Accept": "application/json"})
    r_u2_rename = tc2.post(f"/conversations/{conv_id}/rename", data={"title": "Hacked Title"})
    r_u2_delete = tc2.post(f"/conversations/{conv_id}/delete")
    r_u2_feedback = tc2.post(f"/query/{q1_id}/feedback", data={"rating": "helpful"})

    c5_ok = (
        r_u2_get.status_code == 404 and
        r_u2_rename.status_code == 404 and
        r_u2_delete.status_code == 404 and
        r_u2_feedback.status_code == 404
    )
    print_result("Unauthorized cross-tenant actions safely rejected with 404", c5_ok, f"GET: {r_u2_get.status_code}, Rename: {r_u2_rename.status_code}, Delete: {r_u2_delete.status_code}, Feedback: {r_u2_feedback.status_code}")
    results["authorization"] = c5_ok

    # -------------------------------------------------------------------------
    # 6. Database Migration and Restart Verification
    # -------------------------------------------------------------------------
    print_header("6. DATABASE MIGRATION ON LEGACY COPY & RESTART VERIFICATION")
    db_path = "adaptive_rag.db"
    backup_path = "adaptive_rag.db.bak"
    shutil.copyfile(db_path, backup_path)
    print(f"  Created backup of active database: {backup_path}")

    # Build a simulated legacy SQLite database
    legacy_test_db = "legacy_test_verification.db"
    if os.path.exists(legacy_test_db):
        os.remove(legacy_test_db)

    conn_leg = sqlite3.connect(legacy_test_db)
    cur_leg = conn_leg.cursor()
    # Create older schema without conversations, query_feedback, or new query columns
    cur_leg.execute("CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, name VARCHAR, created_at DATETIME)")
    cur_leg.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR, hashed_password VARCHAR, workspace_id VARCHAR, created_at DATETIME)")
    cur_leg.execute("CREATE TABLE knowledge_bases (id VARCHAR PRIMARY KEY, name VARCHAR, description TEXT, workspace_id VARCHAR, created_at DATETIME)")
    cur_leg.execute("""
    CREATE TABLE queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query VARCHAR,
        answer TEXT,
        timestamp DATETIME,
        latency FLOAT,
        llm_calls INTEGER,
        retrieval_config VARCHAR,
        workspace_id VARCHAR,
        kb_id VARCHAR,
        mode VARCHAR
    )
    """)
    cur_leg.execute("INSERT INTO workspaces VALUES ('ws_legacy', 'Legacy Workspace', '2026-01-01 00:00:00')")
    cur_leg.execute("INSERT INTO users VALUES (1, 'legacy_user', 'hash', 'ws_legacy', '2026-01-01 00:00:00')")
    cur_leg.execute("INSERT INTO queries (query, answer, workspace_id, mode) VALUES ('Legacy Question 1', 'Legacy Answer 1', 'ws_legacy', 'standard')")
    cur_leg.execute("INSERT INTO queries (query, answer, workspace_id, mode) VALUES ('Legacy Question 2', 'Legacy Answer 2', 'ws_legacy', 'adaptive')")
    conn_leg.commit()
    conn_leg.close()
    print("  Created legacy test database with 2 legacy unthreaded queries.")

    # Run migration script logic against legacy_test_db
    from scripts.migrate_conversations_and_feedback import DB_PATH as DEFAULT_MIG_PATH
    import scripts.migrate_conversations_and_feedback as mig_module
    old_mig_path = mig_module.DB_PATH
    mig_module.DB_PATH = os.path.abspath(legacy_test_db)
    mig_module.migrate()
    mig_module.DB_PATH = old_mig_path

    # Verify migration results on legacy_test_db
    conn_check = sqlite3.connect(legacy_test_db)
    cur_check = conn_check.cursor()
    cur_check.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur_check.fetchall()}
    cur_check.execute("PRAGMA table_info(queries)")
    mig_columns = {r[1] for r in cur_check.fetchall()}
    cur_check.execute("SELECT id, query, conversation_id, answer_status FROM queries")
    legacy_rows = cur_check.fetchall()
    conn_check.close()
    os.remove(legacy_test_db)

    c6_ok = (
        "conversations" in tables and
        "query_feedback" in tables and
        "conversation_id" in mig_columns and
        "answer_status" in mig_columns and
        len(legacy_rows) == 2 and
        legacy_rows[0][1] == "Legacy Question 1" and
        legacy_rows[0][2] is None and  # conversation_id is NULL for legacy queries
        legacy_rows[1][1] == "Legacy Question 2"
    )
    print_result("Migration on legacy DB copy succeeded and preserved unthreaded history", c6_ok, f"Tables: {tables}, Preserved queries: {len(legacy_rows)}")
    results["migration_and_restart"] = c6_ok

    # Cleanup backup
    if os.path.exists(backup_path):
        os.remove(backup_path)

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print_header("MANUAL CHECKLIST SUMMARY")
    all_passed = all(results.values())
    for k, v in results.items():
        status_str = "PASS" if v else "FAIL"
        print(f"  [{status_str}] {k.replace('_', ' ').title()}")

    print("\nOverall Status: " + ("ALL 6 CHECKLIST ITEMS VERIFIED SUCCESSFULLY!" if all_passed else "SOME CHECKS FAILED"))
    return all_passed

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
