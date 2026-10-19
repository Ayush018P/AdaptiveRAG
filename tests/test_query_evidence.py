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
from app.core.config_manager import SessionLocal as TestingSessionLocal, config_manager
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
from app.rag.hybrid_retriever import hybrid_retriever
from app.rag.pipeline import rag_pipeline
from app.rag.generator import generator
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
# Test 1: Citation-to-Chunk Mapping and Excerpt Verification
# --------------------------------------------------------------------------
def test_citation_to_chunk_mapping_and_excerpts():
    tc, user, ws_id = create_authenticated_user(f"cite_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Citation KB")
    
    sample_text = (
        "AdaptiveRAG uses dynamic evaluation to detect retrieval failures. "
        "The policy engine proposes parameter interventions when faithfulness drops below 0.7. "
        "Every citation must map back to an immutable persisted chunk."
    )
    doc_id = upload_test_doc(tc, ws_id, kb_id, "architectural_spec.txt", sample_text.encode("utf-8"))

    db = TestingSessionLocal()
    try:
        with patch.object(generator, "generate", return_value={
            "answer": "AdaptiveRAG uses dynamic evaluation [1] and policy interventions [1].",
            "success": True,
            "error": None,
            "llm_calls": 1
        }):
            res = rag_pipeline.run(workspace_id=ws_id, query="How does AdaptiveRAG detect failures?", db=db, kb_id=kb_id)
            
            assert res["answer_status"] == "SUCCESS"
            assert len(res["citations"]) > 0
            
            first_citation = res["citations"][0]
            assert first_citation["citation_id"] == 1
            assert first_citation["document_id"] == doc_id
            assert first_citation["filename"] == "architectural_spec.txt"
            assert first_citation["doc_type"] == ".txt"
            assert first_citation["page_number"] is None
            assert "AdaptiveRAG uses dynamic evaluation" in first_citation["excerpt"]
            
            # Verify database chunk backing
            chunk = db.query(Chunk).filter(Chunk.id == first_citation["chunk_id"]).first()
            assert chunk is not None
            assert chunk.document_id == doc_id
            assert chunk.workspace_id == ws_id
            assert chunk.kb_id == kb_id
            assert chunk.text.startswith(first_citation["excerpt"].rstrip("."))
    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 2: PDF Page Metadata vs Non-PDF Handling
# --------------------------------------------------------------------------
def test_pdf_page_metadata_handling():
    tc, user, ws_id = create_authenticated_user(f"pdf_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "PDF KB")
    
    db = TestingSessionLocal()
    try:
        # Create a PDF document and Chunk with a valid page number
        pdf_doc = Document(
            workspace_id=ws_id,
            kb_id=kb_id,
            filename="user_manual.pdf",
            doc_type=".pdf",
            status="READY",
            source_path="data/uploads/user_manual.pdf"
        )
        db.add(pdf_doc)
        db.commit()
        db.refresh(pdf_doc)
        
        pdf_chunk = Chunk(
            workspace_id=ws_id,
            kb_id=kb_id,
            document_id=pdf_doc.id,
            chunk_index=0,
            text="Section 4.2 describes system calibration on page 4.",
            page_number=4
        )
        db.add(pdf_chunk)

        # Create a TXT document and Chunk with no page number
        txt_doc = Document(
            workspace_id=ws_id,
            kb_id=kb_id,
            filename="notes.txt",
            doc_type=".txt",
            status="READY",
            source_path="data/uploads/notes.txt"
        )
        db.add(txt_doc)
        db.commit()
        db.refresh(txt_doc)
        
        txt_chunk = Chunk(
            workspace_id=ws_id,
            kb_id=kb_id,
            document_id=txt_doc.id,
            chunk_index=0,
            text="Plain text notes on system setup.",
            page_number=None
        )
        db.add(txt_chunk)
        db.commit()
        db.refresh(pdf_chunk)
        db.refresh(txt_chunk)

        mock_retrieval = {
            "hybrid_results": [
                {"chunk_id": pdf_chunk.id, "score": 0.95, "dense_score": 0.95, "bm25_score": 0.90},
                {"chunk_id": txt_chunk.id, "score": 0.85, "dense_score": 0.85, "bm25_score": 0.80}
            ]
        }
        with patch.object(hybrid_retriever, "retrieve", return_value=mock_retrieval), \
             patch.object(generator, "generate", return_value={"answer": "Calibration details [1].", "success": True, "error": None}):
            res = rag_pipeline.run(workspace_id=ws_id, query="calibration", db=db, kb_id=kb_id)
            
            # Find PDF citation
            pdf_cites = [c for c in res["citations"] if c["filename"] == "user_manual.pdf"]
            assert len(pdf_cites) > 0
            assert pdf_cites[0]["page_number"] == 4
            
            # Find TXT citation
            txt_cites = [c for c in res["citations"] if c["filename"] == "notes.txt"]
            if txt_cites:
                assert txt_cites[0]["page_number"] is None

        # Test chunk inspection endpoint GET /chunks/{chunk_id}
        r_pdf = tc.get(f"/chunks/{pdf_chunk.id}")
        assert r_pdf.status_code == 200
        data_pdf = r_pdf.json()
        assert data_pdf["page_number"] == 4
        assert data_pdf["filename"] == "user_manual.pdf"

        r_txt = tc.get(f"/chunks/{txt_chunk.id}")
        assert r_txt.status_code == 200
        data_txt = r_txt.json()
        assert data_txt["page_number"] is None
        assert data_txt["filename"] == "notes.txt"

    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 3: Deleted Documents Never Produce Valid Citations
# --------------------------------------------------------------------------
def test_deleted_documents_purged_from_citations():
    tc, user, ws_id = create_authenticated_user(f"del_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Delete Test KB")
    
    content = b"Confidential financial statement for Q4 2026."
    doc_id = upload_test_doc(tc, ws_id, kb_id, "confidential_q4.txt", content)

    db = TestingSessionLocal()
    try:
        # Before deletion, query finds it
        with patch.object(generator, "generate", return_value={"answer": "Financial results [1]", "success": True, "error": None}):
            res_before = rag_pipeline.run(workspace_id=ws_id, query="financial", db=db, kb_id=kb_id)
            assert len(res_before["citations"]) > 0

        # Now delete the document via API
        r_del = tc.post(f"/documents/{doc_id}/delete", data={"workspace_id": ws_id})
        assert r_del.status_code == 200

        # After deletion, document is gone from citations
        res_after = rag_pipeline.run(workspace_id=ws_id, query="financial", db=db, kb_id=kb_id)
        assert res_after["citations"] == []
        assert res_after["answer_status"] in ("EMPTY_KNOWLEDGE_BASE", "NO_RELEVANT_CHUNKS")

        # Inspecting deleted chunk via API should return 404
        chunks = db.query(Chunk).filter(Chunk.document_id == doc_id).all()
        assert len(chunks) == 0
    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 4: Empty Knowledge Base State (No Citations, Explicit Banner)
# --------------------------------------------------------------------------
def test_empty_knowledge_base_state():
    tc, user, ws_id = create_authenticated_user(f"empty_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Totally Empty KB")

    db = TestingSessionLocal()
    try:
        res = rag_pipeline.run(workspace_id=ws_id, query="What is the architecture?", db=db, kb_id=kb_id)
        assert res["answer_status"] == "EMPTY_KNOWLEDGE_BASE"
        assert res["citations"] == []
        assert res["retrieved_chunks"] == []
        assert res["llm_calls"] == 0
        assert "Selected knowledge base is empty" in res["answer"]

        # Test POST /query web route
        r = tc.post("/query", data={"query": "What is the architecture?", "kb_id": kb_id, "workspace_id": ws_id})
        assert r.status_code == 200
        assert "Target Knowledge Base is Empty" in r.text or "Empty Knowledge Base" in r.text
        assert "contains no ready documents" in r.text or "No documents available" in r.text
    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 5: No Relevant Chunks Found State
# --------------------------------------------------------------------------
def test_no_relevant_chunks_state():
    tc, user, ws_id = create_authenticated_user(f"norelev_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Specific KB")

    upload_test_doc(
        tc,
        ws_id,
        kb_id,
        "baking_recipes.txt",
        b"To bake sourdough bread, mix flour, water, salt, and wild yeast starter."
    )

    db = TestingSessionLocal()
    try:
        with patch.object(hybrid_retriever, "retrieve", return_value={"hybrid_results": []}):
            res = rag_pipeline.run(workspace_id=ws_id, query="quantum physics entanglement", db=db, kb_id=kb_id)
            assert res["answer_status"] == "NO_RELEVANT_CHUNKS"
            assert res["citations"] == []
            assert res["retrieved_chunks"] == []
            assert res["llm_calls"] == 0
            assert "No relevant information found" in res["answer"]

            # Test web route
            r = tc.post("/query", data={"query": "quantum physics entanglement", "kb_id": kb_id, "workspace_id": ws_id})
            assert r.status_code == 200
            assert "Insufficient Retrieved Evidence" in r.text or "No Relevant Chunks" in r.text
    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 6: Cross-Workspace and Cross-Knowledge-Base Isolation
# --------------------------------------------------------------------------
def test_cross_workspace_and_cross_kb_isolation():
    tc_a, user_a, ws_a = create_authenticated_user(f"tenant_a_{uuid.uuid4().hex[:6]}")
    kb_a = create_kb_for_workspace(tc_a, ws_a, "KB Alpha")

    tc_b, user_b, ws_b = create_authenticated_user(f"tenant_b_{uuid.uuid4().hex[:6]}")
    kb_b = create_kb_for_workspace(tc_b, ws_b, "KB Beta")

    # Ingest confidential doc into Tenant A
    doc_a_id = upload_test_doc(
        tc_a,
        ws_a,
        kb_a,
        "tenant_a_top_secret.txt",
        b"SecretProjectOmega top secret proprietary cryptographic key 998877."
    )

    db = TestingSessionLocal()
    try:
        # Tenant A can query and see it
        with patch.object(generator, "generate", return_value={"answer": "Key 998877 [1]", "success": True, "error": None}):
            res_a = rag_pipeline.run(workspace_id=ws_a, query="SecretProjectOmega key", db=db, kb_id=kb_a)
            assert len(res_a["citations"]) > 0
            assert res_a["citations"][0]["filename"] == "tenant_a_top_secret.txt"
            chunk_a_id = res_a["citations"][0]["chunk_id"]

        # Tenant B queries with same terms in their own workspace
        res_b = rag_pipeline.run(workspace_id=ws_b, query="SecretProjectOmega key", db=db, kb_id=kb_b)
        assert res_b["answer_status"] in ("EMPTY_KNOWLEDGE_BASE", "NO_RELEVANT_CHUNKS")
        assert res_b["citations"] == []

        # Tenant B attempts to inspect Tenant A's chunk via GET /chunks/{chunk_id}
        r_hack = tc_b.get(f"/chunks/{chunk_a_id}")
        assert r_hack.status_code == 404

        # Tenant B attempts to run query providing Tenant A's kb_id
        r_kb_hack = tc_b.post("/query", data={"query": "SecretProjectOmega", "kb_id": kb_a, "workspace_id": ws_b})
        assert r_kb_hack.status_code == 404

    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 7: Provider Error and Failure States (No Fabricated Fallbacks)
# --------------------------------------------------------------------------
def test_provider_failure_state():
    tc, user, ws_id = create_authenticated_user(f"prov_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Provider Test KB")

    upload_test_doc(
        tc,
        ws_id,
        kb_id,
        "system_health.txt",
        b"All operational parameters within standard limits."
    )

    db = TestingSessionLocal()
    try:
        # Simulate LLM Provider failure (e.g. rate limit exhaustion / 503)
        with patch.object(generator, "generate", return_value={
            "answer": "Failed to generate answer due to LLM error.",
            "success": False,
            "error": "HTTP 503: Model provider unavailable.",
            "llm_calls": 3
        }):
            res = rag_pipeline.run(workspace_id=ws_id, query="system health", db=db, kb_id=kb_id)
            
            assert res["answer_status"] == "PROVIDER_FAILURE"
            assert res["generation_error"] == "HTTP 503: Model provider unavailable."
            # Citations should still be preserved so user sees what evidence was gathered
            assert len(res["citations"]) > 0
            assert res["citations"][0]["filename"] == "system_health.txt"
            
            # Post to web route
            r = tc.post("/query", data={"query": "system health", "kb_id": kb_id, "workspace_id": ws_id})
            assert r.status_code == 200
            assert "Generation Provider Failure" in r.text or "LLM Provider Error" in r.text
            assert "HTTP 503" in r.text
            # Must NOT fabricate a successful answer
            assert "Generated Response" not in r.text
    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 8: Rejected Adaptive Experiment Preserves Baseline Configuration
# --------------------------------------------------------------------------
def test_rejected_adaptive_experiment_preserves_baseline():
    tc, user, ws_id = create_authenticated_user(f"exp_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Exp KB")

    upload_test_doc(
        tc,
        ws_id,
        kb_id,
        "policy_doc.txt",
        b"Corporate security policies require multi-factor authentication for all logins."
    )

    initial_config = config_manager.get_active_config(ws_id)
    baseline_top_k = initial_config["top_k"]

    db = TestingSessionLocal()
    try:
        # Simulate experiment where candidate performs worse than baseline
        baseline_eval = {
            "overall_score": 0.55,
            "faithfulness_score": 0.50,
            "relevance_score": 0.60,
            "context_score": 0.55,
            "completeness_score": 0.55,
            "failure_detected": True
        }
        candidate_eval = {
            "overall_score": 0.40,  # Worse score!
            "faithfulness_score": 0.35,
            "relevance_score": 0.45,
            "context_score": 0.40,
            "completeness_score": 0.40,
            "failure_detected": True
        }
        hypothesis = {
            "failure_type": "RETRIEVAL_FAILURE",
            "current_config": initial_config,
            "candidate_config": {"top_k": baseline_top_k + 2}
        }
        baseline_res = {
            "total_latency": 0.5,
            "evaluation_latency": 0.2,
            "llm_calls": 1
        }

        with patch.object(rag_pipeline, "run", return_value={
            "answer": "Candidate generated answer",
            "answer_status": "SUCCESS",
            "retrieved_chunks": [],
            "citations": [],
            "retrieval_config": {"top_k": baseline_top_k + 2},
            "total_latency": 0.6,
            "llm_calls": 1,
            "context": ""
        }), patch("app.experiment.experiment_engine.evaluator.evaluate", return_value=candidate_eval):
            
            exp_res = experiment_engine.run_experiment(
                workspace_id=ws_id,
                query="security policies",
                query_id=1,
                hypothesis=hypothesis,
                db=db,
                baseline_result=baseline_res,
                baseline_eval=baseline_eval,
                kb_id=kb_id
            )

            assert exp_res["decision"] == "REJECT"
            
            # Active config must remain initial_config
            current_config = config_manager.get_active_config(ws_id)
            assert current_config["top_k"] == baseline_top_k

    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 9: Accepted Adaptive Experiment Applies Change Transparently
# --------------------------------------------------------------------------
def test_accepted_adaptive_experiment_applies_change():
    tc, user, ws_id = create_authenticated_user(f"exp_acc_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Exp Acc KB")

    initial_config = config_manager.get_active_config(ws_id)
    baseline_top_k = initial_config["top_k"]
    new_top_k = baseline_top_k + 2

    db = TestingSessionLocal()
    try:
        baseline_eval = {
            "overall_score": 0.50,
            "faithfulness_score": 0.45,
            "relevance_score": 0.55,
            "context_score": 0.50,
            "completeness_score": 0.50,
            "failure_detected": True
        }
        candidate_eval = {
            "overall_score": 0.85,  # Substantial improvement!
            "faithfulness_score": 0.85,
            "relevance_score": 0.85,
            "context_score": 0.85,
            "completeness_score": 0.85,
            "failure_detected": False
        }
        hypothesis = {
            "failure_type": "RETRIEVAL_FAILURE",
            "current_config": initial_config,
            "candidate_config": {"top_k": new_top_k}
        }
        baseline_res = {
            "total_latency": 0.5,
            "evaluation_latency": 0.2,
            "llm_calls": 1
        }

        with patch.object(rag_pipeline, "run", return_value={
            "answer": "Candidate improved answer [1]",
            "answer_status": "SUCCESS",
            "retrieved_chunks": [],
            "citations": [],
            "retrieval_config": {"top_k": new_top_k},
            "total_latency": 0.55,
            "llm_calls": 1,
            "context": ""
        }), patch("app.experiment.experiment_engine.evaluator.evaluate", return_value=candidate_eval):
            
            exp_res = experiment_engine.run_experiment(
                workspace_id=ws_id,
                query="improving retrieval",
                query_id=2,
                hypothesis=hypothesis,
                db=db,
                baseline_result=baseline_res,
                baseline_eval=baseline_eval,
                kb_id=kb_id
            )

            assert exp_res["decision"] == "ACCEPT"
            
            # Active config must be updated
            current_config = config_manager.get_active_config(ws_id)
            assert current_config["top_k"] == new_top_k

    finally:
        db.close()


# --------------------------------------------------------------------------
# Test 10: JSON API Response Contract Contains Citations and Answer Status
# --------------------------------------------------------------------------
def test_json_api_response_contract():
    tc, user, ws_id = create_authenticated_user(f"json_user_{uuid.uuid4().hex[:6]}")
    kb_id = create_kb_for_workspace(tc, ws_id, "Contract KB")

    upload_test_doc(
        tc,
        ws_id,
        kb_id,
        "contract_doc.txt",
        b"Sample document to verify full API contract fields."
    )

    with patch.object(generator, "generate", return_value={"answer": "Verified [1]", "success": True, "error": None}):
        # Query with Accept: application/json
        r = tc.post(
            "/query",
            data={"query": "verify contract", "kb_id": kb_id, "workspace_id": ws_id},
            headers={"Accept": "application/json"}
        )
        assert r.status_code == 200
        data = r.json()
        
        assert "query" in data
        assert "baseline" in data
        assert "final" in data
        assert "adaptation" in data
        
        final_block = data["final"]
        assert "answer" in final_block
        assert "citations" in final_block
        assert len(final_block["citations"]) > 0
        citation = final_block["citations"][0]
        assert "citation_id" in citation
        assert "chunk_id" in citation
        assert "document_id" in citation
        assert "filename" in citation
        assert "excerpt" in citation
        assert "score" in citation
