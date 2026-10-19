from fastapi import APIRouter, Depends, Request, Form, HTTPException
from app.core.auth import get_current_user, validate_workspace
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.rag.pipeline import rag_pipeline
from app.evaluation.evaluator import evaluator
from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine
from app.experiment.experiment_engine import experiment_engine
from app.memory.memory_engine import memory_engine
from app.core.config_manager import config_manager
import json
import logging
import time

from app.models.query import QueryLog
from app.models.conversation import Conversation
from app.models.evaluation import Evaluation
from app.models.diagnosis import Diagnosis
from app.models.optimization import Recommendation
from app.models.knowledge_base import KnowledgeBase
from app.models.chunk import Chunk
from app.models.document import Document
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/chunks/{chunk_id}")
async def inspect_chunk(chunk_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    chunk = db.query(Chunk).filter(Chunk.id == chunk_id, Chunk.workspace_id == user.workspace_id).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found or unauthorized")
    doc = db.query(Document).filter(Document.id == chunk.document_id, Document.workspace_id == user.workspace_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or unauthorized")
    return {
        "chunk_id": chunk.id,
        "document_id": doc.id,
        "filename": doc.filename,
        "doc_type": doc.doc_type,
        "page_number": chunk.page_number if (chunk.page_number and doc.doc_type == ".pdf") else None,
        "chunk_index": chunk.chunk_index,
        "workspace_id": chunk.workspace_id,
        "kb_id": chunk.kb_id,
        "text": chunk.text
    }

@router.get("/query")
async def query_page(request: Request, kb_id: str = None, mode: str = "standard", conversation_id: str = None, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    norm_mode = (mode or "standard").strip().lower()
    if norm_mode not in {"standard", "adaptive"}:
        norm_mode = "standard"
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == ws_id).order_by(KnowledgeBase.created_at.asc()).all()
    kbs_data = []
    for kb in kbs:
        doc_count = db.query(Document).filter(Document.workspace_id == ws_id, Document.kb_id == kb.id, Document.status == "READY").count()
        kbs_data.append({
            "id": kb.id,
            "name": kb.name,
            "description": kb.description or "",
            "doc_count": doc_count
        })

    convs = db.query(Conversation).filter(Conversation.workspace_id == ws_id).order_by(Conversation.updated_at.desc()).all()

    return templates.TemplateResponse("query.html", {
        "request": request,
        "knowledge_bases": kbs_data,
        "selected_kb_id": kb_id,
        "selected_mode": norm_mode,
        "mode": norm_mode,
        "conversations": convs,
        "selected_conversation_id": conversation_id
    })

@router.get("/history")
async def history_page(request: Request, workspace_id: str = None, kb_id: str = None, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    q = db.query(QueryLog).filter(QueryLog.workspace_id == ws_id)
    if kb_id:
        q = q.filter(QueryLog.kb_id == kb_id)
    queries = q.order_by(QueryLog.id.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "queries": queries, "selected_kb_id": kb_id})

@router.post("/query")
async def run_query(request: Request, query: str = Form(...), workspace_id: str = Form(None), kb_id: str = Form(None), mode: str = Form("standard"), conversation_id: str = Form(None), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = validate_workspace(workspace_id, user)
    workspace_id = ws_id

    # Validate mode strictly against allowlist
    ALLOWED_MODES = {"standard", "adaptive"}
    norm_mode = (mode or "standard").strip().lower()
    if norm_mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Supported modes are: standard, adaptive")

    # Validate conversation if provided
    conv = None
    clean_conv_id = conversation_id.strip() if conversation_id and conversation_id.strip() else None
    if clean_conv_id:
        conv = db.query(Conversation).filter(
            Conversation.id == clean_conv_id,
            Conversation.workspace_id == ws_id
        ).first()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        if not kb_id and conv.kb_id:
            kb_id = conv.kb_id

    if kb_id:
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.workspace_id == ws_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found or unauthorized")

    # 0. Save initial QueryLog with persisted mode, conversation, and user
    query_log = QueryLog(
        query=query,
        answer="Processing...",
        latency=0.0,
        retrieval_config="{}",
        workspace_id=workspace_id,
        user_id=user.id,
        conversation_id=conv.id if conv else None,
        kb_id=kb_id,
        mode=norm_mode,
        answer_status="PROCESSING"
    )
    db.add(query_log)
    db.commit()
    db.refresh(query_log)
    
    loop_history = []
    
    # 1. BASELINE RUN (both modes use the exact same verified retrieval & evidence pipeline)
    current_config = config_manager.get_active_config(workspace_id)
    baseline_result = rag_pipeline.run(workspace_id, query, db, kb_id=kb_id)
    baseline_result.setdefault("total_latency", baseline_result.get("latency", 0.0))
    baseline_result.setdefault("answer_status", "SUCCESS")
    baseline_result.setdefault("citations", [])
    baseline_result.setdefault("retrieved_chunks", [])
    answer_status = baseline_result.get("answer_status", "SUCCESS")
    
    attempts = 0
    max_attempts = 2

    if norm_mode == "standard":
        # STANDARD MODE: Clean question-answering workflow
        # No automatic evaluation, no diagnosis, no experiments, no parameter mutation.
        current_result = baseline_result
        current_eval = None
        baseline_eval = None
        adaptive_outcome = "STANDARD_MODE"

        query_log.answer = current_result["answer"]
        query_log.latency = current_result.get("total_latency", 0.0)
        query_log.llm_calls = current_result.get("llm_calls", 1)
        query_log.rewritten_query = current_result.get("rewritten_query")
        query_log.retrieval_config = json.dumps(current_config)
        query_log.answer_status = answer_status
        query_log.adaptive_outcome = adaptive_outcome
        query_log.citations_data = json.dumps(current_result.get("citations", []))

        if conv:
            conv.updated_at = datetime.utcnow()
            if conv.title == "New Conversation":
                conv.title = (query[:50].strip() + ("..." if len(query) > 50 else "")) or "Conversation"

        db.commit()

    else:
        # ADAPTIVE MODE: Transparent evaluation, diagnosis, policy & experimentation
        if answer_status == "EMPTY_KNOWLEDGE_BASE":
            baseline_eval = {
                "overall_score": 0.0,
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "context_score": 0.0,
                "completeness_score": 0.0,
                "failure_detected": False,
                "explanation": "Target knowledge base is empty. No documents available for retrieval."
            }
            baseline_result["evaluation_latency"] = 0.0
            current_result = baseline_result
            current_eval = baseline_eval
            adaptive_outcome = "EMPTY_KNOWLEDGE_BASE"

        elif answer_status == "NO_RELEVANT_CHUNKS":
            baseline_eval = {
                "overall_score": 0.0,
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "context_score": 0.0,
                "completeness_score": 0.0,
                "failure_detected": False,
                "explanation": "No relevant context chunks were found matching the query."
            }
            baseline_result["evaluation_latency"] = 0.0
            current_result = baseline_result
            current_eval = baseline_eval
            adaptive_outcome = "NO_RELEVANT_CHUNKS"

        elif answer_status == "PROVIDER_FAILURE":
            baseline_eval = {
                "overall_score": 0.0,
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "context_score": 0.0,
                "completeness_score": 0.0,
                "failure_detected": True,
                "explanation": f"LLM generation failed: {baseline_result.get('generation_error')}"
            }
            baseline_result["evaluation_latency"] = 0.0
            current_result = baseline_result
            current_eval = baseline_eval
            adaptive_outcome = "PROVIDER_FAILURE"

        else:
            # Standard SUCCESS baseline evaluation
            eval_start = time.time()
            baseline_eval = evaluator.evaluate(
                query=query, 
                answer=baseline_result["answer"], 
                context=baseline_result["context"], 
                retrieved_chunks=baseline_result["retrieved_chunks"]
            )
            eval_latency = time.time() - eval_start
            baseline_result["evaluation_latency"] = eval_latency
            
            current_result = baseline_result
            current_eval = baseline_eval
            
            while current_eval["failure_detected"] and attempts < max_attempts:
                attempts += 1
                logger.info(f"[ADAPTIVE] Attempt {attempts} started")
                
                # 2. DIAGNOSIS
                diagnosis = diagnostic_engine.diagnose(
                    query=query,
                    retrieved_chunks=current_result["retrieved_chunks"],
                    answer=current_result["answer"],
                    eval_result=current_eval
                )
                logger.info(f"[DIAGNOSIS] {diagnosis['failure_type']}")
                
                # 3. CHECK MEMORY
                memory_result = memory_engine.find_previous_success(workspace_id, diagnosis["failure_type"], db)
                if memory_result["found"]:
                    hypothesis = memory_result["hypothesis"]
                    hypothesis["reason"] = "Retrieved from memory: Previous successful optimization for this failure type."
                    logger.info(f"[MEMORY] Found previous successful hypothesis: {hypothesis['action']}")
                else:
                    # 4. POLICY HYPOTHESIS
                    hypothesis = policy_engine.recommend(workspace_id, diagnosis)
                    logger.info(f"[POLICY] Suggested Action: {hypothesis['action']}")
                    
                if hypothesis["action"] == "NO_ACTION" or hypothesis["action"] == "FLAG_KNOWLEDGE_BASE":
                    logger.info("[ADAPTIVE] Stopping loop due to NO_ACTION or FLAG_KNOWLEDGE_BASE")
                    loop_history.append({
                        "attempt": attempts,
                        "diagnosis": diagnosis,
                        "hypothesis": hypothesis,
                        "experiment": None
                    })
                    break
                    
                # 5. EXPERIMENT (Scoped to KB)
                experiment_res = experiment_engine.run_experiment(
                    workspace_id=workspace_id,
                    query=query,
                    query_id=query_log.id,
                    hypothesis=hypothesis,
                    db=db,
                    baseline_result=current_result,
                    baseline_eval=current_eval,
                    kb_id=kb_id
                )
                
                loop_history.append({
                    "attempt": attempts,
                    "diagnosis": diagnosis,
                    "hypothesis": hypothesis,
                    "experiment": experiment_res,
                    "memory_used": memory_result["found"]
                })
                
                # 6. DECISION
                if experiment_res["decision"] == "ACCEPT":
                    current_result = experiment_res["candidate_result"]
                    current_eval = experiment_res["candidate_eval"]
                else:
                    break

            if len(loop_history) == 0:
                adaptive_outcome = "BASELINE_ACCEPTED"
            else:
                last_loop = loop_history[-1]
                if last_loop.get("experiment") and last_loop["experiment"]["decision"] == "ACCEPT":
                    adaptive_outcome = "ADAPTATION_ACCEPTED"
                elif last_loop.get("experiment") and last_loop["experiment"]["decision"] == "REJECT":
                    adaptive_outcome = "ADAPTATION_REJECTED"
                else:
                    adaptive_outcome = "ADAPTATION_INCONCLUSIVE"

        # Update Query Log with final result
        query_log.answer = current_result["answer"]
        query_log.latency = current_result.get("total_latency", 0.0) + current_result.get("evaluation_latency", 0.0)
        query_log.llm_calls = current_result.get("llm_calls", 1)
        query_log.rewritten_query = current_result.get("rewritten_query")
        query_log.retrieval_config = json.dumps(config_manager.get_active_config(workspace_id))
        query_log.answer_status = answer_status
        query_log.adaptive_outcome = adaptive_outcome
        query_log.citations_data = json.dumps(current_result.get("citations", []))

        if conv:
            conv.updated_at = datetime.utcnow()
            if conv.title == "New Conversation":
                conv.title = (query[:50].strip() + ("..." if len(query) > 50 else "")) or "Conversation"
        
        # Save final evaluation
        eval_record = Evaluation(
            query_id=query_log.id,
            relevance_score=current_eval["relevance_score"],
            faithfulness_score=current_eval["faithfulness_score"],
            completeness_score=current_eval["completeness_score"],
            context_score=current_eval["context_score"],
            overall_score=current_eval["overall_score"],
            failure_detected=current_eval["failure_detected"],
            explanation=current_eval.get("explanation", "")
        )
        db.add(eval_record)
        
        # If there was a diagnosis, save the first one for the dashboard
        if loop_history:
            first_loop = loop_history[0]
            diag = first_loop["diagnosis"]
            diag_record = Diagnosis(
                query_id=query_log.id,
                failure_type=diag["failure_type"],
                confidence=diag["confidence"],
                evidence=diag["evidence"],
                explanation=diag["explanation"]
            )
            db.add(diag_record)
            
            hypo = first_loop.get("hypothesis")
            if hypo:
                rec_record = Recommendation(
                    query_id=query_log.id,
                    current_config=json.dumps(hypo.get("current_config", {})),
                    recommended_change=hypo["action"],
                    reason=hypo.get("reason", ""),
                    expected_purpose=hypo.get("expected_purpose", "")
                )
                db.add(rec_record)
                
        db.commit()

    current_result.setdefault("total_latency", current_result.get("latency", 0.0))
    current_result.setdefault("answer_status", "SUCCESS")
    current_result.setdefault("citations", [])
    current_result.setdefault("retrieved_chunks", [])

    response_data = {
        "query_id": query_log.id,
        "conversation_id": conv.id if conv else None,
        "query": query,
        "mode": norm_mode,
        "answer_status": current_result.get("answer_status", "SUCCESS"),
        "generation_error": current_result.get("generation_error"),
        "adaptive_outcome": adaptive_outcome,
        "baseline": {
            "configuration": current_config,
            "retrieved_chunks": baseline_result["retrieved_chunks"],
            "citations": baseline_result.get("citations", []),
            "answer": baseline_result["answer"],
            "evaluation": baseline_eval if norm_mode == "adaptive" else None
        },
        "adaptation": {
            "required": len(loop_history) > 0,
            "outcome": adaptive_outcome,
            "attempts": loop_history
        },
        "final": {
            "configuration": config_manager.get_active_config(workspace_id),
            "answer": current_result["answer"],
            "retrieved_chunks": current_result.get("retrieved_chunks", []),
            "citations": current_result.get("citations", []),
            "evaluation": current_eval if norm_mode == "adaptive" else None
        }
    }

    if "application/json" in request.headers.get("accept", ""):
        return response_data

    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.workspace_id == workspace_id).order_by(KnowledgeBase.created_at.asc()).all()
    kbs_data = []
    for kb in kbs:
        doc_count = db.query(Document).filter(Document.workspace_id == workspace_id, Document.kb_id == kb.id, Document.status == "READY").count()
        kbs_data.append({
            "id": kb.id,
            "name": kb.name,
            "description": kb.description or "",
            "doc_count": doc_count
        })

    convs = db.query(Conversation).filter(Conversation.workspace_id == workspace_id).order_by(Conversation.updated_at.desc()).all()

    return templates.TemplateResponse("query.html", {
        "request": request, 
        "query_id": query_log.id,
        "conversation_id": conv.id if conv else None,
        "query": query,
        "knowledge_bases": kbs_data,
        "selected_kb_id": kb_id,
        "selected_mode": norm_mode,
        "mode": norm_mode,
        "conversations": convs,
        "answer_status": current_result.get("answer_status", "SUCCESS"),
        "generation_error": current_result.get("generation_error"),
        "adaptive_outcome": adaptive_outcome,
        "baseline_result": baseline_result,
        "baseline_eval": baseline_eval if norm_mode == "adaptive" else None,
        "baseline_config": current_config,
        "loop_history": loop_history,
        "final_result": current_result,
        "final_eval": current_eval if norm_mode == "adaptive" else None,
        "final_config": config_manager.get_active_config(workspace_id),
        "max_attempts_reached": (attempts >= max_attempts and current_eval["failure_detected"]) if (norm_mode == "adaptive" and current_eval) else False
    })
