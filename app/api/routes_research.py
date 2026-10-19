from fastapi import APIRouter, Depends, Request, Form
from app.core.auth import get_current_user
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.database import get_db
from app.models.query import QueryLog
from app.models.experiment import Experiment
from app.models.diagnosis import Diagnosis
from app.models.configuration import Configuration
from app.core.config_manager import config_manager
from app.rag.pipeline import rag_pipeline
from app.evaluation.evaluator import evaluator
import json
import time

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/memory")
async def memory_page(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    experiments = db.query(Experiment).filter(Experiment.workspace_id == ws_id).order_by(Experiment.created_at.desc()).all()
    memory_records = []
    for exp in experiments:
        try:
            hypo = json.loads(exp.hypothesis)
            baseline_conf = json.loads(exp.baseline_configuration)
            candidate_conf = json.loads(exp.candidate_configuration)
        except:
            continue
            
        memory_records.append({
            "id": exp.id,
            "failure_type": exp.failure_type,
            "action": hypo.get("action", "UNKNOWN"),
            "baseline_config": baseline_conf,
            "candidate_config": candidate_conf,
            "decision": exp.decision,
            "score_delta": exp.score_delta,
            "latency_delta": exp.latency_delta,
            "reason": exp.acceptance_reason or "N/A",
            "timestamp": exp.created_at
        })
    return templates.TemplateResponse("memory.html", {"request": request, "records": memory_records})

@router.get("/lab")
async def failure_analysis_lab(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    diagnoses = db.query(Diagnosis.failure_type, func.count(Diagnosis.id).label("count")).join(QueryLog).filter(QueryLog.workspace_id == ws_id).group_by(Diagnosis.failure_type).all()
    distribution = [{"type": d.failure_type, "count": d.count} for d in diagnoses]
    
    selected_type = request.query_params.get("type")
    detail = None
    if selected_type:
        type_diagnoses = db.query(Diagnosis).join(QueryLog).filter(Diagnosis.failure_type == selected_type, QueryLog.workspace_id == ws_id).all()
        query_ids = [d.query_id for d in type_diagnoses]
        
        experiments = db.query(Experiment).filter(Experiment.failure_type == selected_type, Experiment.workspace_id == ws_id).all()
        accepted = sum(1 for e in experiments if e.decision == "ACCEPT")
        rejected = len(experiments) - accepted
        
        avg_base_score = sum(e.score_delta - e.score_delta + json.loads(e.baseline_metrics).get("overall_score", 0) for e in experiments) / len(experiments) if experiments else 0
        avg_cand_score = sum(json.loads(e.candidate_metrics).get("overall_score", 0) for e in experiments) / len(experiments) if experiments else 0
        
        detail = {
            "type": selected_type,
            "occurrences": len(type_diagnoses),
            "experiments_performed": len(experiments),
            "accepted": accepted,
            "rejected": rejected,
            "avg_base_score": avg_base_score,
            "avg_cand_score": avg_cand_score
        }
        
    return templates.TemplateResponse("lab.html", {"request": request, "distribution": distribution, "detail": detail})

@router.get("/config/history")
async def config_history(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    configs = db.query(Configuration).filter(Configuration.workspace_id == ws_id).order_by(Configuration.version_id.desc()).all()
    return templates.TemplateResponse("config_history.html", {"request": request, "configs": configs})

@router.post("/config/rollback")
async def config_rollback(request: Request, version_id: int = Form(...), db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    try:
        new_v = config_manager.rollback_config(ws_id, version_id)
        message = f"Rollback successful. Active Configuration: v{new_v}"
    except Exception as e:
        message = f"Rollback failed: {str(e)}"
    
    configs = db.query(Configuration).filter(Configuration.workspace_id == ws_id).order_by(Configuration.version_id.desc()).all()
    return templates.TemplateResponse("config_history.html", {"request": request, "configs": configs, "message": message})

@router.get("/experiment/builder")
async def experiment_builder(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    return templates.TemplateResponse("experiment_builder.html", {"request": request})

@router.post("/experiment/run")
async def run_experiment_builder(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    form_data = await request.form()
    query = form_data.get("query")
    
    base_config = {
        "top_k": int(form_data.get("base_top_k", 5)),
        "dense_weight": float(form_data.get("base_dense", 0.65)),
        "bm25_weight": float(form_data.get("base_bm25", 0.35)),
        "enable_reranker": form_data.get("base_rerank") == "on",
        "query_rewrite_enabled": form_data.get("base_rewrite") == "on"
    }
    
    cand_config = {
        "top_k": int(form_data.get("cand_top_k", 5)),
        "dense_weight": float(form_data.get("cand_dense", 0.65)),
        "bm25_weight": float(form_data.get("cand_bm25", 0.35)),
        "enable_reranker": form_data.get("cand_rerank") == "on",
        "query_rewrite_enabled": form_data.get("cand_rewrite") == "on"
    }
    
    # Run Baseline
    config_manager.activate_config(ws_id, base_config, is_experimental=True)
    b_res = rag_pipeline.run(ws_id, query, db)
    b_eval = evaluator.evaluate(query, b_res["answer"], b_res["context"], b_res["retrieved_chunks"])
    
    # Run Candidate
    config_manager.activate_config(ws_id, cand_config, is_experimental=True)
    c_res = rag_pipeline.run(ws_id, query, db)
    c_eval = evaluator.evaluate(query, c_res["answer"], c_res["context"], c_res["retrieved_chunks"])
    
    # Restore original active config
    config_manager.rollback_config(ws_id) # Rolls back the last experimental candidate
    config_manager.rollback_config(ws_id) # Rolls back the first experimental baseline
    
    score_delta = c_eval["overall_score"] - b_eval["overall_score"]
    decision = "ACCEPTED" if score_delta > 0 else "REJECTED"
    reason = "Candidate score improved." if decision == "ACCEPTED" else "Candidate score did not improve."
    
    result = {
        "query": query,
        "baseline": b_res,
        "baseline_eval": b_eval,
        "candidate": c_res,
        "candidate_eval": c_eval,
        "decision": decision,
        "reason": reason
    }
    
    return templates.TemplateResponse("experiment_result.html", {"request": request, "result": result})
