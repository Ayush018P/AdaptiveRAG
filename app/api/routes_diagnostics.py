from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.query import QueryLog
from app.models.experiment import Experiment
from app.core.auth import get_current_user
import json

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/diagnosis")
async def diagnosis_page(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    queries = db.query(QueryLog).filter(QueryLog.workspace_id == ws_id).order_by(QueryLog.id.desc()).all()
    return templates.TemplateResponse("diagnostics.html", {"request": request, "queries": queries})

@router.get("/experiments")
async def experiments_page(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    experiments = db.query(Experiment).filter(Experiment.workspace_id == ws_id).order_by(Experiment.id.desc()).all()
    # parse json fields for jinja
    parsed_experiments = []
    for exp in experiments:
        exp_dict = exp.__dict__.copy()
        try:
            exp_dict["hypothesis_obj"] = json.loads(exp.hypothesis)
            exp_dict["candidate_config_obj"] = json.loads(exp.candidate_configuration)
            baseline_eval = json.loads(exp.baseline_metrics)
            candidate_eval = json.loads(exp.candidate_metrics)
            exp_dict["baseline_score"] = baseline_eval.get("overall_score", 0.0)
            exp_dict["candidate_score"] = candidate_eval.get("overall_score", 0.0)
        except:
            exp_dict["hypothesis_obj"] = {}
            exp_dict["candidate_config_obj"] = {}
            exp_dict["baseline_score"] = 0.0
            exp_dict["candidate_score"] = 0.0
        
        query_log = db.query(QueryLog).filter(QueryLog.id == exp.query_id).first()
        exp_dict["query_text"] = query_log.query if query_log else "Unknown Query"
        
        parsed_experiments.append(exp_dict)
        
    return templates.TemplateResponse("experiments.html", {"request": request, "experiments": parsed_experiments})

@router.get("/benchmark")
async def benchmark_page(request: Request):
    import os
    import scipy.stats as stats
    results_file = "results/benchmark_results.json"
    results = []
    summary = {}
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
            
        if results:
            summary = {
                "total": len(results),
                "static_avg": sum(r["static_indep_score"] for r in results) / len(results),
                "adaptive_avg": sum(r["adaptive_indep_score"] for r in results) / len(results),
                "interventions": sum(r["adaptive_interventions"] for r in results),
                "accepted": sum(r.get("adaptive_accepted", 0) for r in results),
            }
            summary["rejected"] = summary["interventions"] - summary["accepted"]
            
            static_scores = [r["static_indep_score"] for r in results]
            adaptive_scores = [r["adaptive_indep_score"] for r in results]
            
            if len(results) > 1:
                t_stat, p_value = stats.ttest_rel(adaptive_scores, static_scores)
                summary["t_stat"] = t_stat
                summary["p_value"] = p_value
            else:
                summary["t_stat"] = None
                summary["p_value"] = None

    return templates.TemplateResponse("benchmark.html", {"request": request, "results": results, "summary": summary})
