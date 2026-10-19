from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import engine, Base, get_db, SessionLocal
from contextlib import asynccontextmanager

# Import all models to ensure they are registered before create_all
from app.models.user import User
from app.models.workspace import Workspace
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query import QueryLog
from app.models.conversation import Conversation
from app.models.feedback import QueryFeedback
from app.models.evaluation import Evaluation
from app.models.diagnosis import Diagnosis
from app.models.optimization import Recommendation
from app.models.experiment import Experiment
from app.models.retrieval_result import RetrievalResult
from app.models.configuration import Configuration

Base.metadata.create_all(bind=engine)

from app.api import routes_query, routes_documents, routes_conversations, routes_evaluation, routes_diagnostics, routes_research
from app.core.config import settings
from app.rag.ingestion import reindex_all
from app.core.logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing vector stores...")
    db = SessionLocal()
    try:
        reindex_all(db)
        logger.info("Vector stores initialized.")
    finally:
        db.close()
    yield

from starlette.middleware.sessions import SessionMiddleware
from app.api import routes_auth

app = FastAPI(title="AdaptiveRAG", description="A Closed-Loop Autonomous Retrieval Optimization Framework", lifespan=lifespan)

if not settings.session_secret_key:
    raise RuntimeError("Application startup aborted: SESSION_SECRET_KEY is not configured.")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    max_age=settings.session_max_age,
    same_site=settings.session_cookie_same_site,
    https_only=settings.is_cookie_secure
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(routes_auth.router)
app.include_router(routes_documents.router)
app.include_router(routes_query.router)
app.include_router(routes_conversations.router)
app.include_router(routes_evaluation.router)
app.include_router(routes_diagnostics.router)
app.include_router(routes_research.router)

from app.core.auth import get_current_user
from fastapi import HTTPException
from fastapi.responses import RedirectResponse

@app.exception_handler(401)
async def unauthorized_exception_handler(request: Request, exc: HTTPException):
    return RedirectResponse(url="/login", status_code=302)

@app.get("/")
async def home(request: Request, db: Session = Depends(get_db), user = Depends(get_current_user)):
    ws_id = user.workspace_id
    stats = {
        "documents": db.query(Document).filter(Document.workspace_id == ws_id).count(),
        "chunks": db.query(Chunk).filter(Chunk.workspace_id == ws_id).count(),
        "queries_evaluated": db.query(Evaluation).join(QueryLog).filter(QueryLog.workspace_id == ws_id).count(),
        "experiments": db.query(Experiment).filter(Experiment.workspace_id == ws_id).count(),
        "accepted": db.query(Experiment).filter(Experiment.workspace_id == ws_id, Experiment.decision == "ACCEPT").count(),
        "rejected": db.query(Experiment).filter(Experiment.workspace_id == ws_id, Experiment.decision == "REJECT").count(),
        "no_action": db.query(Diagnosis).join(QueryLog).filter(QueryLog.workspace_id == ws_id, Diagnosis.failure_type == "NO_FAILURE").count()
    }
    
    benchmark_stats = None
    import os, json
    results_file = "results/benchmark_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            b_results = json.load(f)
            static_scores = [r["static_indep_score"] for r in b_results]
            adaptive_scores = [r["adaptive_indep_score"] for r in b_results]
            
            avg_static = sum(static_scores) / len(static_scores) if static_scores else 0
            avg_adaptive = sum(adaptive_scores) / len(adaptive_scores) if adaptive_scores else 0
            
            t_stat = 0
            p_value = 1
            try:
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_rel(adaptive_scores, static_scores)
            except ImportError:
                pass
                
            benchmark_stats = {
                "avg_static": avg_static,
                "avg_adaptive": avg_adaptive,
                "interventions": sum(r["adaptive_interventions"] for r in b_results),
                "accepted_experiments": sum(r["adaptive_accepted"] for r in b_results),
                "t_stat": t_stat,
                "p_value": p_value
            }

    recent_queries = db.query(QueryLog).filter(QueryLog.workspace_id == ws_id).order_by(QueryLog.id.desc()).limit(5).all()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "settings": settings, 
        "stats": stats, 
        "recent_queries": recent_queries,
        "benchmark": benchmark_stats
    })

@app.get("/health")
async def health():
    return {"status": "ok"}
