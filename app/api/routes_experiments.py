from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.experiment import Experiment
from app.models.query import QueryLog

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/experiments")
async def get_experiments(request: Request, db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.timestamp.desc()).limit(100).all()
    
    # We might need to attach the query string for each experiment
    for exp in experiments:
        query_log = db.query(QueryLog).filter(QueryLog.id == exp.query_id).first()
        exp.query_text = query_log.query if query_log else "Unknown Query"
        
    return templates.TemplateResponse("experiments.html", {
        "request": request,
        "experiments": experiments
    })
