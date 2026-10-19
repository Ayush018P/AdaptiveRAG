from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.query import QueryLog
from app.core.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/evaluation")
async def evaluation_page(request: Request, db: Session = Depends(get_db), user=Depends(get_current_user)):
    ws_id = user.workspace_id
    evals = db.query(QueryLog).filter(QueryLog.workspace_id == ws_id).order_by(QueryLog.id.desc()).all()
    return templates.TemplateResponse("evaluation.html", {"request": request, "evals": evals})

@router.get("/evaluation/{query_id}")
async def get_evaluation(query_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    pass
