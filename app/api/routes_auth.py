from fastapi import APIRouter, Request, Depends, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.user import User
from app.models.workspace import Workspace
from app.core.auth import get_password_hash, verify_password
import uuid

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    request.session["user_id"] = user.id
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists"})
    
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password too short"})

    # Create workspace
    new_workspace = Workspace(name=f"{username}'s Workspace")
    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)
    
    # Create user
    new_user = User(
        username=username,
        hashed_password=get_password_hash(password),
        workspace_id=new_workspace.id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    request.session["user_id"] = new_user.id
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

@router.post("/logout")
@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
