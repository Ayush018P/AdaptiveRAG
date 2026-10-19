from passlib.context import CryptContext
from fastapi import Request, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_current_user(request: Request, db: Session = Depends(get_db)):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        request.session.clear()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

def validate_workspace(requested_workspace_id: str, user: User):
    if requested_workspace_id and requested_workspace_id != user.workspace_id:
        raise HTTPException(status_code=403, detail="Unauthorized workspace access")
    return user.workspace_id
