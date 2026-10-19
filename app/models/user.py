from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from app.models.database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
