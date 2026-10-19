from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime
import uuid

class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    title = Column(String, nullable=False)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    workspace = relationship("Workspace")
    user = relationship("User")
    knowledge_base = relationship("KnowledgeBase")
    queries = relationship("QueryLog", back_populates="conversation", cascade="all, delete-orphan", order_by="QueryLog.timestamp.asc()")
