from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime
import uuid

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_bases'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    workspace = relationship("Workspace")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")
