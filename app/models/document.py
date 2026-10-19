from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    doc_type = Column(String)
    source_path = Column(String)
    ingestion_timestamp = Column(DateTime, default=datetime.utcnow)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, default="legacy")
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=True, index=True)
    status = Column(String, default="INGESTED")
    error_message = Column(Text, nullable=True)
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    workspace = relationship("Workspace")
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
