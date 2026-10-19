from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime

class Configuration(Base):
    __tablename__ = 'configurations'
    
    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, index=True)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, default="legacy")
    top_k = Column(Integer)
    dense_weight = Column(Float)
    bm25_weight = Column(Float)
    reranking_enabled = Column(Boolean, default=False)
    query_rewrite_enabled = Column(Boolean, default=False)
    source_experiment = Column(String, nullable=True)
    status = Column(String, default="EXPERIMENTAL") # ACTIVE, SUPERSEDED, EXPERIMENTAL, REJECTED
    created_at = Column(DateTime, default=datetime.utcnow)
    workspace = relationship("Workspace")
