from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime

class Experiment(Base):
    __tablename__ = 'experiments'
    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, default="legacy")
    query_id = Column(Integer, index=True)
    failure_type = Column(String)
    hypothesis = Column(Text)
    baseline_configuration = Column(Text)
    candidate_configuration = Column(Text)
    baseline_metrics = Column(Text)
    candidate_metrics = Column(Text)
    score_delta = Column(Float)
    latency_delta = Column(Float)
    decision = Column(String) # ACCEPT, REJECT
    acceptance_reason = Column(Text, nullable=True) # Feature 2
    baseline_llm_calls = Column(Integer, default=0) # Feature 2
    candidate_llm_calls = Column(Integer, default=0) # Feature 2
    created_at = Column(DateTime, default=datetime.utcnow)
    workspace = relationship("Workspace")
