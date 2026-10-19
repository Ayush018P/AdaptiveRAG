from sqlalchemy import Column, Integer, Text, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base

class Evaluation(Base):
    __tablename__ = 'evaluations'
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey('queries.id'))
    relevance_score = Column(Float)
    faithfulness_score = Column(Float)
    completeness_score = Column(Float)
    context_score = Column(Float)
    overall_score = Column(Float)
    failure_detected = Column(Boolean)
    explanation = Column(Text)
    
    query_log = relationship("QueryLog", back_populates="evaluation")
