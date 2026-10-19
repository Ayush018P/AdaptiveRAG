from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base

class Diagnosis(Base):
    __tablename__ = 'diagnoses'
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey('queries.id'))
    failure_type = Column(String)
    confidence = Column(Float)
    evidence = Column(Text)
    explanation = Column(Text)
    
    query_log = relationship("QueryLog", back_populates="diagnosis")
