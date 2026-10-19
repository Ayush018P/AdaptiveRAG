from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base

class Recommendation(Base):
    __tablename__ = 'recommendations'
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey('queries.id'))
    current_config = Column(String)
    recommended_change = Column(String)
    reason = Column(Text)
    expected_purpose = Column(Text)
    
    query_log = relationship("QueryLog", back_populates="recommendation")
