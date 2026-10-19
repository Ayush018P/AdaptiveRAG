from sqlalchemy import Column, Integer, String, Float
from app.models.database import Base

class RetrievalResult(Base):
    __tablename__ = 'retrieval_results'
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer) # avoiding ForeignKey for simplicity since it's just a log
    chunk_id = Column(Integer)
    score = Column(Float)
    method = Column(String)
