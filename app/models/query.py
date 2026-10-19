from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime

class QueryLog(Base):
    __tablename__ = 'queries'
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    latency = Column(Float)
    llm_calls = Column(Integer, default=0) # Feature 2
    retrieval_config = Column(String)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, default="legacy")
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=True, index=True)
    mode = Column(String, default="standard", nullable=False)
    conversation_id = Column(String, ForeignKey('conversations.id', ondelete='SET NULL'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    answer_status = Column(String, default="SUCCESS", nullable=False)
    adaptive_outcome = Column(String, nullable=True)
    citations_data = Column(Text, nullable=True)

    workspace = relationship("Workspace")
    knowledge_base = relationship("KnowledgeBase")
    conversation = relationship("Conversation", back_populates="queries")
    user = relationship("User")
    feedback = relationship("QueryFeedback", back_populates="query_log", cascade="all, delete-orphan")
    
    # Feature 1
    rewritten_query = Column(String, nullable=True)
    query_rewrite_reason = Column(Text, nullable=True)
    query_rewrite_trigger = Column(String, nullable=True)
    
    evaluation = relationship("Evaluation", back_populates="query_log", uselist=False)
    diagnosis = relationship("Diagnosis", back_populates="query_log", uselist=False)
    recommendation = relationship("Recommendation", back_populates="query_log", uselist=False)
