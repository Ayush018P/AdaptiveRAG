from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.models.database import Base
from datetime import datetime

class QueryFeedback(Base):
    __tablename__ = 'query_feedback'

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey('queries.id', ondelete='CASCADE'), nullable=False, index=True)
    conversation_id = Column(String, ForeignKey('conversations.id', ondelete='CASCADE'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, index=True)
    rating = Column(String, nullable=False)  # "helpful" or "not_helpful"
    reason = Column(String, nullable=True)   # "incorrect_answer", "insufficient_evidence", "irrelevant_response", "citation_problem", "other"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    query_log = relationship("QueryLog", back_populates="feedback")
    workspace = relationship("Workspace")
    user = relationship("User")

    __table_args__ = (
        UniqueConstraint('query_id', 'user_id', name='uq_query_feedback_user'),
    )
