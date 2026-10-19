from sqlalchemy import Column, Integer, Text, String, ForeignKey
from sqlalchemy.orm import relationship
from app.models.database import Base

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'))
    text = Column(Text)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer)
    workspace_id = Column(String, ForeignKey('workspaces.id'), nullable=False, default="legacy")
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=True, index=True)
    document = relationship("Document", back_populates="chunks")
    workspace = relationship("Workspace")
    knowledge_base = relationship("KnowledgeBase")

