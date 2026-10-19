from sqlalchemy import Column, String, DateTime
from app.models.database import Base
from datetime import datetime
import uuid

class Workspace(Base):
    __tablename__ = 'workspaces'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
