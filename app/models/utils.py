from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.chunk import Chunk

# Update main.py to fetch stats
def get_stats(db: Session):
    return {
        "documents": db.query(Document).count(),
        "chunks": db.query(Chunk).count()
    }
