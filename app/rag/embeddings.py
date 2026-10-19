from sentence_transformers import SentenceTransformer
from app.core.config import settings
import os

class EmbeddingService:
    def __init__(self):
        # We allow lazy loading to speed up app startup
        self.model = None
        
    def get_model(self):
        if self.model is None:
            # Avoid re-downloading if possible
            self.model = SentenceTransformer(settings.embedding_model)
        return self.model
        
    def embed_text(self, text: str):
        return self.get_model().encode(text)

    def embed_documents(self, texts: list):
        if not texts:
            return []
        return self.get_model().encode(texts)

embedding_service = EmbeddingService()
