from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        if not self.is_loaded:
            logger.info("Loading reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2")
            try:
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                self.is_loaded = True
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                
    def rerank(self, query: str, candidate_chunks: list):
        if not candidate_chunks:
            return []
            
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded:
            # Fallback if model fails to load
            for i, chunk in enumerate(candidate_chunks):
                chunk["pre_rerank_rank"] = i + 1
                chunk["pre_rerank_score"] = chunk.get("score", 0.0)
                chunk["post_rerank_rank"] = i + 1
                chunk["rerank_score"] = chunk.get("score", 0.0)
            return candidate_chunks
            
        pairs = [[query, chunk["text"]] for chunk in candidate_chunks]
        scores = self.model.predict(pairs)
        
        # Attach scores to chunks
        for i, chunk in enumerate(candidate_chunks):
            chunk["pre_rerank_rank"] = i + 1
            chunk["pre_rerank_score"] = chunk.get("score", 0.0)
            chunk["rerank_score"] = float(scores[i])
            # Keep the old score for reference but we'll sort by rerank_score
            
        # Sort by new rerank_score
        reranked = sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Assign post_rerank_rank
        for i, chunk in enumerate(reranked):
            chunk["post_rerank_rank"] = i + 1
            chunk["score"] = chunk["rerank_score"] # Override primary score for downstream
            
        return reranked

reranker_service = RerankerService()
