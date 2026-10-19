from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
from app.rag.embeddings import embedding_service
from app.core.config import settings

class HybridRetriever:
    def retrieve(self, workspace_id: str, query: str, top_k: int = None, dense_weight: float = None, bm25_weight: float = None, kb_id: str = None):
        top_k = top_k or settings.top_k
        dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
        bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight

        index_key = f"{workspace_id}:{kb_id}" if kb_id else workspace_id

        # Dense Retrieval
        query_vector = embedding_service.embed_text(query)
        dense_results = faiss_store.search(index_key, query_vector, top_k * 2) # Get more candidates

        # BM25 Retrieval
        bm25_results = bm25_store.search(index_key, query, top_k * 2)

        # Merge results
        combined_data = {}
        
        for res in dense_results:
            chunk_id = res["chunk_id"]
            if chunk_id not in combined_data:
                combined_data[chunk_id] = {"dense_score": 0.0, "bm25_score": 0.0, "combined": 0.0}
            combined_data[chunk_id]["dense_score"] = res["score"]
            combined_data[chunk_id]["combined"] += (res["score"] * dense_weight)
            
        for res in bm25_results:
            chunk_id = res["chunk_id"]
            if chunk_id not in combined_data:
                combined_data[chunk_id] = {"dense_score": 0.0, "bm25_score": 0.0, "combined": 0.0}
            combined_data[chunk_id]["bm25_score"] = res["score"]
            combined_data[chunk_id]["combined"] += (res["score"] * bm25_weight)

        # Sort and get Top-K
        sorted_results = sorted(combined_data.items(), key=lambda item: item[1]["combined"], reverse=True)[:top_k]
        
        final_results = [{
            "chunk_id": chunk_id, 
            "score": data["combined"],
            "dense_score": data["dense_score"],
            "bm25_score": data["bm25_score"]
        } for chunk_id, data in sorted_results]
        
        return {
            "dense_results": dense_results,
            "bm25_results": bm25_results,
            "hybrid_results": final_results
        }

hybrid_retriever = HybridRetriever()
