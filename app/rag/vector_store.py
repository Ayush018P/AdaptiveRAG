import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.indices = {}
        self.chunk_ids = {}

    def _get_or_create(self, workspace_id: str):
        if workspace_id not in self.indices:
            self.indices[workspace_id] = faiss.IndexFlatL2(self.dimension)
            self.chunk_ids[workspace_id] = []
        return self.indices[workspace_id], self.chunk_ids[workspace_id]

    def add_vectors(self, workspace_id: str, chunk_ids: list, vectors):
        if len(vectors) == 0:
            return
        index, c_ids = self._get_or_create(workspace_id)
        vectors_np = np.array(vectors).astype('float32')
        index.add(vectors_np)
        c_ids.extend(chunk_ids)

    def search(self, workspace_id: str, query_vector, top_k: int = 5):
        if workspace_id not in self.indices or self.indices[workspace_id].ntotal == 0:
            return []
        index = self.indices[workspace_id]
        c_ids = self.chunk_ids[workspace_id]
        
        query_np = np.array([query_vector]).astype('float32')
        distances, indices = index.search(query_np, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                similarity = 1 / (1 + float(dist))
                results.append({"chunk_id": c_ids[idx], "score": similarity})
                
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

    def reset(self, workspace_id: str = None):
        if workspace_id:
            if workspace_id in self.indices:
                del self.indices[workspace_id]
                del self.chunk_ids[workspace_id]
        else:
            self.indices = {}
            self.chunk_ids = {}

    def replace_index(self, workspace_id: str, new_index, new_chunk_ids: list):
        self.indices[workspace_id] = new_index
        self.chunk_ids[workspace_id] = new_chunk_ids

faiss_store = FAISSVectorStore()
