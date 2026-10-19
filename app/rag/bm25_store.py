from rank_bm25 import BM25Okapi
import numpy as np

class BM25Store:
    def __init__(self):
        self.corpus = {}
        self.chunk_ids = {}
        self.bm25 = {}
        
    def _get_or_create(self, workspace_id: str):
        if workspace_id not in self.corpus:
            self.corpus[workspace_id] = []
            self.chunk_ids[workspace_id] = []
            self.bm25[workspace_id] = None
        return self.corpus[workspace_id], self.chunk_ids[workspace_id]

    def fit(self, workspace_id: str, chunk_ids: list, texts: list):
        if not texts:
            return
        corpus, c_ids = self._get_or_create(workspace_id)
        tokenized_corpus = [text.lower().split() for text in texts]
        corpus.extend(tokenized_corpus)
        c_ids.extend(chunk_ids)
        self.bm25[workspace_id] = BM25Okapi(corpus)

    def search(self, workspace_id: str, query: str, top_k: int = 5):
        if workspace_id not in self.bm25 or not self.bm25[workspace_id]:
            return []
            
        bm25_model = self.bm25[workspace_id]
        c_ids = self.chunk_ids[workspace_id]
        
        tokenized_query = query.lower().split()
        scores = bm25_model.get_scores(tokenized_query)
        
        max_score = np.max(scores)
        if max_score == 0:
            max_score = 1
            
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0:
                results.append({
                    "chunk_id": c_ids[idx],
                    "score": float(scores[idx] / max_score)
                })
        return results

    def reset(self, workspace_id: str = None):
        if workspace_id:
            if workspace_id in self.corpus:
                del self.corpus[workspace_id]
                del self.chunk_ids[workspace_id]
                del self.bm25[workspace_id]
        else:
            self.corpus = {}
            self.chunk_ids = {}
            self.bm25 = {}

    def replace_index(self, workspace_id: str, new_corpus: list, new_chunk_ids: list, new_bm25):
        self.corpus[workspace_id] = new_corpus
        self.chunk_ids[workspace_id] = new_chunk_ids
        self.bm25[workspace_id] = new_bm25

bm25_store = BM25Store()
