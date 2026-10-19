import pytest
from app.rag.vector_store import faiss_store
from app.rag.bm25_store import bm25_store
from app.rag.hybrid_retriever import HybridRetriever
from app.core.config_manager import config_manager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.database import Base
from app.models.workspace import Workspace
from app.models.document import Document
from app.models.chunk import Chunk
from app.rag.ingestion import reindex_all
from app.core.config_manager import SessionLocal as TestingSessionLocal
import os

def test_vector_and_bm25_store_isolation():
    faiss_store.reset()
    bm25_store.reset()
    
    # Workspace A
    ws_a_id = "workspace_a"
    ws_a_ids = [1, 2, 5]
    # some dummy 384-dimensional vectors
    ws_a_vectors = [[0.1]*384, [0.2]*384, [0.3]*384]
    ws_a_texts = ["Apple is a fruit", "Banana is yellow", "Cherry is red"]
    
    # Workspace B
    ws_b_id = "workspace_b"
    ws_b_ids = [3, 4, 6]
    ws_b_vectors = [[-0.1]*384, [-0.2]*384, [-0.3]*384]
    ws_b_texts = ["Car is a vehicle", "Bus is big", "Train is long"]
    
    # Fit stores
    faiss_store.add_vectors(ws_a_id, ws_a_ids, ws_a_vectors)
    bm25_store.fit(ws_a_id, ws_a_ids, ws_a_texts)
    
    faiss_store.add_vectors(ws_b_id, ws_b_ids, ws_b_vectors)
    bm25_store.fit(ws_b_id, ws_b_ids, ws_b_texts)
    
    # Test FAISS isolation
    query_vector = [0.1]*384
    faiss_results_a = faiss_store.search(ws_a_id, query_vector, top_k=5)
    faiss_results_b = faiss_store.search(ws_b_id, query_vector, top_k=5)
    
    assert len(faiss_results_a) == 3
    assert all(r["chunk_id"] in ws_a_ids for r in faiss_results_a)
    
    assert len(faiss_results_b) == 3
    assert all(r["chunk_id"] in ws_b_ids for r in faiss_results_b)
    
    # Test BM25 isolation
    query_text = "fruit vehicle"
    bm25_results_a = bm25_store.search(ws_a_id, query_text, top_k=5)
    bm25_results_b = bm25_store.search(ws_b_id, query_text, top_k=5)
    
    # "fruit" should match ws_a
    assert any(r["chunk_id"] == 1 for r in bm25_results_a)
    assert not any(r["chunk_id"] in ws_b_ids for r in bm25_results_a)
    
    # "vehicle" should match ws_b
    assert any(r["chunk_id"] == 3 for r in bm25_results_b)
    assert not any(r["chunk_id"] in ws_a_ids for r in bm25_results_b)
    
def test_hybrid_retriever_isolation():
    from unittest.mock import patch
    
    retriever = HybridRetriever()
    
    # Mock settings since we are testing
    with patch('app.rag.hybrid_retriever.settings') as mock_settings:
        mock_settings.top_k = 5
        mock_settings.dense_weight = 0.5
        mock_settings.bm25_weight = 0.5
        
        # We also need to patch embedding_service since hybrid retriever uses it
        with patch('app.rag.hybrid_retriever.embedding_service') as mock_embedding:
            mock_embedding.embed_text.return_value = [0.1] * 384
            
            with patch('app.rag.hybrid_retriever.faiss_store') as mock_faiss:
                with patch('app.rag.hybrid_retriever.bm25_store') as mock_bm25:
                    
                    mock_faiss.search.return_value = [{"chunk_id": 1, "score": 0.8}]
                    mock_bm25.search.return_value = [{"chunk_id": 1, "score": 0.9}]
                    
                    results = retriever.retrieve("workspace_x", "test query")
                    
                    mock_faiss.search.assert_called_once_with("workspace_x", [0.1]*384, 10)
                    mock_bm25.search.assert_called_once_with("workspace_x", "test query", 10)
                    
                    assert len(results["hybrid_results"]) == 1
                    assert results["hybrid_results"][0]["chunk_id"] == 1
                    
def test_reindex_all_isolation():
    db = TestingSessionLocal()
    
    # Add workspaces
    ws1 = Workspace(id="ws1", name="WS1")
    ws2 = Workspace(id="ws2", name="WS2")
    db.add_all([ws1, ws2])
    db.commit()
    
    # Add documents to satisfy foreign keys
    d1 = Document(id=10001, filename="d1.txt", doc_type=".txt", workspace_id="ws1", status="READY")
    d2 = Document(id=10002, filename="d2.txt", doc_type=".txt", workspace_id="ws2", status="READY")
    db.add_all([d1, d2])
    db.commit()

    # Add chunks
    c1 = Chunk(id=10001, document_id=10001, text="Chunk 1", chunk_index=0, page_number=1, workspace_id="ws1")
    c2 = Chunk(id=10002, document_id=10002, text="Chunk 2", chunk_index=0, page_number=1, workspace_id="ws2")
    db.add_all([c1, c2])
    db.commit()
    
    from unittest.mock import patch
    with patch('app.rag.ingestion.embedding_service.embed_documents') as mock_embed:
        mock_embed.return_value = [[0.1]*384]
        
        reindex_all(db)
        
        # Test if they are in faiss and bm25 under correct workspaces
        faiss_results_1 = faiss_store.search("ws1", [0.1]*384)
        assert len(faiss_results_1) == 1
        assert faiss_results_1[0]["chunk_id"] == c1.id
        
        faiss_results_2 = faiss_store.search("ws2", [0.1]*384)
        assert len(faiss_results_2) == 1
        assert faiss_results_2[0]["chunk_id"] == c2.id

    db.close()
