from sqlalchemy.orm import Session
from app.rag.hybrid_retriever import hybrid_retriever
from app.rag.generator import generator
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.rag.query_rewriter import query_rewriter
from app.rag.reranker import reranker_service
from app.core.config import settings
import time

class RAGPipeline:
    def run(self, workspace_id: str, query: str, db: Session, force_rewrite: bool = False, rewrite_reason: str = None, kb_id: str = None):
        start_time = time.time()
        
        if not workspace_id:
            raise ValueError("Workspace ID is required for RAG pipeline.")
            
        # 0. Check for empty knowledge base upfront
        if kb_id:
            ready_doc_count = db.query(Document).filter(
                Document.workspace_id == workspace_id,
                Document.kb_id == kb_id,
                Document.status == "READY"
            ).count()
            if ready_doc_count == 0:
                total_latency = time.time() - start_time
                return {
                    "query": query,
                    "rewritten_query": None,
                    "answer": "Selected knowledge base is empty. Please upload documents before querying.",
                    "answer_status": "EMPTY_KNOWLEDGE_BASE",
                    "generation_error": None,
                    "llm_calls": 0,
                    "token_usage": None,
                    "retrieval_latency": 0.0,
                    "generation_latency": 0.0,
                    "total_latency": total_latency,
                    "retrieved_chunks": [],
                    "citations": [],
                    "retrieval_config": {
                        "top_k": settings.top_k,
                        "dense_weight": settings.dense_weight,
                        "bm25_weight": settings.bm25_weight,
                        "enable_reranker": settings.enable_reranker,
                        "query_rewrite_enabled": getattr(settings, 'query_rewrite_enabled', False)
                    },
                    "latency": total_latency,
                    "context": ""
                }

        # 1. Query Rewriting
        actual_query = query
        rewritten_query = None
        
        if getattr(settings, 'query_rewrite_enabled', False) or force_rewrite:
            rewritten_query = query_rewriter.rewrite(query, reason=rewrite_reason or "Configured to rewrite.")
            if rewritten_query:
                actual_query = rewritten_query
        
        # 2. Retrieval
        retrieval_start = time.time()
        retrieval_res = hybrid_retriever.retrieve(workspace_id, actual_query, kb_id=kb_id)
        hybrid_results = retrieval_res["hybrid_results"]
        
        # 2.5 Fetch and validate full chunk and document objects
        verified_chunks = []
        for res in hybrid_results:
            chunk = db.query(Chunk).filter(Chunk.id == res["chunk_id"]).first()
            if not chunk:
                continue
            # Enforce strict workspace and KB scoping
            if chunk.workspace_id != workspace_id:
                continue
            if kb_id and chunk.kb_id != kb_id:
                continue

            doc = db.query(Document).filter(Document.id == chunk.document_id).first()
            if not doc:
                continue
            if doc.workspace_id != workspace_id:
                continue
            if kb_id and doc.kb_id != kb_id:
                continue
            if doc.status != "READY":
                continue

            page_num = chunk.page_number if (chunk.page_number is not None and chunk.page_number > 0 and doc.doc_type == ".pdf") else None

            verified_chunks.append({
                "id": chunk.id,
                "document_id": doc.id,
                "document": doc.filename,
                "doc_type": doc.doc_type,
                "page_number": page_num,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "score": res["score"],
                "dense_score": res.get("dense_score", 0.0),
                "bm25_score": res.get("bm25_score", 0.0)
            })
        
        # 3. Reranking
        if settings.enable_reranker:
            retrieved_chunks = reranker_service.rerank(actual_query, verified_chunks)
        else:
            retrieved_chunks = verified_chunks
            
        # Top-K
        retrieved_chunks = retrieved_chunks[:settings.top_k]
        retrieval_latency = time.time() - retrieval_start
        
        # 4. Handle No Relevant Chunks / Insufficient Evidence
        if not retrieved_chunks:
            total_latency = time.time() - start_time
            return {
                "query": query,
                "rewritten_query": rewritten_query,
                "answer": "No relevant information found in the knowledge base for this query.",
                "answer_status": "NO_RELEVANT_CHUNKS",
                "generation_error": None,
                "llm_calls": 0,
                "token_usage": None,
                "retrieval_latency": retrieval_latency,
                "generation_latency": 0.0,
                "total_latency": total_latency,
                "retrieved_chunks": [],
                "citations": [],
                "retrieval_config": {
                    "top_k": settings.top_k,
                    "dense_weight": settings.dense_weight,
                    "bm25_weight": settings.bm25_weight,
                    "enable_reranker": settings.enable_reranker,
                    "query_rewrite_enabled": getattr(settings, 'query_rewrite_enabled', False)
                },
                "latency": total_latency,
                "context": ""
            }

        # 5. Construct Verifiable Citations & Numbered Context
        citations = []
        context_texts = []
        for i, chunk_dict in enumerate(retrieved_chunks, 1):
            page_str = f", Page {chunk_dict['page_number']}" if chunk_dict.get('page_number') else ""
            context_texts.append(f"[Source {i}: {chunk_dict['document']}{page_str}] {chunk_dict['text']}")
            
            citations.append({
                "citation_id": i,
                "chunk_id": chunk_dict["id"],
                "document_id": chunk_dict["document_id"],
                "filename": chunk_dict["document"],
                "doc_type": chunk_dict["doc_type"],
                "page_number": chunk_dict.get("page_number"),
                "chunk_index": chunk_dict.get("chunk_index"),
                "score": chunk_dict["score"],
                "dense_score": chunk_dict.get("dense_score", 0.0),
                "bm25_score": chunk_dict.get("bm25_score", 0.0),
                "excerpt": chunk_dict["text"][:240] + ("..." if len(chunk_dict["text"]) > 240 else ""),
                "text": chunk_dict["text"]
            })
                
        context = "\n\n".join(context_texts)
        
        # 6. Generation
        gen_start = time.time()
        gen_result = generator.generate(actual_query, context)
        answer = gen_result.get("answer", "")
        llm_calls = gen_result.get("llm_calls", 1)
        token_usage = gen_result.get("token_usage", None)
        generation_latency = time.time() - gen_start
        
        # Determine Answer Status & Errors
        is_success = gen_result.get("success", True)
        error_msg = gen_result.get("error")
        if (not is_success) or answer == "Failed to generate answer due to LLM error." or answer.startswith("Failed to generate answer:"):
            answer_status = "PROVIDER_FAILURE"
            if not error_msg:
                error_msg = answer
        else:
            answer_status = "SUCCESS"

        total_latency = time.time() - start_time
        
        return {
            "query": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "answer_status": answer_status,
            "generation_error": error_msg if answer_status == "PROVIDER_FAILURE" else None,
            "llm_calls": llm_calls,
            "token_usage": token_usage,
            "retrieval_latency": retrieval_latency,
            "generation_latency": generation_latency,
            "total_latency": total_latency,
            "retrieved_chunks": retrieved_chunks,
            "citations": citations,
            "retrieval_config": {
                "top_k": settings.top_k,
                "dense_weight": settings.dense_weight,
                "bm25_weight": settings.bm25_weight,
                "enable_reranker": settings.enable_reranker,
                "query_rewrite_enabled": getattr(settings, 'query_rewrite_enabled', False)
            },
            "latency": total_latency,
            "context": context
        }

rag_pipeline = RAGPipeline()
