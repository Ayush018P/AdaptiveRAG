class DiagnosticEngine:
    def diagnose(self, query: str, retrieved_chunks: list, answer: str, eval_result: dict):
        if not eval_result["failure_detected"]:
            return {
                "failure_type": "NO_FAILURE",
                "confidence": 1.0,
                "evidence": "Evaluation overall score is above threshold.",
                "explanation": "The system performed well across all metrics.",
                "suggested_actions": ["NO_ACTION"]
            }

        context_score = eval_result["context_score"] # Context Relevance
        faithfulness = eval_result["faithfulness_score"]
        relevance_score = eval_result["relevance_score"] # Answer Relevance
        
        top_score = retrieved_chunks[0]['score'] if retrieved_chunks else 0.0

        # CASE 1: Low Retrieval Relevance
        if context_score < 0.4 and faithfulness >= 0.75:
            return {
                "failure_type": "LOW_RETRIEVAL_RELEVANCE",
                "confidence": 0.85,
                "evidence": f"Context relevance is {context_score:.2f}, but faithfulness is high ({faithfulness:.2f}).",
                "explanation": "The answer is relatively grounded in the retrieved context, but the retrieved context itself has poor relevance to the query.",
                "suggested_actions": ["INCREASE_TOP_K", "ADJUST_BM25_WEIGHT"]
            }

        # CASE 2: Hallucination
        if context_score >= 0.6 and faithfulness < 0.5:
            return {
                "failure_type": "POSSIBLE_HALLUCINATION",
                "confidence": 0.90,
                "evidence": f"Context relevance is {context_score:.2f} but faithfulness is {faithfulness:.2f}.",
                "explanation": "The answer contains information not found in the retrieved context despite relevant context being retrieved.",
                "suggested_actions": ["REGENERATE_WITH_STRICTER_CONTEXT"]
            }

        # CASE 3: Insufficient Top K
        if 0.4 <= context_score < 0.8 and len(retrieved_chunks) > 0:
            return {
                "failure_type": "INSUFFICIENT_TOP_K",
                "confidence": 0.75,
                "evidence": f"Context relevance is moderate ({context_score:.2f}) with multiple chunks.",
                "explanation": "Multiple relevant chunks exist, but useful information appears to be beyond the Top-K cutoff.",
                "suggested_actions": ["INCREASE_TOP_K"]
            }

        # CASE 4: Ranking Failure
        if context_score < 0.5 and top_score > 0.0 and len(retrieved_chunks) > 1:
            return {
                "failure_type": "RANKING_FAILURE",
                "confidence": 0.7,
                "evidence": f"Context relevance is low ({context_score:.2f}) despite retrieving {len(retrieved_chunks)} chunks with max score {top_score:.2f}.",
                "explanation": "Chunks are retrieved but the most relevant information might be ranked lower.",
                "suggested_actions": ["ENABLE_RERANKING"]
            }

        # CASE 5: Missing Document
        if context_score < 0.3 and top_score < 0.3:
            return {
                "failure_type": "POSSIBLE_MISSING_DOCUMENT",
                "confidence": 0.80,
                "evidence": f"Context relevance is very low ({context_score:.2f}) and top chunk score is low ({top_score:.2f}).",
                "explanation": "No highly relevant information exists in the indexed documents.",
                "suggested_actions": ["FLAG_KNOWLEDGE_BASE"]
            }
            
        # CASE 6: Ambiguity / Generic Weakness
        return {
            "failure_type": "QUERY_AMBIGUITY",
            "confidence": 0.5,
            "evidence": f"Context score: {context_score:.2f}, Faithfulness: {faithfulness:.2f}, Relevance: {relevance_score:.2f}",
            "explanation": "Retrieval scores are weak across all methods and query might be vague or underspecified.",
            "suggested_actions": ["REWRITE_QUERY"]
        }

diagnostic_engine = DiagnosticEngine()
