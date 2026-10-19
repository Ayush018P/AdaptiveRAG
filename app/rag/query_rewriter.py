from app.rag.generator import generator

class QueryRewriter:
    def rewrite(self, query: str, reason: str = "Query ambiguity diagnosed."):
        prompt = f"""You are an expert search query rewriter for a Retrieval-Augmented Generation (RAG) system.
The original query needs to be rewritten to improve retrieval quality.
Reason for rewriting: {reason}

Original Query: "{query}"

Output ONLY the rewritten query text. Do not include quotes, explanations, or introductory text. Make it optimized for semantic similarity and keyword retrieval."""
        
        try:
            # Reusing the existing generator to make an LLM call for rewriting
            gen_result = generator.generate(query=prompt, context="")
            rewritten = gen_result.get("answer", "")
            return rewritten.strip('"').strip()
        except Exception as e:
            return None # Indicate failure so pipeline can fallback

query_rewriter = QueryRewriter()
