import json
import re
from pydantic import BaseModel, Field, ValidationError
from app.rag.generator import generator

class EvaluationResult(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    context_relevance: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_completeness: float = Field(..., ge=0.0, le=1.0)
    reason: str
    missing_information: list[str] = Field(default_factory=list)

class Evaluator:
    def evaluate(self, query: str, answer: str, context: str, retrieved_chunks: list):
        # LLM Evaluator Logic
        prompt = f"""
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
Please evaluate the following interaction:

QUERY: {query}
ANSWER: {answer}
RETRIEVED CONTEXT: {context}

Please provide your evaluation in strict JSON format with the following keys and float values between 0.0 and 1.0:
- "faithfulness": Does the answer only contain facts found in the retrieved context? (1.0 = highly faithful, 0.0 = hallucination)
- "context_relevance": Are the retrieved chunks relevant to the query? (1.0 = highly relevant, 0.0 = irrelevant)
- "answer_relevance": Does the answer directly address the query? (1.0 = highly relevant, 0.0 = irrelevant/evasive)
- "context_completeness": Does the retrieved context contain ALL the information needed to fully answer the query?
- "reason": A brief explanation of the scores (string).
- "missing_information": A list of strings detailing any information missing from the context needed to fully answer the query.
"""
        
        for attempt in range(2):
            try:
                response_str = generator.generate_json(prompt)
                
                # Strip markdown code blocks if any slipped through
                if response_str.startswith("```json"):
                    response_str = response_str[7:]
                if response_str.endswith("```"):
                    response_str = response_str[:-3]
                    
                data = json.loads(response_str.strip())
                result = EvaluationResult(**data)
                
                overall = (result.faithfulness + result.context_relevance + result.answer_relevance + result.context_completeness) / 4.0
                
                # Explicit trigger logic:
                # An overall score >= 0.65 is generally good, but if a specific critical metric
                # like context relevance or faithfulness is too low, we should still trigger adaptation.
                failure = (
                    overall < 0.65 or 
                    result.faithfulness < 0.7 or 
                    result.context_relevance < 0.8
                )
                
                return {
                    "faithfulness_score": result.faithfulness,
                    "context_score": result.context_relevance, # Mapping context_relevance to context_score for DB
                    "relevance_score": result.answer_relevance, # Mapping answer_relevance to relevance_score for DB
                    "completeness_score": result.context_completeness,
                    "overall_score": float(overall),
                    "failure_detected": failure,
                    "explanation": result.reason,
                    "missing_information": result.missing_information
                }
            except (json.JSONDecodeError, ValidationError, Exception) as e:
                prompt += f"\n\nCRITICAL: Your previous response was invalid JSON or failed ({str(e)}). Ensure it is perfectly formatted JSON matching the schema."
                
        # Fallback to heuristics
        print("Falling back to heuristic evaluator")
        return self._evaluate_heuristic(query, answer, context)
        
    def _evaluate_heuristic(self, query: str, answer: str, context: str):
        query_terms = set(re.findall(r'\w+', query.lower()))
        answer_terms = set(re.findall(r'\w+', answer.lower()))
        relevance_score = len(query_terms.intersection(answer_terms)) / max(len(query_terms), 1)
        
        context_terms = set(re.findall(r'\w+', context.lower()))
        context_score = len(query_terms.intersection(context_terms)) / max(len(query_terms), 1)
        
        is_abstention = "unavailable" in answer.lower()
        if is_abstention:
            if context_score < 0.5:
                faithfulness_score = 1.0  # Correctly grounded abstention
            else:
                faithfulness_score = 0.0  # Unsupported answer
        else:
            faithfulness_score = len(answer_terms.intersection(context_terms)) / max(len(answer_terms), 1)
            
        completeness_score = (relevance_score + faithfulness_score) / 2
        overall_score = (relevance_score + faithfulness_score + context_score + completeness_score) / 4
        
        failure = (
            overall_score < 0.65 or 
            faithfulness_score < 0.7 or 
            context_score < 0.8
        )
        
        return {
            "relevance_score": float(relevance_score),
            "faithfulness_score": float(faithfulness_score),
            "completeness_score": float(completeness_score),
            "context_score": float(context_score),
            "overall_score": float(overall_score),
            "failure_detected": failure,
            "explanation": f"Heuristic evaluation fallback. Overall: {overall_score:.2f}",
            "missing_information": []
        }

evaluator = Evaluator()
