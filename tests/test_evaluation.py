from app.evaluation.evaluator import evaluator

def test_evaluation_good_match():
    query = "What is the token expiration time?"
    answer = "The token expiration time is 60 minutes."
    context = "The authentication uses JWT tokens. Tokens expire in 60 minutes."
    chunks = [{"score": 0.8, "text": context}]
    
    result = evaluator.evaluate(query, answer, context, chunks)
    assert result["relevance_score"] > 0
    assert result["faithfulness_score"] > 0
    assert result["failure_detected"] is False

def test_evaluation_hallucination():
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    context = "The document discusses dog breeds."
    chunks = [{"score": 0.1, "text": context}]
    
    result = evaluator.evaluate(query, answer, context, chunks)
    assert result["faithfulness_score"] == 0.0
    assert result["failure_detected"] is True
