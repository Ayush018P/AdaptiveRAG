import pytest
import json
from unittest.mock import patch
from app.evaluation.evaluator import evaluator

def test_abstention_absent_evidence():
    # Context does not contain the answer. Answer abstains.
    query = "What is the token expiration time?"
    answer = "The information is unavailable."
    context = "The sky is blue."
    
    result = evaluator._evaluate_heuristic(query, answer, context)
    # context score is 0. abstention is correct.
    assert result["faithfulness_score"] == 1.0

def test_unsupported_abstention():
    # Context contains the answer, but answer abstains anyway
    query = "What is the token expiration time?"
    answer = "The information is unavailable."
    context = "The token expiration time is 60 minutes."
    
    result = evaluator._evaluate_heuristic(query, answer, context)
    # context score is high (contains query terms). abstention is incorrect.
    assert result["faithfulness_score"] == 0.0

def test_supported_answer_normal_path():
    query = "What is the token expiration time?"
    answer = "The token expiration time is 60 minutes."
    context = "The authentication uses JWT tokens. The token expiration time is 60 minutes."
    
    result = evaluator._evaluate_heuristic(query, answer, context)
    assert result["faithfulness_score"] > 0.5
    
@patch("app.evaluation.evaluator.generator.generate_json")
def test_malformed_evaluator_output_fallback(mock_generate_json):
    # LLM returns malformed JSON
    mock_generate_json.return_value = "{ invalid json"
    
    query = "What is the token expiration time?"
    answer = "The information is unavailable."
    context = "The sky is blue."
    
    # Should fall back to heuristic (which returns 1.0 for correctly grounded abstention)
    result = evaluator.evaluate(query, answer, context, [])
    assert result["faithfulness_score"] == 1.0
    assert mock_generate_json.call_count == 2 # 1 initial + 1 retry before fallback

@patch("app.evaluation.evaluator.generator.generate_json")
def test_missing_metrics_fallback(mock_generate_json):
    # LLM returns valid JSON but missing fields
    mock_generate_json.return_value = '{"reason": "good"}'
    
    query = "What is the token expiration time?"
    answer = "The information is unavailable."
    context = "The sky is blue."
    
    result = evaluator.evaluate(query, answer, context, [])
    # Falls back to heuristic, does not crash
    assert result["faithfulness_score"] == 1.0
