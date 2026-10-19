import pytest
from app.evaluation.evaluator import evaluator
from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine
from app.experiment.experiment_engine import experiment_engine
from app.core.config_manager import config_manager
from app.rag.generator import generator
import json

def test_heuristic_fallback():
    # Force generator to return invalid JSON
    original_generate = generator.generate_json
    generator.generate_json = lambda prompt: "INVALID_JSON"
    
    try:
        eval_res = evaluator.evaluate("query", "answer", "context", [])
        assert "faithfulness_score" in eval_res
        assert eval_res["explanation"].startswith("Heuristic evaluation fallback")
    finally:
        generator.generate_json = original_generate

def test_diagnosis_rules():
    # CASE 1: LOW_RETRIEVAL_RELEVANCE
    eval_mock = {
        "failure_detected": True,
        "context_score": 0.2,
        "faithfulness_score": 0.9,
        "relevance_score": 0.5
    }
    chunks_mock = [{"score": 0.8}]
    diag = diagnostic_engine.diagnose("query", chunks_mock, "answer", eval_mock)
    assert diag["failure_type"] == "LOW_RETRIEVAL_RELEVANCE"

    # CASE 2: POSSIBLE_HALLUCINATION
    eval_mock = {
        "failure_detected": True,
        "context_score": 0.8,
        "faithfulness_score": 0.2,
        "relevance_score": 0.5
    }
    diag = diagnostic_engine.diagnose("query", chunks_mock, "answer", eval_mock)
    assert diag["failure_type"] == "POSSIBLE_HALLUCINATION"

def test_policy_rules():
    from app.core.config import settings
    original_top_k = settings.top_k
    settings.top_k = 5
    
    try:
        diag = {"failure_type": "LOW_RETRIEVAL_RELEVANCE", "suggested_actions": ["INCREASE_TOP_K"]}
        hypo = policy_engine.recommend("legacy", diag)
        assert hypo["action"] == "INCREASE_TOP_K"
        assert hypo["candidate_config"]["top_k"] > hypo["current_config"]["top_k"]
    finally:
        settings.top_k = original_top_k

def test_config_rollback():
    from app.core.config import settings
    original_top_k = settings.top_k
    
    # We must reset DB state or just test the logic directly
    base_config = config_manager.get_active_config("legacy")
    base_top_k = base_config["top_k"]
    
    new_v = config_manager.activate_config("legacy", {"top_k": 99})
    assert config_manager.get_active_config("legacy")["top_k"] == 99
    
    config_manager.rollback_config("legacy", new_v - 1)
    # It rolls back to the active config before activate_config.
    # Due to db state leakage across tests, we just check if it rolled back to base_top_k
    assert config_manager.get_active_config("legacy")["top_k"] == base_top_k
