from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine

def test_diagnosis_insufficient_topk():
    query = "test query"
    answer = "unknown"
    chunks = [{"score": 0.5, "text": "test"}]
    eval_result = {
        "failure_detected": True,
        "context_score": 0.5, # moderate context coverage
        "faithfulness_score": 1.0,
        "relevance_score": 1.0,
        "overall_score": 0.5
    }
    
    diag = diagnostic_engine.diagnose(query, chunks, answer, eval_result)
    assert diag["failure_type"] == "INSUFFICIENT_TOP_K"
    
def test_policy_increase_topk():
    diag = {"failure_type": "INSUFFICIENT_TOP_K", "suggested_actions": ["INCREASE_TOP_K"]}
    policy = policy_engine.recommend("legacy", diag)
    assert policy["action"] == "INCREASE_TOP_K"
