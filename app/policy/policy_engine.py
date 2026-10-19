from app.core.config_manager import config_manager

MAX_TOP_K = 20

class PolicyEngine:
    def recommend(self, workspace_id: str, diagnosis: dict):
        suggested_actions = diagnosis.get("suggested_actions", ["NO_ACTION"])
        action = suggested_actions[0] if suggested_actions else "NO_ACTION"
        
        active_config = config_manager.get_active_config(workspace_id)
        current_top_k = max(1, min(active_config["top_k"], MAX_TOP_K))
        
        current_config = {
            "top_k": current_top_k,
            "dense_weight": active_config["dense_weight"],
            "bm25_weight": active_config["bm25_weight"]
        }
        
        candidate_config = current_config.copy()
        reason = "System is performing well."
        expected_effect = "Maintain performance."

        if action == "INCREASE_TOP_K":
            candidate_config["top_k"] = min(MAX_TOP_K, current_config["top_k"] + 5)
            reason = "Diagnosis indicates insufficient Top-K or low retrieval relevance."
            expected_effect = "Increase retrieval recall by fetching more chunks."
            
        elif action == "DECREASE_TOP_K":
            candidate_config["top_k"] = max(1, current_config["top_k"] - 3)
            reason = "Retrieval might be too noisy."
            expected_effect = "Increase precision by fetching fewer chunks."
            
        elif action == "ADJUST_BM25_WEIGHT":
            candidate_config["bm25_weight"] = 0.7
            candidate_config["dense_weight"] = 0.3
            reason = "Diagnosis suggests dense retrieval is missing exact keywords."
            expected_effect = "Ground retrieval in exact matches."
            
        elif action == "REGENERATE_WITH_STRICTER_CONTEXT":
            reason = "Hallucination detected."
            expected_effect = "Force the LLM to adhere strictly to the context."
            
        elif action == "FLAG_KNOWLEDGE_BASE":
            reason = "Missing document diagnosed."
            expected_effect = "Alert administrators to missing knowledge."
            
        elif action == "ENABLE_RERANKING":
            reason = "Ranking failure diagnosed."
            expected_effect = "Improve sorting precision of retrieved chunks."
            
        elif action == "REWRITE_QUERY":
            reason = "Query ambiguity diagnosed."
            expected_effect = "Improve semantic matching."
            
        return {
            "action": action,
            "current_config": current_config,
            "candidate_config": candidate_config,
            "reason": reason,
            "expected_effect": expected_effect
        }

policy_engine = PolicyEngine()
