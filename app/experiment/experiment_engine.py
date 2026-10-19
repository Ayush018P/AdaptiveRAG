import json
from app.rag.pipeline import rag_pipeline
from app.evaluation.evaluator import evaluator
from app.core.config_manager import config_manager
from sqlalchemy.orm import Session
from app.models.experiment import Experiment

class ExperimentEngine:
    def run_experiment(self, workspace_id: str, query: str, query_id: int, hypothesis: dict, db: Session, baseline_result: dict, baseline_eval: dict, kb_id: str = None):
        baseline_config = hypothesis["current_config"]
        candidate_config = hypothesis["candidate_config"]
        
        # 1. Activate Candidate Config
        config_manager.activate_config(workspace_id, candidate_config)
        
        # 2. Run Pipeline scoped to KB
        candidate_result = rag_pipeline.run(workspace_id, query, db, kb_id=kb_id)
        
        # 3. Evaluate Candidate
        import time
        eval_start = time.time()
        candidate_eval = evaluator.evaluate(
            query=query,
            answer=candidate_result["answer"],
            context=candidate_result["context"],
            retrieved_chunks=candidate_result["retrieved_chunks"]
        )
        eval_latency = time.time() - eval_start
        candidate_result["evaluation_latency"] = eval_latency
        
        # 4. Compare
        baseline_score = baseline_eval["overall_score"]
        candidate_score = candidate_eval["overall_score"]
        score_delta = candidate_score - baseline_score
        
        baseline_latency = baseline_result.get("total_latency", 0.0) + baseline_result.get("evaluation_latency", 0.0)
        candidate_latency = candidate_result.get("total_latency", 0.0) + candidate_result.get("evaluation_latency", 0.0)
        latency_delta = candidate_latency - baseline_latency
        
        # Multi-objective policy
        decision = "REJECT"
        reason = ""
        
        if candidate_result.get("answer_status") == "PROVIDER_FAILURE":
            decision = "REJECT"
            reason = "Candidate run encountered LLM provider failure."
        elif score_delta > 0.05 and latency_delta < 0.5:
            decision = "ACCEPT"
            reason = "Candidate improved quality significantly while remaining within latency tolerance."
        elif abs(score_delta) <= 0.05 and latency_delta < -0.1:
            decision = "ACCEPT"
            reason = "Candidate maintained quality while improving latency."
        elif score_delta > 0.1:
            decision = "ACCEPT"
            reason = "Candidate improved quality substantially despite latency increase."
        else:
            decision = "REJECT"
            reason = "Candidate failed to improve quality or violated latency constraints."
            
        # 5. Save Experiment
        experiment = Experiment(
            workspace_id=workspace_id,
            query_id=query_id,
            failure_type=hypothesis.get("failure_type", "UNKNOWN"),
            hypothesis=json.dumps(hypothesis),
            baseline_configuration=json.dumps(baseline_config),
            candidate_configuration=json.dumps(candidate_config),
            baseline_metrics=json.dumps(baseline_eval),
            candidate_metrics=json.dumps(candidate_eval),
            score_delta=score_delta,
            latency_delta=latency_delta,
            baseline_llm_calls=baseline_result.get("llm_calls", 1) + 1, # +1 for evaluation
            candidate_llm_calls=candidate_result.get("llm_calls", 1) + 1, # +1 for evaluation
            decision=decision,
            acceptance_reason=reason
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)
        
        # 6. Apply Decision
        if decision == "REJECT":
            config_manager.activate_config(
                workspace_id,
                baseline_config,
                source_experiment=f"Rollback rejected experiment for query {query_id}"
            )
            
        return {
            "experiment_id": experiment.id,
            "decision": decision,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "score_delta": score_delta,
            "baseline_latency": baseline_latency,
            "candidate_latency": candidate_latency,
            "latency_delta": latency_delta,
            "candidate_result": candidate_result,
            "candidate_eval": candidate_eval
        }

experiment_engine = ExperimentEngine()
