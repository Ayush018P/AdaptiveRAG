import os
import sys
import json
import csv
import time
import numpy as np
from datetime import datetime

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import SessionLocal
from app.rag.pipeline import rag_pipeline
from app.evaluation.evaluator import evaluator
from app.diagnosis.diagnostic_engine import diagnostic_engine
from app.policy.policy_engine import policy_engine
from app.experiment.experiment_engine import experiment_engine
from app.core.config_manager import config_manager
from app.rag.ingestion import reindex_all
from sentence_transformers import SentenceTransformer, util

# Load the independent evaluation model
eval_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_independent_score(generated_answer: str, expected_answer: str) -> float:
    if expected_answer.lower() in ["the information is unavailable.", "none"]:
        # Handle absent knowledge base scenarios
        if "unavailable" in generated_answer.lower():
            return 1.0
        return 0.0
        
    emb1 = eval_model.encode(generated_answer, convert_to_tensor=True)
    emb2 = eval_model.encode(expected_answer, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emb1, emb2)
    return float(cosine_scores[0][0])

def run_query_static(query: str, expected_answer: str, db):
    # Just run the baseline RAG
    config_manager.activate_config({"top_k": 5, "dense_weight": 0.65, "bm25_weight": 0.35})
    result = rag_pipeline.run(query, db)
    eval_result = evaluator.evaluate(query, result["answer"], result["context"], result["retrieved_chunks"])
    
    indep_score = get_independent_score(result["answer"], expected_answer)
    
    return {
        "answer": result["answer"],
        "latency": result.get("total_latency", result.get("latency", 0)),
        "eval_result": eval_result,
        "interventions": 0,
        "accepted_experiments": 0,
        "independent_score": indep_score
    }

def run_query_adaptive(query: str, expected_answer: str, db):
    # Reset config for fair comparison
    config_manager.activate_config({"top_k": 5, "dense_weight": 0.65, "bm25_weight": 0.35})
    
    baseline_result = rag_pipeline.run(query, db)
    baseline_eval = evaluator.evaluate(query, baseline_result["answer"], baseline_result["context"], baseline_result["retrieved_chunks"])
    
    current_result = baseline_result
    current_eval = baseline_eval
    
    attempts = 0
    max_attempts = 2
    interventions = 0
    accepted = 0
    
    while current_eval["failure_detected"] and attempts < max_attempts:
        attempts += 1
        
        diagnosis = diagnostic_engine.diagnose(query, current_result["retrieved_chunks"], current_result["answer"], current_eval)
        hypothesis = policy_engine.recommend(diagnosis)
        
        if hypothesis["action"] in ["NO_ACTION", "FLAG_KNOWLEDGE_BASE"]:
            break
            
        interventions += 1
        exp_res = experiment_engine.run_experiment(query, 999, hypothesis, db, current_result, current_eval)
        
        if exp_res["decision"] == "ACCEPT":
            current_result = exp_res["candidate_result"]
            current_eval = exp_res["candidate_eval"]
            accepted += 1
        else:
            break
            
    indep_score = get_independent_score(current_result["answer"], expected_answer)
            
    return {
        "answer": current_result["answer"],
        "latency": current_result.get("total_latency", current_result.get("latency", 0)), # Assuming final candidate latency
        "eval_result": current_eval,
        "interventions": interventions,
        "accepted_experiments": accepted,
        "independent_score": indep_score
    }

def run_benchmark():
    benchmark_file = "data/benchmark/benchmark.json"
    if not os.path.exists(benchmark_file):
        print("Benchmark file not found.")
        return
        
    with open(benchmark_file, "r") as f:
        questions = json.load(f)
        
    db = SessionLocal()
    
    # Initialize Vector Stores
    print("Initializing Vector Stores...")
    reindex_all(db)
    
    results = []
    
    for q in questions:
        print(f"\nProcessing Q{q['id']}: {q['question']}")
        
        # Mode 1: Static
        print("  -> Running Static RAG...")
        t0 = time.time()
        static_res = run_query_static(q['question'], q['expected_answer'], db)
        t_static = time.time() - t0
        
        # Mode 2: Adaptive
        print("  -> Running Adaptive RAG...")
        t0 = time.time()
        adaptive_res = run_query_adaptive(q['question'], q['expected_answer'], db)
        t_adaptive = time.time() - t0
        
        results.append({
            "id": q["id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            
            "static_faithfulness": static_res["eval_result"]["faithfulness_score"],
            "static_context_rel": static_res["eval_result"]["context_score"],
            "static_answer_rel": static_res["eval_result"]["relevance_score"],
            "static_latency": static_res["latency"],
            "static_indep_score": static_res["independent_score"],
            
            "adaptive_faithfulness": adaptive_res["eval_result"]["faithfulness_score"],
            "adaptive_context_rel": adaptive_res["eval_result"]["context_score"],
            "adaptive_answer_rel": adaptive_res["eval_result"]["relevance_score"],
            "adaptive_latency": adaptive_res["latency"],
            "adaptive_interventions": adaptive_res["interventions"],
            "adaptive_accepted": adaptive_res["accepted_experiments"],
            "adaptive_indep_score": adaptive_res["independent_score"],
            
            "score_delta": adaptive_res["independent_score"] - static_res["independent_score"]
        })
        
    db.close()
    
    # Save Results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open(f"{results_dir}/benchmark_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nBenchmark complete! Saved to {results_dir}/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
