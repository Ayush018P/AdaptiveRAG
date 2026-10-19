import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def compare_results():
    results_file = "results/benchmark_results.json"
    if not os.path.exists(results_file):
        print("Results file not found. Run benchmark first.")
        return
        
    with open(results_file, "r") as f:
        results = json.load(f)
        
    static_scores = [r["static_indep_score"] for r in results]
    adaptive_scores = [r["adaptive_indep_score"] for r in results]
    
    avg_static = np.mean(static_scores)
    avg_adaptive = np.mean(adaptive_scores)
    
    # Statistical Test: Paired T-Test
    t_stat, p_value = stats.ttest_rel(adaptive_scores, static_scores)
    is_significant = p_value < 0.05
    
    # Plotting
    labels = ['Static RAG', 'Adaptive RAG']
    means = [avg_static, avg_adaptive]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, color=['#6c757d', '#0d6efd'], width=0.6)
    
    ax.set_ylabel('Average Independent Score (Cosine Sim)')
    ax.set_title('Static vs Adaptive RAG Performance on Benchmark')
    ax.set_ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')
        
    # Add significance marker if applicable
    if is_significant:
        ax.text(0.5, max(means) + 0.1, f'* p={p_value:.3f}', ha='center', va='bottom', fontweight='bold', color='green', transform=ax.transAxes)
    else:
        ax.text(0.5, max(means) + 0.1, f'ns p={p_value:.3f}', ha='center', va='bottom', color='red', transform=ax.transAxes)

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/benchmark_comparison.png", dpi=300, bbox_inches='tight')
    print("Graph saved to results/benchmark_comparison.png")
    
    print("\n--- Benchmark Summary ---")
    print(f"Total Queries: {len(results)}")
    print(f"Average Static Score:   {avg_static:.3f}")
    print(f"Average Adaptive Score: {avg_adaptive:.3f}")
    print(f"Total Interventions:    {sum(r['adaptive_interventions'] for r in results)}")
    print(f"Total Accepted Experiments: {sum(r['adaptive_accepted'] for r in results)}")
    print(f"\n--- Statistical Significance (Paired T-Test) ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_value:.4f}")
    if is_significant:
        print("Conclusion: The improvement in Adaptive RAG is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("Conclusion: The improvement in Adaptive RAG is NOT statistically significant (p >= 0.05).")
    
if __name__ == "__main__":
    compare_results()
