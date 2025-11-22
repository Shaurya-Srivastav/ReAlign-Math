
import argparse
import json
import os
import re
import time
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from our refactored module
from local_llm_evaluator import get_llm_reasoning, parse_reasoning_to_steps, StepAligner, EMBEDDING_MODEL

# --- Configuration ---
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"
DEFAULT_API_ENDPOINT = "http://127.0.0.1:1234/v1/completions"
DATASET_NAME = "EleutherAI/hendrycks_math"
RESULTS_FILE = "realign_benchmark_results.jsonl"
REPORT_DIR = "Benchmarking/figures-and-statistics"

# --- Helper Functions ---

def extract_boxed_answer(text: str) -> str | None:
    """
    Extracts the content inside \boxed{...}.
    Handles nested braces to some extent.
    """
    if not text:
        return None
        
    # Simple regex for non-nested boxed
    match = re.search(r'\\boxed\{([^{}]*)\}', text)
    if match:
        return match.group(1)
        
    # If simple regex fails, try a more robust brace counting method
    start_idx = text.find(r'\boxed{')
    if start_idx == -1:
        return None
        
    content_start = start_idx + 7 # len("\boxed{")
    brace_count = 1
    for i in range(content_start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            
        if brace_count == 0:
            return text[content_start:i]
            
    return None

def grade_correctness_llm(question: str, benchmark_answer: str, llm_answer: str, 
                          model_name: str, api_endpoint: str, mock: bool = False) -> tuple[bool, str]:
    """
    Asks an LLM to grade the correctness of the answer.
    Returns (is_correct, reasoning).
    """
    if mock:
        return True, "Mock grading: Correct"

    prompt = f"""You are a strict math grader. 

Problem:
{question}

Correct Answer:
{benchmark_answer}

Student Answer:
{llm_answer}

Is the Student Answer correct? The student answer might be formatted differently but must be mathematically equivalent.
Reply with exactly one line: "CORRECT" or "INCORRECT", followed by a short explanation on the next line.
"""
    
    response = get_llm_reasoning(prompt, mock=mock, model_name=model_name, 
                                 api_endpoint=api_endpoint, temperature=0.0, max_tokens=100)
    
    if not response:
        return False, "LLM Grading Failed"
        
    lines = response.strip().split('\n')
    verdict = lines[0].strip().upper()
    reasoning = "\n".join(lines[1:]) if len(lines) > 1 else ""
    
    is_correct = "CORRECT" in verdict and "INCORRECT" not in verdict
    return is_correct, reasoning

# --- Reporting Module ---

def generate_heatmap(aligner, bench_steps, llm_steps, title, filename):
    """Generates and saves a similarity heatmap."""
    if not bench_steps or not llm_steps:
        return
        
    sim_matrix = aligner.compute_similarity_matrix(bench_steps, llm_steps)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[f"LLM {i+1}" for i in range(len(llm_steps))],
                yticklabels=[f"Bench {i+1}" for i in range(len(bench_steps))])
    plt.title(title, fontsize=14)
    plt.xlabel("LLM Steps")
    plt.ylabel("Benchmark Steps")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_analysis_report(results_file: str, output_dir: str):
    """
    Generates industry-standard figures and statistics from the benchmark results.
    """
    print(f"\n--- Generating Analysis Report in {output_dir} ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    data = []
    with open(results_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if not data:
        print("No data found to report.")
        return

    df = pd.DataFrame(data)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # --- 1. Accuracy by Category (Bar Chart) ---
    plt.figure(figsize=(12, 6))
    subset_acc = df.groupby('subset')['final_correct'].mean().sort_values(ascending=False)
    ax = sns.barplot(x=subset_acc.index, y=subset_acc.values, palette="viridis", hue=subset_acc.index, legend=False)
    plt.title('Accuracy by Math Category', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Category', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    for i, v in enumerate(subset_acc.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_category.png'), dpi=300)
    plt.close()

    # --- 2. Alignment Score Distribution (Histogram/KDE) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='alignment_score', hue='final_correct', kde=True, element="step", palette="coolwarm")
    plt.title('Distribution of Alignment Scores', fontsize=16)
    plt.xlabel('Alignment Score', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alignment_distribution.png'), dpi=300)
    plt.close()

    # --- 3. Quadrant Analysis (Scatter Plot) ---
    plt.figure(figsize=(10, 8))
    
    # Add jitter to y-axis (correctness) for better visualization
    jitter = np.random.normal(0, 0.05, size=len(df))
    y_jittered = df['final_correct'].astype(int) + jitter
    
    sns.scatterplot(x=df['alignment_score'], y=y_jittered, hue=df['final_correct'], 
                    palette="coolwarm", alpha=0.6, s=100)
    
    # Add quadrant lines
    plt.axvline(x=0.7, color='gray', linestyle='--', label='Alignment Threshold')
    plt.axhline(y=0.5, color='gray', linestyle='--')
    
    # Annotate Quadrants
    plt.text(0.95, 0.9, "Robust Reasoning\n(High Align, Correct)", ha='right', va='center', fontsize=12, fontweight='bold')
    plt.text(0.05, 0.9, "Lucky Guess / Shortcut\n(Low Align, Correct)", ha='left', va='center', fontsize=12, fontweight='bold')
    plt.text(0.95, 0.1, "Hallucination / Error\n(High Align, Wrong)", ha='right', va='center', fontsize=12, fontweight='bold')
    plt.text(0.05, 0.1, "Complete Failure\n(Low Align, Wrong)", ha='left', va='center', fontsize=12, fontweight='bold')

    plt.title('Quadrant Analysis: Alignment vs. Correctness', fontsize=16)
    plt.xlabel('Alignment Score', fontsize=14)
    plt.ylabel('Correctness (Jittered)', fontsize=14)
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.legend(title='Is Correct?')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quadrant_analysis.png'), dpi=300)
    plt.close()

    # --- 4. First Divergence Step Distribution ---
    divergence_indices = []
    for entry in data:
        alignment = entry['alignment_details']
        # Find first step where score < 0.7 (arbitrary threshold for "good match")
        divergence_idx = -1
        for idx, step in enumerate(alignment):
            if step['type'] == 'match' and step['score'] < 0.7:
                divergence_idx = idx
                break
            elif step['type'] != 'match': # Missing or Extra step
                divergence_idx = idx
                break
        
        if divergence_idx != -1:
            divergence_indices.append(divergence_idx + 1) # 1-based index

    if divergence_indices:
        plt.figure(figsize=(10, 6))
        sns.histplot(divergence_indices, bins=range(1, max(divergence_indices)+2), kde=False, color="salmon")
        plt.title('Distribution of First Divergence Step', fontsize=16)
        plt.xlabel('Step Number', fontsize=14)
        plt.ylabel('Frequency of Failure Start', fontsize=14)
        plt.xticks(range(1, max(divergence_indices)+1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'divergence_step_distribution.png'), dpi=300)
        plt.close()

    # --- 5. Representative Heatmaps ---
    # Initialize aligner just for heatmap generation (needs embedding model)
    # We try to avoid reloading if possible, but for a report script it's safer to reload or pass it
    # Here we re-instantiate efficiently
    print("Loading embedding model for heatmaps...")
    aligner = StepAligner(EMBEDDING_MODEL)

    # Find Best Case (Correct, Highest Alignment)
    correct_df = df[df['final_correct'] == True]
    if not correct_df.empty:
        best_idx = correct_df['alignment_score'].idxmax()
        best_entry = df.loc[best_idx]
        generate_heatmap(aligner, best_entry['benchmark_steps'], best_entry['llm_steps'], 
                         f"Best Case: {best_entry['subset']} (Score: {best_entry['alignment_score']:.2f})",
                         os.path.join(output_dir, 'heatmap_example_best.png'))

    # Find Worst Case (Incorrect, Lowest Alignment)
    incorrect_df = df[df['final_correct'] == False]
    if not incorrect_df.empty:
        worst_idx = incorrect_df['alignment_score'].idxmin()
        worst_entry = df.loc[worst_idx]
        generate_heatmap(aligner, worst_entry['benchmark_steps'], worst_entry['llm_steps'], 
                         f"Worst Case: {worst_entry['subset']} (Score: {worst_entry['alignment_score']:.2f})",
                         os.path.join(output_dir, 'heatmap_example_worst.png'))
    
    # Find "Hallucination" Case (Incorrect, High Alignment)
    hallucination_df = df[(df['final_correct'] == False) & (df['alignment_score'] > 0.7)]
    if not hallucination_df.empty:
        hal_idx = hallucination_df['alignment_score'].idxmax()
        hal_entry = df.loc[hal_idx]
        generate_heatmap(aligner, hal_entry['benchmark_steps'], hal_entry['llm_steps'], 
                         f"Hallucination: {hal_entry['subset']} (Score: {hal_entry['alignment_score']:.2f})",
                         os.path.join(output_dir, 'heatmap_example_hallucination.png'))


    # --- 6. Summary Statistics Table (CSV) ---
    summary = df.groupby('subset').agg(
        Count=('id', 'count'),
        Accuracy=('final_correct', 'mean'),
        Avg_Alignment=('alignment_score', 'mean'),
        Std_Alignment=('alignment_score', 'std')
    ).round(4)
    
    # Add total row
    total_row = pd.DataFrame({
        'Count': [len(df)],
        'Accuracy': [df['final_correct'].mean()],
        'Avg_Alignment': [df['alignment_score'].mean()],
        'Std_Alignment': [df['alignment_score'].std()]
    }, index=['TOTAL']).round(4)
    
    summary = pd.concat([summary, total_row])
    summary.to_csv(os.path.join(output_dir, 'summary_stats.csv'))
    
    # --- 7. LaTeX Table for Paper ---
    latex_table = summary.to_latex(float_format="%.2f")
    with open(os.path.join(output_dir, 'latex_table.txt'), 'w') as f:
        f.write(latex_table)
        
    # --- 8. Correlation Stats ---
    corr_pearson, _ = stats.pearsonr(df['alignment_score'], df['final_correct'])
    corr_spearman, _ = stats.spearmanr(df['alignment_score'], df['final_correct'])
    
    with open(os.path.join(output_dir, 'correlations.txt'), 'w') as f:
        f.write(f"Pearson Correlation (Alignment vs Correctness): {corr_pearson:.4f}\n")
        f.write(f"Spearman Correlation (Alignment vs Correctness): {corr_spearman:.4f}\n")

    print("Report generation complete.")
    print(f"Figures saved to: {os.path.abspath(output_dir)}")

# --- Main Benchmark Logic ---

def main():
    parser = argparse.ArgumentParser(description="ReAlign Benchmark for DeepSeek Math")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model name for API")
    parser.add_argument("--api", type=str, default=DEFAULT_API_ENDPOINT, help="API Endpoint")
    parser.add_argument("--limit", type=int, default=-1, help="Limit examples per subset")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    parser.add_argument("--subsets", nargs='+', help="Specific subsets to run (default: all)")
    args = parser.parse_args()

    print(f"--- Starting ReAlign Benchmark ---")
    print(f"Model: {args.model}")
    print(f"Dataset: {DATASET_NAME}")
    
    # Initialize Aligner
    aligner = StepAligner()
    
    # List of subsets in Hendrycks Math
    all_subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 
                   'number_theory', 'prealgebra', 'precalculus']
    
    subsets_to_run = args.subsets if args.subsets else all_subsets
    
    # Open results file in append mode (or clear it first if you prefer)
    # We'll clear it for this run
    with open(RESULTS_FILE, 'w') as f:
        pass

    for subset in subsets_to_run:
        print(f"\n=== Benchmarking Subset: {subset} ===")
        try:
            ds = load_dataset(DATASET_NAME, subset, split='test') # Using 'test' split for benchmarking
        except Exception as e:
            print(f"Error loading subset {subset}: {e}")
            continue
            
        examples = list(ds)
        if args.limit > 0:
            examples = examples[:args.limit]
            
        for i, example in enumerate(tqdm(examples, desc=f"Processing {subset}")):
            question = example['problem']
            benchmark_solution = example['solution']
            
            # 1. Generate Solution
            llm_solution = get_llm_reasoning(question, mock=args.mock, 
                                             model_name=args.model, api_endpoint=args.api)
            
            if not llm_solution:
                print(f"Skipping example {i} due to generation failure.")
                continue
                
            # 2. Step Alignment
            bench_steps = parse_reasoning_to_steps(benchmark_solution)
            llm_steps = parse_reasoning_to_steps(llm_solution)
            alignment, align_score = aligner.align_steps(bench_steps, llm_steps)
            
            # 3. Grading (Symbolic Check)
            bench_boxed = extract_boxed_answer(benchmark_solution)
            llm_boxed = extract_boxed_answer(llm_solution)
            
            symbolic_correct = False
            if bench_boxed and llm_boxed:
                # Basic normalization for comparison
                norm_bench = bench_boxed.replace(" ", "")
                norm_llm = llm_boxed.replace(" ", "")
                symbolic_correct = norm_bench == norm_llm
            
            # 4. Grading (LLM Judge)
            # We use the same model as a judge for now, or you could specify a different one
            llm_correct, grade_reasoning = grade_correctness_llm(question, benchmark_solution, 
                                                                 llm_solution, args.model, args.api, 
                                                                 mock=args.mock)
            
            final_correct = symbolic_correct or llm_correct
                
            # Log Result
            result_entry = {
                "subset": subset,
                "id": i,
                "question": question,
                "benchmark_solution": benchmark_solution,
                "llm_solution": llm_solution,
                "benchmark_steps": bench_steps, # Save steps for heatmaps
                "llm_steps": llm_steps,         # Save steps for heatmaps
                "alignment_score": align_score,
                "symbolic_correct": symbolic_correct,
                "llm_correct": llm_correct,
                "final_correct": final_correct,
                "grade_reasoning": grade_reasoning,
                "alignment_details": alignment
            }
            
            # Write to JSONL immediately
            with open(RESULTS_FILE, 'a') as f:
                f.write(json.dumps(result_entry) + "\n")
                
    # --- Generate Report ---
    generate_analysis_report(RESULTS_FILE, REPORT_DIR)

if __name__ == "__main__":
    main()
