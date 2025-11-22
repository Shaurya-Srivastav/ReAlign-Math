
import os
import json
import requests
import numpy as np
import argparse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from pathlib import Path

# --- MLX Support ---
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.tuner.lora import LoRALinear
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

MLX_MODEL = None
MLX_TOKENIZER = None


# --- Configuration ---
DEFAULT_LLM_API_ENDPOINT = "http://127.0.0.1:1234/v1/completions"
DEFAULT_LLM_MODEL_NAME = "mistral-7b"  # The model name your local server expects
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
DATASET_NAME = 'hkust-nlp/dart-math-hard'
RESULTS_FILE = 'evaluation_results.json'

# --- Processing Controls ---
# --- Processing Controls ---
MAX_EXAMPLES = 5

# --- MLX Helper Functions ---
def apply_lora_layers(model, adapter_path: str):
    """
    Applies LoRA layers to the model based on the adapter configuration.
    """
    adapter_path = Path(adapter_path)
    config_path = adapter_path.parent / "adapter_config.json"
    
    if not config_path.exists():
        print(f"WARNING: Adapter config not found at {config_path}. Using default rank=8, alpha=16.")
        lora_rank = 8
        lora_alpha = 16.0
    else:
        with open(config_path, "r") as f:
            adapter_config = json.load(f)
        lora_rank = adapter_config.get("lora_rank", 8)
        lora_alpha = adapter_config.get("lora_alpha", 16.0)

    lora_scale = lora_alpha / lora_rank
    print(f"Applying LoRA adapters (rank={lora_rank}, alpha={lora_alpha})...")

    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            if hasattr(layer.self_attn, name):
                original_linear = getattr(layer.self_attn, name)
                # Check dimensions
                if hasattr(original_linear, "weight"):
                    out_dim, in_dim = original_linear.weight.shape
                else: # QuantizedLinear
                    in_dim = original_linear.input_dims
                    out_dim = original_linear.output_dims
                    
                lora_layer = LoRALinear(
                    input_dims=in_dim,
                    output_dims=out_dim,
                    r=lora_rank,
                    scale=lora_scale,
                    dropout=0.0,
                )
                lora_layer.linear = original_linear
                setattr(layer.self_attn, name, lora_layer)

def initialize_mlx_model(model_path: str, adapter_path: str = None):
    """
    Initializes the global MLX model and tokenizer.
    """
    global MLX_MODEL, MLX_TOKENIZER
    
    if not MLX_AVAILABLE:
        print("ERROR: MLX not installed. Cannot load local model.")
        return

    print(f"Loading MLX model from {model_path}...")
    model, tokenizer = load(model_path)
    
    if adapter_path:
        print(f"Loading adapters from {adapter_path}...")
        apply_lora_layers(model, adapter_path)
        model.load_weights(str(adapter_path))
    
    MLX_MODEL = model
    MLX_TOKENIZER = tokenizer
    print("MLX Model initialized successfully.")

def get_mlx_reasoning(question: str, max_tokens: int = 1024, temperature: float = 0.6) -> str:
    """
    Generates reasoning using the loaded MLX model.
    """
    if MLX_MODEL is None:
        return "Error: MLX Model not initialized."
        
    prompt = f"You are a helpful math tutor. Solve the problem step by step and give the final answer at the end.\n\n### Problem:\n{question}\n\n### Solution:\n"
    
    response = generate(
        MLX_MODEL, 
        MLX_TOKENIZER, 
        prompt=prompt, 
        max_tokens=max_tokens, 
        verbose=False,
        temp=temperature
    )
    return response


# --- 1. Language Model (LLM) Interfacing ---
def get_llm_reasoning(question: str, mock: bool = False, 
                      model_name: str = DEFAULT_LLM_MODEL_NAME, 
                      api_endpoint: str = DEFAULT_LLM_API_ENDPOINT,
                      temperature: float = 0.2,
                      max_tokens: int = 512) -> str | None:
    """
    Prompts the local LLM to get a step-by-step reasoning for the given question.
    """
    if mock:
        # Return a dummy response for testing
        return f"Step 1: To solve {question}, we first analyze the problem.\nStep 2: Then we apply the formula.\nStep 3: Finally, we get the answer."

    # Check if MLX model is loaded
    if MLX_MODEL is not None:
        return get_mlx_reasoning(question, max_tokens=max_tokens, temperature=temperature)

    # A more structured prompt template


    # A more structured prompt template
    prompt = f"""Below is a math problem. Provide a step-by-step solution.

### Problem
{question}

### Solution
"""
    
    # OpenAI-compatible API payload for a completion endpoint
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # "stop": ["\n\n"] 
    }
    
    headers = {"Content-Type": "application/json"}

    print(f"--- Sending prompt to local LLM at {api_endpoint} ---")
    try:
        response = requests.post(api_endpoint, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx) 
        
        # Extract the generated text
        result = response.json()
        llm_output_string = result['choices'][0]['text'].strip()
        
        print("--- Received LLM Response ---")
        print(llm_output_string)
        return llm_output_string

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERROR: Could not connect to the LLM server. ---")
        print(f"Please ensure your local LLM is running and accessible at: {api_endpoint}")
        print(f"Error details: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"\n--- ERROR: Received an unexpected response format from the LLM server. ---")
        print(f"Response JSON: {response.text}")
        print(f"Error details: {e}")
        return None

# --- 2. Text Processing ---
def parse_reasoning_to_steps(reasoning_string: str) -> list[str]:
    """
    Breaks down a raw reasoning string into a list of individual steps.
    Handles numbered lists and cleans up common artifacts.
    """
    if not reasoning_string:
        return []

    # Split by newlines first
    raw_lines = reasoning_string.split('\n')
    steps = []
    
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
            
        # Filter out common header/footer noise
        if line.lower().startswith("sure") or line.lower().startswith("here is"):
            continue
            
        # Remove numbering (e.g., "1.", "Step 1:", "(a)")
        # Regex matches:
        # ^(?:Step\s+)?\d+[\.:\)]\s* -> Matches "1.", "1:", "1)", "Step 1.", "Step 1:"
        # ^\([a-zA-Z0-9]+\)\s* -> Matches "(a)", "(1)"
        cleaned_line = re.sub(r'^(?:Step\s+)?\d+[\.:\)]\s*', '', line)
        cleaned_line = re.sub(r'^\([a-zA-Z0-9]+\)\s*', '', cleaned_line)
        
        if len(cleaned_line.strip()) > 1:
            steps.append(cleaned_line.strip())
            
    return steps

# --- 3. Alignment Logic ---
class StepAligner:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL, device: str = 'cpu'):
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

    def compute_similarity_matrix(self, steps_a: list[str], steps_b: list[str]) -> np.ndarray:
        """
        Computes the cosine similarity matrix between two lists of steps.
        """
        if not steps_a or not steps_b:
            return np.zeros((len(steps_a), len(steps_b)))
            
        embeddings_a = self.embedding_model.encode(steps_a)
        embeddings_b = self.embedding_model.encode(steps_b)
        
        return cosine_similarity(embeddings_a, embeddings_b)

    def align_steps(self, steps_benchmark: list[str], steps_llm: list[str]):
        """
        Finds the optimal alignment between benchmark steps and LLM steps
        using a dynamic programming approach (Global Alignment).
        """
        n = len(steps_benchmark)
        m = len(steps_llm)
        
        if n == 0 or m == 0:
            return [], 0.0

        sim_matrix = self.compute_similarity_matrix(steps_benchmark, steps_llm)
        
        # DP table for scores
        # dp[i][j] stores the max score to align first i steps of benchmark with first j steps of llm
        dp = np.zeros((n + 1, m + 1))
        
        # Backtracking table to reconstruct path
        # 0: diagonal (match), 1: up (skip benchmark step), 2: left (skip llm step)
        path = np.zeros((n + 1, m + 1), dtype=int)
        
        # Gap penalty (penalty for skipping a step)
        # Small penalty to encourage matching over skipping, but allow skipping if similarity is very low
        GAP_PENALTY = -0.1 
        
        # Initialize first row and column with gap penalties
        for i in range(1, n + 1):
            dp[i][0] = i * GAP_PENALTY
            path[i][0] = 1 # Up
        for j in range(1, m + 1):
            dp[0][j] = j * GAP_PENALTY
            path[0][j] = 2 # Left
            
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score_match = dp[i-1][j-1] + sim_matrix[i-1][j-1]
                score_skip_bench = dp[i-1][j] + GAP_PENALTY
                score_skip_llm = dp[i][j-1] + GAP_PENALTY
                
                best_score = max(score_match, score_skip_bench, score_skip_llm)
                dp[i][j] = best_score
                
                if best_score == score_match:
                    path[i][j] = 0
                elif best_score == score_skip_bench:
                    path[i][j] = 1
                else:
                    path[i][j] = 2
                    
        # Backtrack to find alignment
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and path[i][j] == 0:
                # Match
                alignment.append({
                    'type': 'match',
                    'benchmark_idx': i-1,
                    'llm_idx': j-1,
                    'score': float(sim_matrix[i-1][j-1]),
                    'benchmark_step': steps_benchmark[i-1],
                    'llm_step': steps_llm[j-1]
                })
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or path[i][j] == 1):
                # Skip Benchmark Step (Deletion in LLM)
                alignment.append({
                    'type': 'skip_benchmark',
                    'benchmark_idx': i-1,
                    'llm_idx': None,
                    'score': 0.0,
                    'benchmark_step': steps_benchmark[i-1],
                    'llm_step': None
                })
                i -= 1
            else:
                # Skip LLM Step (Insertion in LLM)
                alignment.append({
                    'type': 'skip_llm',
                    'benchmark_idx': None,
                    'llm_idx': j-1,
                    'score': 0.0,
                    'benchmark_step': None,
                    'llm_step': steps_llm[j-1]
                })
                j -= 1
                
        alignment.reverse()
        
        # Calculate average score of matches only, or total normalized score
        # Here we return the average score of the matched steps to represent quality of alignment
        matches = [a['score'] for a in alignment if a['type'] == 'match']
        avg_score = sum(matches) / len(matches) if matches else 0.0
        
        return alignment, avg_score

# --- 4. Main Evaluation Workflow ---
def main():
    """
    Orchestrates the entire evaluation pipeline.
    """
    parser = argparse.ArgumentParser(description="Robust Local LLM Evaluator")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without connecting to LLM")
    parser.add_argument("--max_examples", type=int, default=MAX_EXAMPLES, help="Number of examples to process")
    args = parser.parse_args()

    print("--- Initializing ReAlign Semantic Evaluation (Robust) ---")
    
    aligner = StepAligner(EMBEDDING_MODEL)
    
    # Load the dataset
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        ds = load_dataset(DATASET_NAME)
        train_ds = ds['train']
    except Exception as e:
        print(f"ERROR: Failed to load dataset. {e}")
        return

    all_results = []
    
    num_to_process = len(train_ds) if args.max_examples == -1 else min(len(train_ds), args.max_examples)
    print(f"Processing {num_to_process} examples...")

    for i in range(num_to_process):
        example = train_ds[i]
        question = example['query']
        benchmark_solution = example['response']
        
        print(f"\n\n--- Processing Example {i+1}/{num_to_process} ---")
        print(f"Question: {question}")

        # Get reasoning from the local LLM
        llm_solution_string = get_llm_reasoning(question, mock=args.mock)
        if llm_solution_string is None:
            print("Aborting evaluation due to LLM connection failure.")
            break # Exit the loop if the LLM server is down

        # Process both solutions into steps
        benchmark_steps = parse_reasoning_to_steps(benchmark_solution)
        llm_steps = parse_reasoning_to_steps(llm_solution_string)

        # Perform Alignment
        alignment, avg_score = aligner.align_steps(benchmark_steps, llm_steps)

        # Compare and store results
        result_entry = {
            'id': i,
            'question': question,
            'benchmark_steps': benchmark_steps,
            'llm_steps': llm_steps,
            'alignment': alignment,
            'average_score': avg_score
        }
        
        print("\n--- Optimal Step Alignment ---")
        for step in alignment:
            if step['type'] == 'match':
                print(f"  [MATCH] Score: {step['score']:.4f}")
                print(f"    BENCH: {step['benchmark_step']}")
                print(f"    LLM  : {step['llm_step']}")
            elif step['type'] == 'skip_benchmark':
                print(f"  [MISSING IN LLM]")
                print(f"    BENCH: {step['benchmark_step']}")
            elif step['type'] == 'skip_llm':
                print(f"  [EXTRA IN LLM]")
                print(f"    LLM  : {step['llm_step']}")

        print(f"\n  Average Alignment Score: {avg_score:.4f}")
        
        all_results.append(result_entry)

    # --- 5. Save Results ---
    print(f"\n--- Saving {len(all_results)} results to {RESULTS_FILE} ---")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
