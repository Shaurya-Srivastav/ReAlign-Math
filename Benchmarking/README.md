# ReAlign Benchmarking Suite

This directory contains a comprehensive benchmarking toolkit for evaluating the reasoning capabilities of Large Language Models (LLMs) on mathematical problems using the **ReAlign** methodology.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Code Architecture](#code-architecture)
  - [local_llm_evaluator.py](#1-local_llm_evaluatorpy)
  - [ReAlign-Benchmark.py](#2-realign-benchmarkpy)
- [Usage Guide](#usage-guide)
- [Output & Reporting](#output--reporting)
- [Methodology](#methodology)

## Overview

The ReAlign Benchmarking Suite implements a novel approach to evaluating LLM reasoning quality by measuring **step-level semantic alignment** between model-generated solutions and benchmark solutions. Rather than only checking final answer correctness, this suite evaluates the entire reasoning chain.

### Key Innovation

Traditional benchmarks only check if the final answer is correct. ReAlign introduces:
- **Step Alignment Score**: Measures how closely the model's reasoning follows the benchmark's logical progression
- **Quadrant Analysis**: Distinguishes between "Robust Reasoning" and "Lucky Guesses"
- **Divergence Detection**: Identifies at which step reasoning typically breaks down

---

## Installation

### Dependencies

Install required Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn sentence-transformers datasets tqdm scipy
```

### LLM Server Setup

You need a local LLM server that provides an OpenAI-compatible API endpoint:

**Option 1: LM Studio**
1. Download [LM Studio](https://lmstudio.ai/)
2. Load `deepseek-ai/deepseek-math-7b-instruct`
3. Start the local server (default: `http://127.0.0.1:1234`)

**Option 2: vLLM**
```bash
vllm serve deepseek-ai/deepseek-math-7b-instruct --port 1234
```

---

## Code Architecture

### 1. `local_llm_evaluator.py`

This is the **core module** containing reusable components for LLM evaluation. It can be imported by other scripts.

#### 1.1 Configuration Constants

```python
DEFAULT_LLM_API_ENDPOINT = "http://127.0.0.1:1234/v1/completions"
DEFAULT_LLM_MODEL_NAME = "mistral-7b"
EMBEDDING_MODEL = 'BAAI/bge-large-en-v1.5'
```

#### 1.2 Key Functions

##### `get_llm_reasoning(question, mock, model_name, api_endpoint, temperature, max_tokens)`
**Purpose**: Sends a math problem to the local LLM and retrieves the step-by-step solution.

**Parameters**:
- `question` (str): The math problem to solve
- `mock` (bool): If True, returns dummy data (for testing)
- `model_name` (str): Model identifier for the API
- `api_endpoint` (str): URL of the LLM server
- `temperature` (float): Sampling temperature (0.0 = deterministic)
- `max_tokens` (int): Maximum response length

**Returns**: The LLM's solution as a string, or `None` if the request fails.

**How it works**:
1. Constructs a prompt with the problem
2. Sends POST request to the LLM API
3. Extracts the generated text from the JSON response
4. Handles errors (connection failures, malformed responses)

##### `parse_reasoning_to_steps(reasoning_string)`
**Purpose**: Breaks down a raw reasoning string into individual logical steps.

**Algorithm**:
1. Splits by newlines
2. Filters out empty lines and common headers ("Sure, here is...")
3. Removes step numbering using regex patterns:
   - `"1."`, `"Step 1:"`, `"(a)"` → Removed
4. Returns a clean list of step strings

**Example**:
```
Input: "1. First, solve for x\n2. Then substitute\n3. Finally, simplify"
Output: ["First, solve for x", "Then substitute", "Finally, simplify"]
```

#### 1.3 The `StepAligner` Class

This is the **heart of the ReAlign methodology**. It uses Dynamic Programming (Global Alignment) to optimally match student steps to benchmark steps.

##### Constructor: `__init__(embedding_model_name, device)`
- Loads a SentenceTransformer model for computing semantic embeddings
- Default: `BAAI/bge-large-en-v1.5` (state-of-the-art English embeddings)

##### Method: `compute_similarity_matrix(steps_a, steps_b)`
**Purpose**: Computes the pairwise cosine similarity between all steps.

**Returns**: A 2D NumPy array where `matrix[i][j]` = similarity between `steps_a[i]` and `steps_b[j]`.

##### Method: `align_steps(steps_benchmark, steps_llm)`
**Purpose**: Finds the optimal alignment path using Dynamic Programming.

**Algorithm** (Needleman-Wunsch / Global Alignment):
1. Create a DP table `dp[i][j]` = best score to align first `i` benchmark steps with first `j` LLM steps
2. Three choices at each cell:
   - **Match**: `dp[i-1][j-1] + similarity[i-1][j-1]`
   - **Delete (Skip Benchmark)**: `dp[i-1][j] + GAP_PENALTY`
   - **Insert (Skip LLM)**: `dp[i][j-1] + GAP_PENALTY`
3. Backtrack from `dp[n][m]` to reconstruct the alignment path
4. Return alignment details and average match score

**Returns**:
- `alignment` (list): Each entry describes a match/skip with type, indices, and score
- `avg_score` (float): Average similarity of matched steps only

**Example Alignment Output**:
```python
[
  {'type': 'match', 'benchmark_idx': 0, 'llm_idx': 0, 'score': 0.95, ...},
  {'type': 'skip_llm', 'llm_idx': 1, ...},  # LLM had an extra step
  {'type': 'match', 'benchmark_idx': 1, 'llm_idx': 2, 'score': 0.88, ...}
]
```

---

### 2. `ReAlign-Benchmark.py`

This is the **main benchmarking script** that orchestrates the entire evaluation pipeline and generates reports.

#### 2.1 Configuration

```python
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"
DEFAULT_API_ENDPOINT = "http://127.0.0.1:1234/v1/completions"
DATASET_NAME = "EleutherAI/hendrycks_math"
RESULTS_FILE = "realign_benchmark_results.jsonl"
REPORT_DIR = "Benchmarking/figures-and-statistics"
```

#### 2.2 Helper Functions

##### `extract_boxed_answer(text)`
**Purpose**: Extracts the final answer from LaTeX `\boxed{...}` notation.

**Algorithm**:
- Uses regex to find `\boxed{content}`
- Handles nested braces with brace-counting logic
- Returns the extracted answer or `None`

##### `grade_correctness_llm(question, benchmark_answer, llm_answer, ...)`
**Purpose**: Uses an LLM as a grader to judge answer correctness.

**Workflow**:
1. Constructs a grading prompt comparing benchmark vs LLM answer
2. Asks the LLM to respond with "CORRECT" or "INCORRECT"
3. Parses the verdict and stores the reasoning

**Why use LLM grading?**
- Math answers can be equivalent but formatted differently (e.g., `1/2` vs `0.5`)
- Symbolic comparison alone may miss valid alternative forms

##### `generate_heatmap(aligner, bench_steps, llm_steps, title, filename)`
**Purpose**: Creates a visualization of the similarity matrix.

**Output**: A heatmap PNG showing which LLM steps align well with which benchmark steps.

#### 2.3 Reporting Module: `generate_analysis_report(results_file, output_dir)`

This function generates all visualizations and statistics after the benchmark completes.

**Visualizations Created**:

1. **Accuracy by Category** (Bar Chart)
   - Groups results by math domain (Algebra, Geometry, etc.)
   - Shows percentage of correct answers per domain

2. **Alignment Distribution** (Histogram + KDE)
   - Shows the spread of alignment scores
   - Colored by correctness (Correct = Blue, Incorrect = Red)

3. **Quadrant Analysis** (Scatter Plot)
   - X-axis: Alignment Score
   - Y-axis: Correctness (with jitter for visibility)
   - Identifies four categories:
     - **Top-Right**: Robust Reasoning (High Align + Correct)
     - **Bottom-Right**: Hallucination (High Align + Wrong)
     - **Top-Left**: Lucky Guess (Low Align + Correct)
     - **Bottom-Left**: Complete Failure (Low Align + Wrong)

4. **Divergence Step Distribution** (Histogram)
   - For each problem, finds the first step where similarity drops below 0.7 or a skip occurs
   - Shows distribution of these "failure points"
   - Insight: Do models fail early (concept) or late (calculation)?

5. **Representative Heatmaps** (3 PNG files)
   - Best Case: Correct answer, highest alignment
   - Worst Case: Incorrect answer, lowest alignment
   - Hallucination Case: Incorrect answer, high alignment (if found)

**Statistics Generated**:

1. **`summary_stats.csv`**: Per-subset breakdown with:
   - Count of examples
   - Mean Accuracy
   - Mean Alignment Score
   - Standard Deviation of Alignment

2. **`latex_table.txt`**: Auto-formatted LaTeX table for papers

3. **`correlations.txt`**: Statistical validation
   - Pearson correlation (linear relationship)
   - Spearman correlation (monotonic relationship)
   - Between Alignment Score and Correctness

#### 2.4 Main Benchmark Workflow: `main()`

**Step-by-Step Execution**:

1. **Parse Arguments** (`argparse`)
   - Model name, API endpoint, limit, mock mode, subsets

2. **Initialize Aligner**
   - Loads the embedding model once (reused for all examples)

3. **For Each Subset** (Algebra, Geometry, etc.):
   - Load dataset from HuggingFace
   - Apply `--limit` if specified

4. **For Each Problem**:
   - **Generate**: Call `get_llm_reasoning()` to get model's solution
   - **Parse**: Extract steps from both benchmark and LLM solutions
   - **Align**: Use `StepAligner.align_steps()` to compute alignment
   - **Grade (Symbolic)**: Extract `\boxed{...}` and compare
   - **Grade (LLM)**: Ask LLM grader for verdict
   - **Combine**: Final correctness = Symbolic OR LLM correct
   - **Log**: Write detailed result to JSONL file

5. **Generate Report**
   - Calls `generate_analysis_report()` to create all figures

---

## Usage Guide

### Basic Benchmark

Run the full benchmark on all math categories:

```bash
python3 ReAlign-Benchmark.py
```

**Expected Runtime**: Several hours (depending on LLM speed and dataset size)

### Quick Test

Test on 5 examples per category:

```bash
python3 ReAlign-Benchmark.py --limit 5
```

### Specific Categories

Only run Algebra and Geometry:

```bash
python3 ReAlign-Benchmark.py --subsets algebra geometry
```

### Mock Mode (Development/Testing)

Test the entire pipeline without a real LLM:

```bash
python3 ReAlign-Benchmark.py --mock --limit 2
```

This uses dummy data to verify the script logic, alignment algorithm, and report generation.

### Custom Model/API

Use a different model or endpoint:

```bash
python3 ReAlign-Benchmark.py --model "meta/llama-3-70b" --api "http://localhost:8080/v1/completions"
```

---

## Output & Reporting

### Directory Structure

After running, you'll find:

```
Benchmarking/
├── realign_benchmark_results.jsonl       # Raw results (JSONL format)
└── figures-and-statistics/
    ├── accuracy_by_category.png          # Bar chart
    ├── alignment_distribution.png        # Histogram with KDE
    ├── quadrant_analysis.png            # Scatter plot
    ├── divergence_step_distribution.png # Histogram
    ├── heatmap_example_best.png         # Similarity matrix
    ├── heatmap_example_worst.png        # Similarity matrix
    ├── heatmap_example_hallucination.png # Similarity matrix (if found)
    ├── summary_stats.csv                # Statistics table
    ├── latex_table.txt                  # LaTeX formatted table
    └── correlations.txt                 # Correlation coefficients
```

### Detailed Output Descriptions

#### 1. Raw Results (`realign_benchmark_results.jsonl`)

**Format**: JSONL (JSON Lines) - Each line is a complete JSON object.

**Contents**: One entry per problem evaluated.

**Example Entry**:
```json
{
  "subset": "algebra",
  "id": 42,
  "question": "Solve for x: 2x + 5 = 13",
  "benchmark_solution": "Step 1: Subtract 5 from both sides...",
  "llm_solution": "First, we subtract 5...",
  "benchmark_steps": ["Subtract 5 from both sides", "Divide by 2", "x = 4"],
  "llm_steps": ["We subtract 5 from both sides", "Then divide by 2", "Therefore x equals 4"],
  "alignment_score": 0.94,
  "symbolic_correct": true,
  "llm_correct": true,
  "final_correct": true,
  "grade_reasoning": "The answer is mathematically equivalent.",
  "alignment_details": [
    {"type": "match", "benchmark_idx": 0, "llm_idx": 0, "score": 0.96, ...},
    {"type": "match", "benchmark_idx": 1, "llm_idx": 1, "score": 0.93, ...},
    {"type": "match", "benchmark_idx": 2, "llm_idx": 2, "score": 0.92, ...}
  ]
}
```

**Use Case**: 
- Error analysis (inspect specific failures)
- Training data for meta-learning
- Debugging the benchmark itself

#### 2. Summary Statistics (`summary_stats.csv`)

**Format**: CSV with columns: `subset`, `Count`, `Accuracy`, `Avg_Alignment`, `Std_Alignment`

**Example Contents**:
```
,Count,Accuracy,Avg_Alignment,Std_Alignment
algebra,1187,0.6234,0.7812,0.1456
geometry,479,0.5823,0.7234,0.1687
number_theory,540,0.4921,0.6845,0.1923
TOTAL,5000,0.5982,0.7456,0.1634
```

**Interpretation**:
- **Count**: Total problems in this category
- **Accuracy**: Percentage of correct answers (0.6234 = 62.34%)
- **Avg_Alignment**: Mean alignment score across all problems
- **Std_Alignment**: Standard deviation (higher = more inconsistent)

**Key Insight**: If `Avg_Alignment` is high but `Accuracy` is low, the model follows logical structure but makes conceptual errors.

#### 3. Correlation Statistics (`correlations.txt`)

**Example Output**:
```
Pearson Correlation (Alignment vs Correctness): 0.6847
Spearman Correlation (Alignment vs Correctness): 0.7012
```

**What This Means**:
- **Pearson (0.6847)**: Strong linear relationship. Higher alignment → more likely correct.
- **Spearman (0.7012)**: Strong monotonic relationship (rank-based).

**Critical for Papers**: These numbers validate that your Alignment Score is a meaningful metric. A correlation > 0.6 is considered strong in social sciences.

**Use Case**: If correlation is LOW (<0.3), it suggests:
- The model is guessing randomly, OR
- Your alignment metric needs refinement

#### 4. LaTeX Table (`latex_table.txt`)

**Example Contents**:
```latex
\begin{tabular}{lrrrr}
\toprule
{} &  Count &  Accuracy &  Avg\_Alignment &  Std\_Alignment \\
\midrule
algebra &   1187 &      0.62 &            0.78 &            0.15 \\
geometry &    479 &      0.58 &            0.72 &            0.17 \\
TOTAL &   5000 &      0.60 &            0.75 &            0.16 \\
\bottomrule
\end{tabular}
```

**Use Case**: Copy-paste directly into your LaTeX paper. No manual formatting needed!

#### 5. Visualizations (PNG Files)

##### `accuracy_by_category.png`
**What It Shows**: Bar chart with accuracy percentage for each math domain.

**Example Insight**: "The model achieves 72% on Algebra but only 48% on Number Theory, suggesting weakness in modular arithmetic and prime factorization."

**Why It Matters**: Identifies strengths and weaknesses across domains for targeted improvement.

##### `alignment_distribution.png`
**What It Shows**: Histogram of alignment scores, split by correctness (blue=correct, red=incorrect).

**Example Observation**: 
- Correct answers cluster around 0.8-0.9 alignment
- Incorrect answers have bimodal distribution (some at 0.3, some at 0.7)

**Interpretation**: The 0.7 incorrect group = "close but not quite" (calculation errors). The 0.3 group = fundamental misunderstanding.

##### `quadrant_analysis.png`
**What It Shows**: Scatter plot dividing results into 4 quadrants.

**The Four Quadrants**:
1. **Top-Right (High Align, Correct)**: 65% of total
   - **Meaning**: Robust reasoning. The model "gets it."
   
2. **Top-Left (Low Align, Correct)**: 8% of total
   - **Meaning**: Lucky guess or undocumented shortcut. Correct answer via unexpected method.
   - **Example**: Solving via graphing when algebra was expected.
   
3. **Bottom-Right (High Align, Incorrect)**: 12% of total
   - **Meaning**: HALLUCINATION. Model is confident but wrong.
   - **Example**: Applied correct method to wrong equation.
   
4. **Bottom-Left (Low Align, Incorrect)**: 15% of total
   - **Meaning**: Complete failure. Wrong method AND wrong answer.

**Critical for Papers**: This quadrant breakdown is your main result. "Only 12% hallucinations suggests high reliability."

##### `divergence_step_distribution.png`
**What It Shows**: Histogram of "at which step number did reasoning diverge?"

**Example Finding**: 
```
Step 1: 5% of failures
Step 2: 8% of failures  
Step 3: 22% of failures ← Peak
Step 4: 15% of failures
Step 5+: 10% of failures
```

**Interpretation**: Most errors happen at Step 3, suggesting a common conceptual stumbling block (e.g., applying the quadratic formula correctly).

**Use Case**: Curriculum design for AI training. Focus on Step 3 complexity.

##### `heatmap_example_best.png` / `worst.png` / `hallucination.png`
**What They Show**: Color-coded similarity matrices.

**Example Heatmap Interpretation**:
```
               LLM1   LLM2   LLM3   LLM4
Benchmark1    [0.95] [0.42] [0.38] [0.21]
Benchmark2    [0.41] [0.93] [0.44] [0.35]
Benchmark3    [0.33] [0.47] [0.91] [0.38]
Benchmark4    [0.22] [0.38] [0.42] [0.88]
```

**Reading**: The diagonal (0.95, 0.93, 0.91, 0.88) shows perfect 1-to-1 alignment. This is a "best case."

**Worst Case Heatmap**:
```
               LLM1   LLM2   LLM3
Benchmark1    [0.65] [0.34] [0.29]
Benchmark2    [0.41] [0.58] [0.33]
Benchmark3    [0.38] [0.44] [0.62]
```

**Reading**: No clear diagonal. Model steps don't map to benchmark steps. This is incoherent reasoning.

**Use in Papers**: Include these as Figure 3 and Figure 4 to visually demonstrate your methodology.

### Interpreting Combined Results

**Scenario 1: High Accuracy (78%), High Alignment (0.85)**
→ **Strong Model**. Ready for deployment.

**Scenario 2: High Accuracy (75%), Low Alignment (0.52)**
→ **Overfitting/Memorization**. Model might fail on out-of-distribution problems.

**Scenario 3: Low Accuracy (45%), High Alignment (0.81)**
→ **Conceptual Errors**. Model understands the process but lacks domain knowledge. Needs more training data.

**Scenario 4: Low Accuracy (42%), Low Alignment (0.49)**
→ **Fundamental Failure**. Model doesn't understand the task. Complete retraining needed.

---

## Methodology

### Why Step-Level Alignment?

Traditional LLM benchmarks only check final answers. This has limitations:

1. **No Process Credit**: A model with correct reasoning but a typo gets 0 points
2. **No Penalty for Hallucination**: A lucky guess gets full points
3. **No Insight into Failure Modes**: Can't diagnose where reasoning breaks down

### The ReAlign Approach

**Core Idea**: Evaluate the reasoning *process*, not just the final answer.

**How**:
1. Parse both the model's solution and the benchmark into discrete steps
2. Compute semantic similarity for each step using embeddings
3. Use Dynamic Programming to find the optimal alignment
4. Compute an Alignment Score (0-1) based on average matched step similarity

**Benefits**:
- Rewards models that "think correctly" even if they make minor errors
- Identifies hallucinations (high confidence in wrong reasoning)
- Provides granular feedback for model improvement

### Dynamic Programming Alignment

The algorithm is inspired by sequence alignment in bioinformatics (Needleman-Wunsch):

- **Match Reward**: If two steps are similar (high cosine similarity), we align them
- **Gap Penalty**: Skipping a step (insertion/deletion) incurs a small penalty
- **Goal**: Maximize total similarity while minimizing gaps

This produces an alignment that best explains how the LLM's reasoning relates to the benchmark's reasoning.

---

## For Research Papers

This suite is designed for academic use. The generated figures and LaTeX tables are publication-ready. Key metrics to report:

1. **Overall Accuracy**: From `summary_stats.csv`
2. **Alignment Score Distribution**: Cite mean and std dev
3. **Correlation Coefficient**: From `correlations.txt` (validates that alignment predicts correctness)
4. **Quadrant Percentages**: E.g., "12% of correct answers had low alignment, suggesting memorization"

Include the Quadrant Analysis and Heatmaps as figures in your paper to illustrate the methodology.
