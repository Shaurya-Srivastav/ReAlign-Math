# ReAlign: Process-Aware Benchmarking & Efficient Fine-Tuning for Math LLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MLX](https://img.shields.io/badge/Framework-Apple%20MLX-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**ReAlign** is a comprehensive research toolkit designed to advance mathematical reasoning in Large Language Models (LLMs). It addresses two critical challenges:
1.  **Evaluation**: Moving beyond binary "correct/incorrect" grading to evaluate the *quality* and *alignment* of the reasoning process.
2.  **Training**: Democratizing the fine-tuning of reasoning models on consumer hardware (specifically Apple Silicon) using efficient LoRA and QLoRA techniques.

---

## 📂 Repository Structure

This project is organized into three main components:

### 1. [Benchmarking Suite (`Benchmarking/`)](./Benchmarking)
The core of the ReAlign framework. It implements a novel evaluation metric using **Dynamic Programming (Needleman-Wunsch)** and **Semantic Embeddings** to align student reasoning steps with ground-truth solutions.

*   **Key Features**:
    *   **StepAligner**: Algorithms to parse and align reasoning chains.
    *   **Quadrant Analysis**: Visualizes "Robust Reasoning" vs. "Hallucinations" vs. "Lucky Guesses".
    *   **Dual-Mode Grading**: Combines symbolic verification with LLM-based judging.
    *   **Publication-Ready Figures**: Automatically generates heatmaps and statistical reports.

### 2. [LoRA Training (`LORA/`)](./LORA)
A standard Low-Rank Adaptation (LoRA) training pipeline optimized for Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx).

*   **Features**:
    *   Fine-tunes **DeepSeek-Math-7B** on **OpenR1-Math-220k**.
    *   Automatic model conversion from PyTorch to MLX.
    *   Gradient accumulation for large effective batch sizes.

### 3. [QLoRA Training (`QLORA/`)](./QLORA)
**[NEW]** An advanced **Quantized LoRA** pipeline that enables training 7B models on devices with as little as **8GB-16GB RAM**.

*   **Features**:
    *   **4-bit Quantization**: Loads the base model in 4-bit NormalFloat precision.
    *   **Memory Efficiency**: Reduces memory footprint by ~60% compared to standard LoRA.
    *   **Full Pipeline**: Includes training (`qlora.py`) and inference (`inference.py`) scripts.

---

## 🚀 Getting Started

### Prerequisites
-   **Hardware**: Apple Silicon Mac (M1/M2/M3) recommended.
-   **Software**: Python 3.10+, `pip`.

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ReAlign.git
cd ReAlign

# Install core dependencies
pip install mlx mlx-lm datasets sentence-transformers pandas matplotlib seaborn scipy tqdm
```

---

## 📊 Running the Benchmark

To evaluate a model (e.g., DeepSeek-Math-7B) using the ReAlign framework:

```bash
cd Benchmarking
python3 ReAlign-Benchmark.py --model "deepseek-ai/deepseek-math-7b-instruct" --limit 100
```

**Output**: Results will be saved to `realign_benchmark_results.jsonl` and figures (heatmaps, quadrant analysis) will be generated in `figures-and-statistics/`.

---

## 🧠 Training a Model

### Option A: Standard LoRA (Best for M2/M3 Max/Ultra with >32GB RAM)
```bash
cd LORA
python3 lora.py
```

### Option B: QLoRA (Best for M1/M2/M3 Air/Pro with 8GB-16GB RAM)
```bash
cd QLORA
python3 qlora.py
```

After training, you can run inference with your new adapters:
```bash
python3 inference.py --prompt "Solve: integral of x^2" --adapter-path qlora_adapters/adapters.safetensors
```

---

## 📄 Project Report

A detailed academic report describing the methodology, theoretical framework, and experimental results is available in [Project_Report.tex](./Project_Report.tex).

**To compile the report:**
```bash
pdflatex Project_Report.tex
```

---

## 🧩 Methodology Highlights

### The Alignment Score
Unlike standard benchmarks that only check the final answer, ReAlign calculates an **Alignment Score** ($S_{align} \in [0, 1]$).

$$
D[i][j] = \max \begin{cases} 
D[i-1][j-1] + \text{sim}(s_i, b_j) & \text{(Match)} \\
D[i-1][j] + \gamma & \text{(Deletion)} \\
D[i][j-1] + \gamma & \text{(Insertion)}
\end{cases}
$$

This allows us to distinguish between:
*   **Robust Reasoning**: High Alignment + Correct Answer.
*   **Hallucination**: High Alignment + Incorrect Answer (Calculation error).
*   **Lucky Guess**: Low Alignment + Correct Answer (Heuristic/Shortcut).

---

## 📝 Citation

If you use this codebase or the ReAlign framework, please cite:

```bibtex
@article{realign2025,
  title={ReAlign: Process-Aware Benchmarking for Mathematical Reasoning},
  author={Srivastav, Shaurya},
  journal={ECS 289 Project Report},
  year={2025}
}
```
