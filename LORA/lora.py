# -*- coding: utf-8 -*-
"""
MLX LoRA Fine-Tuning Script for Apple Silicon

This script fine-tunes the DeepSeek-Math-7B model on the OpenR1-Math-220k dataset
using LoRA (Low-Rank Adaptation) with the MLX framework.

Workflow:
1.  Sets up logging and configuration.
2.  Checks if the model has been converted to MLX format. If not, it automatically
    downloads the PyTorch model from Hugging Face and converts it.
3.  Loads the dataset and formats it into a prompt-response structure.
4.  Loads the quantized model and attaches LoRA adapters.
5.  Runs the training loop with detailed progress bars (tqdm).
6.  Saves the trained LoRA adapters.
"""

import argparse
import logging
import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from huggingface_hub import hf_hub_download, snapshot_download
import torch

from mlx_lm.utils import load
from mlx_lm.models.deepseek import Model as DeepSeekModel, ModelArgs as DeepSeekModelArgs
from mlx_lm.tuner.lora import LoRALinear

from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
class TrainingConfig:
    # Model and Dataset
    hf_model_name: str = "deepseek-ai/deepseek-math-7b-instruct"
    mlx_model_path: str = "mlx_model"  # Local path for converted model
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    adapter_path: str = "adapters"

    # Training Hyperparameters
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-5
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: int = 16 # Number of layers to apply LoRA to
    max_seq_len: int = 2048
    log_every_n_steps: int = 1

    # Prompt Template
    PROMPT_TEMPLATE: str = """You are a helpful math tutor. Solve the problem step by step and give the final answer at the end.

### Problem:
{problem}

### Solution:
{solution} """

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# MODEL CONVERSION (FROM .BIN)
# ==========================================
def fetch_hf_files(repo_id, hf_path):
    """Downloads necessary model files from Hugging Face."""
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    weight_files = []
    if "model.safetensors.index.json" in [f.name for f in hf_path.iterdir()]:
        # This case is for safetensors, which we aren't using but good to have
        pass # Safetensors logic would go here
    else:
        # Handle .bin files
        pytorch_bins = [f for f in hf_path.glob("*.bin")]
        if not pytorch_bins:
            raise FileNotFoundError(f"No .bin files found in {hf_path}")
        for bin_file in pytorch_bins:
            weight_files.append(str(bin_file.name))

    return config, weight_files

def convert_from_bin_to_mlx(hf_repo: str, mlx_path: str):
    """Converts a HF model with .bin weights to MLX format."""
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    hf_path = Path(snapshot_download(repo_id=hf_repo))
    config, weight_files = fetch_hf_files(hf_repo, hf_path)

    # Load weights from all .bin files
    weights = {}
    for wf in tqdm(weight_files, desc="Loading .bin weights"):
        state_dict = torch.load(hf_path / wf, map_location="cpu")
        weights.update(state_dict)

    # Convert to MLX Model
    model_args = DeepSeekModelArgs.from_dict(config)
    model = DeepSeekModel(model_args)
    model.load_weights([(k, mx.array(v.to(torch.float32).numpy())) for k, v in weights.items()])

    # Save MLX model
    logging.info("Saving converted MLX model to disk (this may take a minute)...")
    model.save_weights(str(mlx_path / "model.safetensors"))
    with open(mlx_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Copy tokenizer files
    logging.info("Copying tokenizer files...")
    import shutil
    for file_path in hf_path.glob("*"):
        if file_path.name.startswith("tokenizer") or file_path.suffix == ".model":
            shutil.copy(file_path, mlx_path / file_path.name)
    
    logging.info("Model saved.")


def convert_model_if_needed(config: TrainingConfig):
    """Checks for an MLX model and converts it from Hugging Face if not found."""
    mlx_path = Path(config.mlx_model_path)
    # Check if the model has been converted and is valid
    weights_path = mlx_path / "model.safetensors"
    if mlx_path.exists() and weights_path.exists() and weights_path.stat().st_size > 1_000_000_000: # 1GB
        logging.info(f"Found valid converted MLX model at: {mlx_path}")
        return

    logging.warning(f"MLX model not found at '{mlx_path}'. Starting conversion from Hugging Face.")
    logging.info("This is a one-time process and may take a while...")

    try:
        convert_from_bin_to_mlx(config.hf_model_name, config.mlx_model_path)
        logging.info(f"Successfully converted model and saved to {mlx_path}")
    except Exception as e:
        logging.error(f"An error occurred during model conversion: {e}")
        raise

# ==========================================
# DATA HANDLING
# ==========================================
def load_and_prepare_data(config: TrainingConfig, tokenizer, debug: bool = False):
    """Load dataset, format prompts, and tokenize within length limits."""
    logging.info(f"Loading dataset '{config.dataset_name}'...")
    dataset = load_dataset(config.dataset_name, split="train")

    if debug:
        logging.warning("Debug mode active: using only 10 samples.")
        dataset = dataset.select(range(10))

    def format_prompt(example):
        problem = str(example.get("problem", ""))
        solution = str(example.get("solution", ""))
        return config.PROMPT_TEMPLATE.format(problem=problem, solution=solution)

    tokenized_chunks = []
    truncated_examples = 0
    total_prompts = len(dataset)

    logging.info("Formatting prompts and splitting into max_len chunks...")
    for example in tqdm(dataset, desc="Formatting prompts"):
        prompt = format_prompt(example)
        tokens = tokenizer.encode(prompt)
        prompt_len = len(tokens)

        if prompt_len > config.max_seq_len:
            truncated_examples += 1

        start = 0
        while start < prompt_len:
            slice_tokens = tokens[start:start + config.max_seq_len]
            if len(slice_tokens) <= 1:
                break
            tokenized_chunks.append(mx.array(slice_tokens, dtype=mx.int32))
            start += config.max_seq_len

    logging.info(
        "Created %d token chunks (max_len=%d). %d/%d prompts exceeded the limit and were split.",
        len(tokenized_chunks),
        config.max_seq_len,
        truncated_examples,
        total_prompts,
    )
    return tokenized_chunks

# ==========================================
# TRAINING
# ==========================================
class LoRATraining: 
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def loss(self, model, inputs, targets):
        outputs = model(inputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        logits = logits.astype(mx.float32)

        # Flatten logits/targets so cross_entropy sees (N, vocab) vs (N,)
        if logits.ndim == 3:
            bsz, seq_len, vocab = logits.shape
            logits = logits.reshape(bsz * seq_len, vocab)
            targets = targets.reshape(bsz * seq_len)
        elif logits.ndim == 2 and targets.ndim > 1:
            # logits already (seq, vocab); ensure targets flattened
            targets = targets.reshape(-1)

        ce = nn.losses.cross_entropy(logits, targets)
        return mx.mean(ce)

    def _apply_lora(self):
        """Manually apply LoRA to the model layers."""
        for layer in self.model.model.layers:
            # Target Attention Projections (Query and Value are standard for LoRA)
            # You can also add "k_proj", "o_proj", "gate_proj", etc. if you want more trainable params.
            target_modules = ["q_proj", "v_proj"]
            
            for name in target_modules:
                if hasattr(layer.self_attn, name):
                    original_linear = getattr(layer.self_attn, name)
                    # Create LoRA layer directly
                    # Check if weight is transposed (common in MLX linear layers)
                    # Usually MLX Linear weights are (out, in)
                    out_dim, in_dim = original_linear.weight.shape
                    
                    # Calculate scale based on alpha and rank (standard LoRA formula)
                    # scale = alpha / r
                    lora_scale = self.config.lora_alpha / self.config.lora_rank
                    
                    lora_layer = LoRALinear(
                        input_dims=in_dim,
                        output_dims=out_dim,
                        r=self.config.lora_rank,
                        scale=lora_scale,
                        dropout=0.05
                    )
                    # Copy original weights
                    lora_layer.linear = original_linear
                    setattr(layer.self_attn, name, lora_layer)

    def train(self, dataset):
        self.model.train()
        
        # Freeze all model parameters
        self.model.freeze()
        
        # Apply LoRA adapters
        self._apply_lora()

        p = sum(v.size for _, v in tree_flatten(self.model.trainable_parameters()))
        logging.info(f"Starting LoRA training with {p / 1e6:.3f}M trainable parameters.")

        # Optimizer
        optimizer = optim.Adam(learning_rate=self.config.learning_rate)
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss)

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size

            logging.info("""
====================
Epoch %d / %d
--------------------
batches: %d | batch_size: %d | lr: %.2e
====================""".strip(),
                epoch + 1,
                self.config.num_epochs,
                num_batches,
                self.config.batch_size,
                self.config.learning_rate,
            )

            with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                for i, batch_start in enumerate(range(0, len(dataset), self.config.batch_size)):
                    batch = dataset[batch_start:batch_start + self.config.batch_size]
                    step_start = time.time()
                    
                    # Prepare inputs and targets
                    inputs = mx.stack([b[:-1] for b in batch])
                    targets = mx.stack([b[1:] for b in batch])
                    tokens_this_step = sum(int(arr.size) for arr in batch)

                    # Compute loss and gradients
                    (loss_val, grads) = loss_and_grad_fn(self.model, inputs, targets)
                    optimizer.update(self.model, grads)
                    mx.eval(self.model.parameters(), optimizer.state)
                    
                    epoch_loss += loss_val.item()
                    pbar.set_postfix(loss=f"{epoch_loss / (i+1):.3f}")
                    pbar.update(1)

                    if ((i + 1) % self.config.log_every_n_steps) == 0:
                        logging.info(
                            "Epoch %d Step %d/%d | loss=%.4f | tokens=%d | step_time=%.2fs | running_loss=%.4f",
                            epoch + 1,
                            i + 1,
                            num_batches,
                            loss_val.item(),
                            tokens_this_step,
                            time.time() - step_start,
                            epoch_loss / (i + 1),
                        )

            logging.info(f"Epoch {epoch+1} average loss: {epoch_loss / num_batches:.3f}")

        self.save_adapters()

    def save_adapters(self):
        """Save the trained LoRA adapters."""
        adapter_path = Path(self.config.adapter_path)
        adapter_path.mkdir(exist_ok=True)
        self.model.save_weights(str(adapter_path / "adapters.safetensors"))
        
        # Save LoRA config
        with open(adapter_path / "adapter_config.json", "w") as f:
            json.dump({
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
            }, f, indent=4)
        logging.info(f"Saved trained LoRA adapters to '{adapter_path}'.")

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA training script for MLX.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with a small sample of data.")
    args = parser.parse_args()

    config = TrainingConfig()
    
    # Step 1: Convert model if it doesn't exist
    logging.info("=== Step 1: Verifying MLX Model ===")
    step_time = time.time()
    convert_model_if_needed(config)
    logging.info(f"Model verification complete. Took {time.time() - step_time:.2f}s.")
    
    # Step 2: Load model and tokenizer
    logging.info("=== Step 2: Loading Model and Tokenizer ===")
    step_time = time.time()
    model, tokenizer = load(config.mlx_model_path)
    logging.info(f"Model loading complete. Took {time.time() - step_time:.2f}s.")

    # Step 3: Load and process data
    logging.info("=== Step 3: Preparing Dataset ===")
    step_time = time.time()
    dataset = load_and_prepare_data(config, tokenizer, debug=args.debug)
    logging.info(f"Dataset preparation complete. Took {time.time() - step_time:.2f}s.")
    
    # Step 4: Train
    logging.info("=== Step 4: Starting LoRA Training ===")
    step_time = time.time()
    trainer = LoRATraining(model, tokenizer, config)
    trainer.train(dataset)
    logging.info(f"Training complete. Took {time.time() - step_time:.2f}s.")
    
    logging.info("=== All Steps Finished ===")
