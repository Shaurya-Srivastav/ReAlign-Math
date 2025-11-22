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
import pickle
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

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
    max_seq_len: int = 1024
    log_every_n_steps: int = 1
    grad_accumulation_steps: int = 8
    max_training_time_hours: float = 16.0

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
    cache_file = Path("dataset_cache.pkl")
    
    # 1. Check Cache
    if cache_file.exists() and not debug:
        logging.info(f"Loading prepared dataset from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                tokenized_chunks = pickle.load(f)
            logging.info(f"Loaded {len(tokenized_chunks)} chunks from cache.")
            return tokenized_chunks
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}. Re-processing dataset.")

    logging.info(f"Loading dataset '{config.dataset_name}' (subset='all', split='train')...")
    dataset = load_dataset(config.dataset_name, name="all", split="train")

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
    
    # 2. Save Cache
    if not debug:
        logging.info(f"Saving processed dataset to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(tokenized_chunks, f)
            
    return tokenized_chunks


def collate_batch(batch, tokenizer):
    """
    Pads a batch of token arrays to the maximum length in the batch.
    Returns:
        inputs: (batch_size, max_len)
        targets: (batch_size, max_len)
        lengths: (batch_size,) - for masking
    """
    # Calculate max length in this batch
    max_len = max(len(item) - 1 for item in batch)
    
    # Pad token is typically 0 or tokenizer.pad_token_id
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    inputs_list = []
    targets_list = []
    lengths_list = []
    
    for item in batch:
        # item is [t1, t2, ..., tn]
        # input: [t1, ..., tn-1]
        # target: [t2, ..., tn]
        
        inp = item[:-1]
        tgt = item[1:]
        length = len(inp)
        lengths_list.append(length)
        
        # Create padded arrays
        pad_len = max_len - length
        
        if pad_len > 0:
            # Pad inputs with pad_token
            inp_padded = mx.concatenate([inp, mx.full((pad_len,), pad_len, dtype=mx.int32)]) # Using pad_len as filler, actual value doesn't matter if masked
            # Pad targets with a special ignore index (e.g., -100), though we will use explicit masking
            tgt_padded = mx.concatenate([tgt, mx.full((pad_len,), pad_token, dtype=mx.int32)])
        else:
            inp_padded = inp
            tgt_padded = tgt
            
        inputs_list.append(inp_padded)
        targets_list.append(tgt_padded)
        
    return mx.stack(inputs_list), mx.stack(targets_list), mx.array(lengths_list)


# ==========================================
# TRAINING
# ==========================================
class LoRATraining: 
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def loss(self, model, inputs, targets, lengths):
        outputs = model(inputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        logits = logits.astype(mx.float32)

        # Create mask based on lengths
        # inputs shape: (B, L)
        B, L = inputs.shape
        indices = mx.arange(L)[None, :] # (1, L)
        mask = indices < lengths[:, None] # (B, L)
        
        # Flatten for cross entropy
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        mask = mask.reshape(-1)

        ce = nn.losses.cross_entropy(logits, targets)
        
        # Apply mask and average over valid tokens only
        ce = ce * mask
        return ce.sum() / mask.sum()

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

        # Optimizer: AdamW is generally better for LLMs
        optimizer = optim.AdamW(learning_rate=self.config.learning_rate)
        
        # Value and grad function now expects (model, inputs, targets, lengths)
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss)

        training_start_time = time.time()
        time_limit_seconds = self.config.max_training_time_hours * 3600
        deadline = training_start_time + time_limit_seconds
        logging.info(f"Training started at {time.ctime(training_start_time)}")
        logging.info(f"Training deadline set for {time.ctime(deadline)} ({self.config.max_training_time_hours} hours)")
        
        stop_training = False
        
        for epoch in range(self.config.num_epochs):
            if stop_training:
                break
                
            epoch_loss = 0.0
            
            # Shuffle dataset at start of epoch
            import random
            random.shuffle(dataset)
            
            num_batches = (len(dataset) + self.config.batch_size - 1) // self.config.batch_size
            
            logging.info("""
====================
Epoch %d / %d
--------------------
batches: %d | batch_size: %d | accum: %d | eff_batch: %d | lr: %.2e
====================""".strip(),
                epoch + 1,
                self.config.num_epochs,
                num_batches,
                self.config.batch_size,
                self.config.grad_accumulation_steps,
                self.config.batch_size * self.config.grad_accumulation_steps,
                self.config.learning_rate,
            )

            # Accumulate gradients state
            accumulated_grads = None
            accumulated_loss = 0.0
            steps_since_update = 0
            
            step_start = time.time()

            with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config.num_epochs} [Deadline: {time.strftime('%H:%M', time.localtime(deadline))}]") as pbar:
                for i, batch_start in enumerate(range(0, len(dataset), self.config.batch_size)):
                    # 1. Prepare Batch
                    raw_batch = dataset[batch_start:batch_start + self.config.batch_size]
                    inputs, targets, lengths = collate_batch(raw_batch, self.tokenizer)
                    
                    # 2. Compute Loss and Gradients
                    (loss_val, grads) = loss_and_grad_fn(self.model, inputs, targets, lengths)
                    
                    # CRITICAL MEMORY FIX: Force evaluation of gradients immediately
                    # This runs the backward pass now and frees activations
                    mx.eval(grads, loss_val)
                    
                    # 3. Accumulate Gradients
                    # Scale gradients by accumulation steps
                    grads = tree_map(lambda x: x / self.config.grad_accumulation_steps, grads)
                    
                    if accumulated_grads is None:
                        accumulated_grads = grads
                    else:
                        # Add current grads to accumulated grads
                        accumulated_grads = tree_map(lambda x, y: x + y, accumulated_grads, grads)
                        # Force evaluation of the accumulator to prevent graph growth
                        mx.eval(accumulated_grads)
                    
                    accumulated_loss += loss_val.item()
                    steps_since_update += 1
                    
                    # 4. Update Weights (if accumulation complete)
                    if steps_since_update >= self.config.grad_accumulation_steps or (i == num_batches - 1):
                        optimizer.update(self.model, accumulated_grads)
                        mx.eval(self.model.parameters(), optimizer.state)
                        
                        # Reset accumulation
                        accumulated_grads = None
                        steps_since_update = 0
                        
                        # Logging
                        tokens_this_step = int(lengths.sum().item())
                        avg_loss = accumulated_loss / self.config.grad_accumulation_steps
                        accumulated_loss = 0.0
                        
                        epoch_loss += avg_loss
                        
                        # Time tracking for progress bar
                        elapsed_h = (time.time() - training_start_time) / 3600
                        remain_h = max(0, self.config.max_training_time_hours - elapsed_h)
                        
                        pbar.set_postfix(loss=f"{avg_loss:.3f}", elapsed=f"{elapsed_h:.2f}h", remain=f"{remain_h:.2f}h")
                        
                        if ((i + 1) % self.config.log_every_n_steps) == 0:
                            logging.info(
                                "Step %d/%d | loss=%.4f | tokens=%d | time=%.2fs | remain=%.2fh",
                                i + 1,
                                num_batches,
                                avg_loss,
                                tokens_this_step,
                                time.time() - step_start,
                                remain_h
                            )
                            step_start = time.time()
                        
                        # Check time limit
                        elapsed_time = time.time() - training_start_time
                        if elapsed_time > time_limit_seconds:
                            logging.info(f"Time limit of {self.config.max_training_time_hours} hours reached. Stopping training.")
                            stop_training = True
                            break

                    pbar.update(1)

            logging.info(f"Epoch {epoch+1} average loss: {epoch_loss / (num_batches / self.config.grad_accumulation_steps):.3f}")

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
