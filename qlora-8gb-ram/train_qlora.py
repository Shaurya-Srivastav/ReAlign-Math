# -*- coding: utf-8 -*-
"""
QLoRA Fine-Tuning Script for DeepSeek-Math-7B

This script fine-tunes the DeepSeek-Math-7B model on the OpenR1-Math-220k dataset
using QLoRA (Quantized LoRA) with 4-bit precision.

Workflow:
1.  Sets up logging and configuration.
2.  Loads the OpenR1-Math-220k dataset and formats it into prompts.
3.  Loads the base model with 4-bit quantization using BitsAndBytes.
4.  Attaches LoRA adapters to specific layers.
5.  Runs training using HuggingFace Trainer with gradient accumulation.
6.  Saves the trained LoRA adapters and provides inference utilities.
"""

import argparse
import logging
import time
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ==========================================
# CONFIGURATION
# ==========================================
class TrainingConfig:
    """Centralized configuration for QLoRA training"""
    
    # Model and Dataset
    model_name: str = "deepseek-ai/deepseek-math-7b-instruct"
    dataset_name: str = "open-r1/OpenR1-Math-220k"
    output_dir: str = "deepseek-math-qlora-adapters"
    
    # QLoRA Configuration
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype = torch.bfloat16
    
    # LoRA Hyperparameters
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will be set in __post_init__
    
    # Training Hyperparameters
    num_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_length: int = 2048
    
    # Memory optimization flags
    use_8gb_mode: bool = False  # Set to True for 8GB GPUs
    
    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 2
    
    # Optimizer
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    
    # Prompt Template
    PROMPT_TEMPLATE: str = """You are a helpful math tutor. Solve the problem step by step and give the final answer at the end.

Problem:
{question}

Write your full solution and final answer clearly.
"""
    
    def __post_init__(self):
        """Set default target modules if not specified"""
        if self.lora_target_modules is None:
            if self.use_8gb_mode:
                # Fewer target modules for 8GB GPUs
                self.lora_target_modules = ["q_proj", "v_proj"]
                logger.info("8GB mode: Using minimal target modules (q_proj, v_proj)")
            else:
                self.lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
        
        # Apply 8GB optimizations
        if self.use_8gb_mode:
            logger.info("=" * 70)
            logger.info("8GB MEMORY MODE ACTIVATED")
            logger.info("Applying aggressive memory optimizations...")
            logger.info("=" * 70)
            self.max_seq_length = 512  # Reduce from 2048
            self.gradient_accumulation_steps = 4  # Reduce from 16
            self.lora_rank = 32  # Reduce from 64
            logger.info(f"  - Max sequence length: {self.max_seq_length} (was 2048)")
            logger.info(f"  - Gradient accumulation: {self.gradient_accumulation_steps} (was 16)")
            logger.info(f"  - LoRA rank: {self.lora_rank} (was 64)")
            logger.info(f"  - Target modules: {len(self.lora_target_modules)} (was 7)")
            logger.info("=" * 70)

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# DATA HANDLING
# ==========================================
def _first_nonempty(example, keys):
    """Return first non-empty value from example for any key in keys."""
    for k in keys:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (list, tuple)):
            joined = " ".join([str(x).strip() for x in v if str(x).strip()])
            if joined:
                return joined
    return ""

def load_and_prepare_data(config: TrainingConfig, debug: bool = False):
    """Load dataset and format prompts."""
    logger.info(f"Loading dataset '{config.dataset_name}'...")
    
    # Load dataset
    raw_ds = load_dataset(config.dataset_name, split="train")
    
    if debug:
        logger.warning("Debug mode active: using only 100 samples.")
        raw_ds = raw_ds.select(range(100))
    
    logger.info(f"Dataset loaded with {len(raw_ds)} examples.")
    logger.info(f"Dataset columns: {raw_ds.column_names}")
    
    # Remove old 'text' column if it exists
    if "text" in raw_ds.column_names:
        raw_ds = raw_ds.remove_columns("text")
    
    # Define candidate keys for question and answer
    Q_KEYS = ["query", "question", "problem", "Problem", "input", "prompt"]
    A_KEYS = ["response", "solution", "answer", "target", "output"]
    
    def build_text_batch(examples):
        """Batched map function to create 'text' field."""
        first_col = list(examples.keys())[0]
        n = len(examples[first_col])
        texts = []
        has_q = []
        has_a = []
        
        for i in range(n):
            ex = {k: examples.get(k, [""] * n)[i] for k in examples.keys()}
            q = _first_nonempty(ex, Q_KEYS)
            a = _first_nonempty(ex, A_KEYS)
            
            prompt = config.PROMPT_TEMPLATE.format(question=q)
            full_text = f"{prompt}\n\nSolution:\n{a}"
            
            texts.append(full_text)
            has_q.append(bool(q))
            has_a.append(bool(a))
        
        return {"text": texts, "has_question": has_q, "has_answer": has_a}
    
    # Apply formatting
    logger.info("Formatting prompts...")
    train_ds = raw_ds.map(build_text_batch, batched=True, batch_size=512)
    
    # Diagnostics
    total = len(train_ds)
    num_has_q = sum(train_ds["has_question"])
    num_has_a = sum(train_ds["has_answer"])
    logger.info(f"Out of {total} examples: has_question={num_has_q}, has_answer={num_has_a}")
    
    # Preview first example
    logger.info(f"First example preview:\n{train_ds[0]['text'][:500]}...")
    
    return train_ds

def tokenize_dataset(dataset, tokenizer, config: TrainingConfig):
    """Tokenize the dataset."""
    logger.info("Tokenizing dataset...")
    
    def tokenize_fn(example):
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding=False,
        )
        # Copy input_ids to labels for causal LM
        out["labels"] = out["input_ids"].copy()
        return out
    
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"Tokenization complete. Dataset size: {len(tokenized)}")
    return tokenized

# ==========================================
# MODEL SETUP
# ==========================================
def load_model_and_tokenizer(config: TrainingConfig):
    """Load base model with 4-bit quantization and tokenizer."""
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Cleared GPU cache")
    
    logger.info(f"Loading tokenizer from '{config.model_name}'...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set tokenizer.pad_token to: {tokenizer.pad_token}")
    
    # Configure 4-bit quantization
    logger.info("Configuring 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
    )
    
    # Load model
    logger.info(f"Loading model '{config.model_name}' with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("Model loaded successfully.")
    return model, tokenizer

def attach_lora_adapters(model, config: TrainingConfig):
    """Attach LoRA adapters to the model."""
    logger.info("Configuring LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    logger.info("=" * 50)
    model.print_trainable_parameters()
    logger.info("=" * 50)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    return model

# ==========================================
# TRAINING
# ==========================================
def train_model(model, tokenizer, train_dataset, config: TrainingConfig):
    """Train the model using HuggingFace Trainer."""
    logger.info("Setting up training arguments...")
    
    # Determine precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=0.3,  # Gradient clipping
        ddp_find_unused_parameters=False,
    )
    
    logger.info(f"Training Configuration:")
    logger.info(f"  - Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  - Total epochs: {config.num_epochs}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Enable input gradients (required for LoRA)
    model.enable_input_require_grads()
    model.train()
    
    # Start training
    logger.info("=" * 50)
    logger.info("Starting training...")
    logger.info("=" * 50)
    training_start = time.time()
    
    trainer.train()
    
    training_time = time.time() - training_start
    logger.info("=" * 50)
    logger.info(f"Training complete! Time: {training_time / 3600:.2f} hours")
    logger.info("=" * 50)
    
    return trainer

# ==========================================
# SAVING
# ==========================================
def save_model(trainer, tokenizer, config: TrainingConfig):
    """Save the trained LoRA adapters and tokenizer."""
    logger.info(f"Saving LoRA adapters to '{config.output_dir}'...")
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"LoRA adapters and tokenizer saved to '{config.output_dir}'")
    
    # Save config summary
    config_summary = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
    }
    
    import json
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    
    logger.info("Training configuration saved.")

# ==========================================
# INFERENCE UTILITIES
# ==========================================
def create_inference_script(config: TrainingConfig):
    """Create a separate inference script for easy testing."""
    inference_script = """#!/usr/bin/env python3
\"\"\"
Inference script for trained QLoRA model.
Usage: python inference.py "Your math question here"
\"\"\"

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(adapter_path, base_model_name):
    \"\"\"Load the base model and merge LoRA adapters.\"\"\"
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_solution(model, tokenizer, question, max_new_tokens=512):
    \"\"\"Generate a solution for the given question.\"\"\"
    prompt_template = \"\"\"You are a helpful math tutor. Solve the problem step by step and give the final answer at the end.

Problem:
{question}

Write your full solution and final answer clearly.

Solution:
\"\"\"
    
    prompt = prompt_template.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    ADAPTER_PATH = "OUTPUT_DIR_PLACEHOLDER"
    BASE_MODEL = "MODEL_NAME_PLACEHOLDER"
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py 'Your math question here'")
        sys.exit(1)
    
    question = sys.argv[1]
    
    print("Loading model...")
    model, tokenizer = load_model(ADAPTER_PATH, BASE_MODEL)
    
    print(f"\\nQuestion: {question}\\n")
    print("Generating solution...\\n")
    
    solution = generate_solution(model, tokenizer, question)
    print(solution)
"""
    
    # Replace placeholders
    inference_script = inference_script.replace("OUTPUT_DIR_PLACEHOLDER", config.output_dir)
    inference_script = inference_script.replace("MODEL_NAME_PLACEHOLDER", config.model_name)
    
    # Save script
    script_path = Path(config.output_dir) / "inference.py"
    with open(script_path, "w") as f:
        f.write(inference_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Inference script created at: {script_path}")

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="QLoRA training script for DeepSeek-Math-7B.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with 100 samples.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--8gb", dest="use_8gb", action="store_true", 
                       help="Enable 8GB GPU mode (aggressive memory optimization)")
    args = parser.parse_args()
    
    # Initialize config
    config = TrainingConfig()
    
    # Apply 8GB mode if requested
    if args.use_8gb:
        config.use_8gb_mode = True
    
    config.__post_init__()
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    logger.info("=" * 70)
    logger.info("QLoRA Training Script for DeepSeek-Math-7B")
    logger.info("=" * 70)
    
    # Step 1: Load and prepare data
    logger.info("\n=== Step 1: Loading and Preparing Dataset ===")
    step_time = time.time()
    train_dataset = load_and_prepare_data(config, debug=args.debug)
    logger.info(f"Step 1 complete. Time: {time.time() - step_time:.2f}s\n")
    
    # Step 2: Load model and tokenizer
    logger.info("=== Step 2: Loading Model and Tokenizer ===")
    step_time = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info(f"Step 2 complete. Time: {time.time() - step_time:.2f}s\n")
    
    # Step 3: Tokenize dataset
    logger.info("=== Step 3: Tokenizing Dataset ===")
    step_time = time.time()
    tokenized_dataset = tokenize_dataset(train_dataset, tokenizer, config)
    logger.info(f"Step 3 complete. Time: {time.time() - step_time:.2f}s\n")
    
    # Step 4: Attach LoRA adapters
    logger.info("=== Step 4: Attaching LoRA Adapters ===")
    step_time = time.time()
    model = attach_lora_adapters(model, config)
    logger.info(f"Step 4 complete. Time: {time.time() - step_time:.2f}s\n")
    
    # Step 5: Train
    logger.info("=== Step 5: Training ===")
    step_time = time.time()
    trainer = train_model(model, tokenizer, tokenized_dataset, config)
    logger.info(f"Step 5 complete. Time: {time.time() - step_time:.2f}s\n")
    
    # Step 6: Save
    logger.info("=== Step 6: Saving Model ===")
    step_time = time.time()
    save_model(trainer, tokenizer, config)
    create_inference_script(config)
    logger.info(f"Step 6 complete. Time: {time.time() - step_time:.2f}s\n")
    
    logger.info("=" * 70)
    logger.info("All steps finished successfully!")
    logger.info(f"LoRA adapters saved to: {config.output_dir}")
    logger.info(f"To test: python {config.output_dir}/inference.py 'Your question'")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()