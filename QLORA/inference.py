#!/usr/bin/env python3
"""
Lightweight Inference Script for 8GB GPUs

This version is optimized for low memory usage.
Usage: python inference_8gb.py "Your math question here"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL = "deepseek-ai/deepseek-math-7b-instruct"
ADAPTER_PATH = "deepseek-math-qlora-adapters"

PROMPT_TEMPLATE = """You are a helpful math tutor. Solve the problem step by step and give the final answer at the end.

Problem:
{question}

Write your full solution and final answer clearly.

Solution:
"""

def load_model():
    """Load model with minimal memory footprint."""
    print("=" * 70)
    print("Loading Model (8GB GPU Mode - This may take a minute)...")
    print("=" * 70)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    print("2. Configuring 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print("3. Loading base model (this takes the longest)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapters
    print("4. Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    
    print("\n✓ Model loaded successfully!")
    print("=" * 70)
    
    return model, tokenizer

def generate_solution(model, tokenizer, question, max_tokens=512):
    """Generate solution with minimal memory usage."""
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(question=question)
    
    # Tokenize (truncate if needed)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
    
    # Generate
    print("\n🤔 Generating solution...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract solution
    if "Solution:" in full_text:
        solution = full_text.split("Solution:")[-1].strip()
        return solution
    return full_text

def main():
    print("\n" + "=" * 70)
    print(" Math Tutor - 8GB GPU Edition ".center(70))
    print("=" * 70 + "\n")
    
    # Get question
    if len(sys.argv) < 2:
        print("Usage: python inference_8gb.py \"Your math question here\"\n")
        print("Example:")
        question = "Solve for x: 2x + 5 = 13"
        print(f"  python inference_8gb.py \"{question}\"\n")
        print("Running example question...")
    else:
        question = " ".join(sys.argv[1:])
    
    print("\n📝 Question:")
    print("-" * 70)
    print(question)
    print("-" * 70)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your adapter path is correct")
        print("2. Clear GPU memory: python3 -c 'import torch; torch.cuda.empty_cache()'")
        print("3. Close other programs using GPU")
        sys.exit(1)
    
    # Generate solution
    try:
        solution = generate_solution(model, tokenizer, question)
        
        print("\n" + "=" * 70)
        print("📖 Solution:")
        print("=" * 70)
        print(solution)
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error generating solution: {e}")
        sys.exit(1)
    
    # Cleanup
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()