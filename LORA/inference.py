#!/usr/bin/env python3
"""Simple inference helper for the fine-tuned LoRA model."""

import argparse
import logging
from pathlib import Path

import mlx.core as mx

from mlx_lm.utils import load
from mlx_lm.tuner.lora import LoRALinear

from lora import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def apply_lora_layers(model, config: TrainingConfig):
    """Mirror the LoRA wiring used during training."""
    lora_scale = config.lora_alpha / config.lora_rank
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            if hasattr(layer.self_attn, name):
                original_linear = getattr(layer.self_attn, name)
                out_dim, in_dim = original_linear.weight.shape
                lora_layer = LoRALinear(
                    input_dims=in_dim,
                    output_dims=out_dim,
                    r=config.lora_rank,
                    scale=lora_scale,
                    dropout=0.0,
                )
                lora_layer.linear = original_linear
                setattr(layer.self_attn, name, lora_layer)


def load_with_adapters(config: TrainingConfig, adapter_file: Path):
    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_file}")

    logging.info("Loading base MLX model from %s", config.mlx_model_path)
    model, tokenizer = load(config.mlx_model_path)

    logging.info("Applying LoRA adapters (rank=%d, alpha=%.1f)...", config.lora_rank, config.lora_alpha)
    apply_lora_layers(model, config)

    logging.info("Loading fine-tuned adapter weights from %s", adapter_file)
    model.load_weights(str(adapter_file))
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens, dtype=mx.int32)[None, :]
    generated = tokens[:]

    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        next_logits = logits[:, -1, :] / temperature
        next_token = int(mx.argmax(next_logits, axis=-1).item())
        generated.append(next_token)

        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            break

        input_ids = mx.array(generated, dtype=mx.int32)[None, :]

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned LoRA model.")
    parser.add_argument("--prompt", type=str, default="Solve: What is 12 * 17?",
                        help="Prompt to feed the model.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (1.0 = greedy).")
    parser.add_argument("--adapter-path", type=Path, default=Path("adapters/adapters.safetensors"),
                        help="Path to the saved adapter weights (safetensors file).")
    args = parser.parse_args()

    config = TrainingConfig()
    model, tokenizer = load_with_adapters(config, args.adapter_path)

    logging.info("Generating response...")
    output = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
    print("\n=== Model Output ===\n")
    print(output)


if __name__ == "__main__":
    main()
