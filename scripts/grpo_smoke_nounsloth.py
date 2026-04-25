#!/usr/bin/env python3
"""GRPO smoke test WITHOUT Unsloth — uses standard transformers + PEFT + TRL.

Why this exists: Unsloth's `fast_lora` kernel has a known dtype mismatch bug with
torch 2.10's new autocast API ("got Half and Float"). Pinning Unsloth versions
hasn't reliably worked. This script bypasses Unsloth entirely — at the cost of
slightly more VRAM and slower training, but it WORKS.

Use this for the cloud-GPU smoke test. For real training tomorrow we can either
re-attempt Unsloth (with more patience), or use this script's approach (slower
but reliable). The reward function and env behavior are identical either way.

Run with::

    python scripts/grpo_smoke_nounsloth.py --steps 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Quiet a noisy warning from tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID. Use the *original* (un-quantized) model — "
        "we apply 4-bit quantization on the fly via bitsandbytes.",
    )
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--prompts-per-task", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    args = p.parse_args()

    # Heavy imports lazy so --help is fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset

    from phonepilot_env.agent_io import (
        AgentParseError,
        build_chat_prompt,
        observation_to_prompt,
        parse_completion_to_action,
    )
    from phonepilot_env.env import build_env
    from phonepilot_env.grpo_reward import rollout_reward

    print("=" * 70)
    print("GRPO smoke test (no-Unsloth path)")
    print(f"  model: {args.model}")
    print(f"  steps: {args.steps}")
    print(f"  group_size: {args.num_generations}")
    print(f"  prompts_per_task: {args.prompts_per_task}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print("=" * 70)

    # ------------------------------------------------------------------ load model
    print("\n[1/4] loading model + tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("  ✓ model + LoRA ready")

    # ------------------------------------------------------------------ build dataset
    print("\n[2/4] building smoke prompt dataset...")
    rows = []
    smoke_tasks = ["easy_ria_late", "medium_jay_standup"]
    for task_id in smoke_tasks:
        for seed in range(1, args.prompts_per_task + 1):
            env = build_env()
            obs = env.reset(seed=seed, episode_id=f"smoke_{task_id}_{seed}", task_id=task_id)
            prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
            rows.append({"prompt": prompt, "task_id": task_id, "seed": seed})
    dataset = Dataset.from_list(rows)
    print(f"  ✓ {len(rows)} prompts across {len(smoke_tasks)} tasks")

    # ------------------------------------------------------------------ run GRPO
    print(f"\n[3/4] running GRPO for {args.steps} steps (no-Unsloth)...")
    grpo_args = GRPOConfig(
        output_dir="/tmp/grpo-smoke-nounsloth",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_len - 256,
        max_completion_length=200,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        max_steps=args.steps,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rollout_reward,
        args=grpo_args,
        train_dataset=dataset,
    )
    trainer.train()
    print(f"  ✓ {args.steps} GRPO steps completed without crash")

    # ------------------------------------------------------------------ sanity
    print("\n[4/4] post-training sanity...")
    model.eval()
    env = build_env()
    obs = env.reset(seed=99, episode_id="smoke_post", task_id="easy_ria_late")
    prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"  sample completion (truncated): {completion[:200]!r}")
    try:
        action = parse_completion_to_action(completion)
        print(f"  ✓ parses: tool={action.body.tool}")
    except AgentParseError as e:
        print(f"  ⚠ parse failed (expected for un-SFT base model): {e}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — GRPO pipeline runs end-to-end.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
