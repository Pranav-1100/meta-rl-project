#!/usr/bin/env python3
"""Standalone GRPO smoke test — run BEFORE the full SFT+GRPO training cycle.

This script verifies that the entire GRPO pipeline (model load → dataset build →
rollout generation → reward callback → policy update) runs end-to-end without
crashing, on the actual model + actual env. It does NOT verify that GRPO learns
anything — most rollouts from a base, un-SFT'd model will be parse failures
collecting the format-error floor reward. The point is to catch:

  * Unsloth / TRL / transformers version mismatch
  * CUDA OOM on the chosen model size
  * Reward callback signature compatibility
  * GRPOConfig parameter regressions
  * PeriodicEvalCallback wiring

Run on HF Jobs A10G or Colab Pro GPU. CPU-only execution will fail at model load —
use ``tests/test_grpo_reward.py`` for the CPU-only reward-function smoke test.

Usage::

    # On HF Jobs / Colab with a GPU runtime:
    python scripts/grpo_smoke.py
    # OR with model override:
    python scripts/grpo_smoke.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 3

Time / cost: ~10 min on A10G, ~$0.20 in HF Jobs credits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="unsloth/gemma-2-9b-it-bnb-4bit",
        help="HF model ID (4-bit Unsloth recommended). Override to Qwen2.5-7B for "
        "smaller VRAM footprint or Gemma-3-1b for free-T4 dev.",
    )
    p.add_argument("--steps", type=int, default=3, help="Number of GRPO steps to run.")
    p.add_argument(
        "--prompts-per-task", type=int, default=4,
        help="Prompt batch size per task in the smoke dataset.",
    )
    p.add_argument(
        "--num-generations", type=int, default=4,
        help="GRPO group size — completions per prompt. Lower = faster smoke test.",
    )
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    args = p.parse_args()

    # Lazy imports — the script's CLI prints help without the heavy stack loaded.
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    from trl import GRPOConfig, GRPOTrainer  # type: ignore[import-not-found]
    from datasets import Dataset  # type: ignore[import-not-found]

    from phonepilot_env.agent_io import build_chat_prompt, observation_to_prompt
    from phonepilot_env.env import build_env
    from phonepilot_env.grpo_reward import rollout_reward
    from phonepilot_env.tasks import training_task_ids

    print("=" * 70)
    print(f"GRPO smoke test")
    print(f"  model: {args.model}")
    print(f"  steps: {args.steps}")
    print(f"  group_size: {args.num_generations}")
    print(f"  prompts_per_task: {args.prompts_per_task}")
    print("=" * 70)

    # ------------------------------------------------------------------ load model
    print("\n[1/4] loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(f"  ✓ model loaded ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params)")

    # ------------------------------------------------------------------ build dataset
    print("\n[2/4] building smoke prompt dataset...")
    rows = []
    # Pick a small subset of training tasks for the smoke run — Easy + Medium are
    # fastest and least likely to OOM on long observations.
    smoke_tasks = ["easy_ria_late", "medium_jay_standup"]
    for task_id in smoke_tasks:
        if task_id not in training_task_ids():
            continue
        for seed in range(1, args.prompts_per_task + 1):
            env = build_env()
            obs = env.reset(seed=seed, episode_id=f"smoke_{task_id}_{seed}", task_id=task_id)
            prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
            rows.append({"prompt": prompt, "task_id": task_id, "seed": seed})
    dataset = Dataset.from_list(rows)
    print(f"  ✓ {len(rows)} prompts across {len(smoke_tasks)} tasks")

    # ------------------------------------------------------------------ run GRPO
    print(f"\n[3/4] running GRPO for {args.steps} steps...")
    grpo_args = GRPOConfig(
        output_dir="/tmp/grpo-smoke",
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

    # ------------------------------------------------------------------ verify
    print("\n[4/4] post-training sanity...")
    FastLanguageModel.for_inference(model)
    env = build_env()
    obs = env.reset(seed=99, episode_id="smoke_post", task_id="easy_ria_late")
    prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    completion = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"  sample completion (truncated): {completion[:200]!r}...")
    try:
        from phonepilot_env.agent_io import AgentParseError, parse_completion_to_action
        action = parse_completion_to_action(completion)
        print(f"  ✓ parses: tool={action.body.tool}")
    except AgentParseError as e:
        print(f"  ⚠ parse failed (expected for un-SFT base model): {e}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — GRPO pipeline runs end-to-end.")
    print("Ready to proceed to full SFT + GRPO training run.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
