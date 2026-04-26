#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "transformers>=4.51,<5.0",
#     "trl>=0.18,<0.20",
#     "peft>=0.14,<0.17",
#     "accelerate>=1.0",
#     "bitsandbytes>=0.43.0",
#     "datasets>=3.0",
#     "pydantic>=2.9",
#     "huggingface_hub>=0.30",
#     "fastapi",
#     "uvicorn",
#     "anthropic",
#     "openenv-core",
#     "python-dotenv",
# ]
# ///
"""PhonePilot — SFT-only training run on HF Jobs.

Differences from ``train_full_hf.py``:

  * **No GRPO** — only Phase B (SFT). The GRPO regime was unstable on the
    post-SFT distribution; for the hackathon submission we report SFT-only.
  * **Saves the trainer's per-step ``log_history``** into ``training_log.json``
    on the Hub, so we have real loss-curve evidence (the previous script only
    saved summary stats and we ended up with an empty plot).

Usage on HF Jobs (vinnykc08 — Gemma 2 9B SFT)::

    hf jobs run --flavor a10g-large --secrets HF_TOKEN --timeout 5400 \\
      ghcr.io/astral-sh/uv:python3.12-bookworm uv run \\
      https://raw.githubusercontent.com/Pranav-1100/meta-rl-project/master/scripts/train_sft_only.py \\
      --model google/gemma-2-9b-it \\
      --hub-repo vinnykc08/phonepilot-gemma9b

Authentication: ``HF_TOKEN`` must be set (HF Jobs ``--secrets HF_TOKEN``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-2-9b-it",
                   help="HF model ID. Tested: google/gemma-2-9b-it, Qwen/Qwen2.5-7B-Instruct.")
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub model repo (e.g., 'vinnykc08/phonepilot-gemma9b').")
    p.add_argument("--repo-url", default="https://github.com/Pranav-1100/meta-rl-project.git")
    p.add_argument("--repo-branch", default="master")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--sft-batch-size", type=int, default=1)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--logging-steps", type=int, default=5,
                   help="How often the trainer emits a {'loss': ...} log line.")
    args = p.parse_args()

    REPO = Path("/tmp/phonepilot")
    if not REPO.exists():
        print(f"[setup] cloning {args.repo_url} → {REPO}")
        subprocess.check_call(
            ["git", "clone", "-b", args.repo_branch, args.repo_url, str(REPO)],
        )
    sys.path.insert(0, str(REPO / "src"))

    print("[setup] importing heavy stack...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    from huggingface_hub import HfApi, create_repo

    from phonepilot_env.agent_io import (
        AgentParseError,
        build_chat_prompt,
        messages_for_template,
        observation_to_prompt,
        parse_completion_to_action,
    )
    from phonepilot_env.env import build_env

    OUT = Path("/tmp/output")
    OUT.mkdir(parents=True, exist_ok=True)
    SFT_DIR = OUT / "sft_lora"

    print(f"[setup] model={args.model}")
    print(f"[setup] hub_repo={args.hub_repo}")
    print(f"[setup] CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    api = HfApi()
    print(f"[hub] creating repo {args.hub_repo} (idempotent)...")
    create_repo(args.hub_repo, exist_ok=True, repo_type="model")

    print("[model] loading 4-bit quantized base...")
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print(f"[lora] attaching adapters (r={args.lora_r})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] {n_trainable:,} trainable params")

    # --------------------------------------------------------------- SFT
    traj_dir = REPO / "data" / "trajectories"
    traj_files = sorted(traj_dir.glob("*.jsonl"))
    if not traj_files:
        print(f"[sft] ERROR: no trajectory files at {traj_dir}")
        sys.exit(1)
    print(f"[sft] loading {len(traj_files)} trajectory files (manual JSON parse)")
    all_msgs = []
    for f in traj_files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            if "messages" in ep:
                all_msgs.append({"messages": ep["messages"]})
    print(f"[sft] {len(all_msgs)} episodes loaded")
    ds = Dataset.from_list(all_msgs)

    def to_chat_text(row):
        msgs = messages_for_template(tokenizer, row["messages"])
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    train_ds = ds.map(to_chat_text, remove_columns=ds.column_names)

    sft_args = SFTConfig(
        output_dir="/tmp/sft-out",
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accum,
        learning_rate=args.sft_lr,
        num_train_epochs=args.sft_epochs,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        packing=False,
        gradient_checkpointing=True,
    )
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
    )
    t0 = time.time()
    sft_trainer.train()
    sft_secs = time.time() - t0
    print(f"[sft] done in {sft_secs/60:.1f} min")

    # The fix vs train_full_hf.py: capture the trainer's log_history.
    # This is the per-step record of {'loss', 'epoch', 'learning_rate', ...}
    # that lets us plot a real loss curve.
    log_history = list(sft_trainer.state.log_history)
    loss_points = [r for r in log_history if "loss" in r]
    print(f"[sft] captured {len(loss_points)} loss points "
          f"({loss_points[0]['loss']:.3f} → {loss_points[-1]['loss']:.3f})"
          if loss_points else "[sft] no loss points captured")

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(SFT_DIR))
    tokenizer.save_pretrained(str(SFT_DIR))
    print(f"[sft] adapter saved → {SFT_DIR}")

    print(f"[sft] uploading adapter to {args.hub_repo}/sft_lora/...")
    api.upload_folder(
        folder_path=str(SFT_DIR),
        repo_id=args.hub_repo,
        path_in_repo="sft_lora",
        commit_message=f"SFT done — {sft_secs/60:.1f} min, {len(ds)} episodes",
    )

    # post-SFT sanity check
    print("[sft] sanity check on easy_ria_late...")
    model.eval()
    env = build_env()
    obs = env.reset(seed=1, episode_id="sft-check", task_id="easy_ria_late")
    prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    sample = tokenizer.decode(
        out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    parses = False
    try:
        action = parse_completion_to_action(sample)
        print(f"  ✓ parses post-SFT: tool={action.body.tool}")
        parses = True
    except AgentParseError as e:
        print(f"  ⚠ parse fails post-SFT: {e}")

    summary = {
        "model": args.model,
        "hub_repo": args.hub_repo,
        "config": {
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "sft_epochs": args.sft_epochs,
            "sft_lr": args.sft_lr,
            "sft_batch_size": args.sft_batch_size,
            "sft_grad_accum": args.sft_grad_accum,
            "logging_steps": args.logging_steps,
        },
        "sft": {
            "phase": "sft",
            "seconds": sft_secs,
            "episodes": len(ds),
            "files": len(traj_files),
            "sft_sample": sample[:300],
            "sft_parses": parses,
            "loss_first": loss_points[0]["loss"] if loss_points else None,
            "loss_last": loss_points[-1]["loss"] if loss_points else None,
            "loss_history": loss_points,
            "full_log_history": log_history,
        },
    }
    summary_path = OUT / "training_log.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] summary saved with {len(loss_points)} loss points")
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="training_log.json",
        repo_id=args.hub_repo,
        commit_message=f"SFT log — {len(loss_points)} loss points",
    )

    print("\n" + "=" * 70)
    print(f"DONE. Artifacts at: https://huggingface.co/{args.hub_repo}")
    print(f"     SFT loss: {loss_points[0]['loss']:.3f} → {loss_points[-1]['loss']:.3f} "
          if loss_points else "     (no loss history captured)", "in", f"{sft_secs/60:.1f} min")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
