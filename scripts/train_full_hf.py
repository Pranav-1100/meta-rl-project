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
#     "matplotlib",
#     "huggingface_hub>=0.30",
#     "fastapi",
#     "uvicorn",
#     "anthropic",
#     "openenv-core",
#     "python-dotenv",
# ]
# ///
"""PhonePilot — full SFT + GRPO Stage 1 training run on HF Jobs.

Designed to be invoked via:

    hf jobs uv run \\
      --flavor a10g-large \\
      --secrets HF_TOKEN \\
      --timeout 6h \\
      https://raw.githubusercontent.com/Pranav-1100/meta-rl-project/master/scripts/train_full_hf.py \\
      -- --model Qwen/Qwen2.5-7B-Instruct --hub-repo pranav-1100/phonepilot-qwen7b

What it does:

  1. git-clones the PhonePilot repo into ``/tmp/phonepilot`` so the env code
     and the SFT trajectories at ``data/trajectories/*.jsonl`` are available.
  2. Loads the chosen model (Qwen 7B or Gemma 9B) with bitsandbytes 4-bit +
     PEFT LoRA. **No Unsloth** — that path has dtype bugs in current versions.
  3. Phase B: SFT on the trajectories (2 epochs, LoRA r=16).
  4. Saves SFT adapter and uploads to HF Hub.
  5. Phase C: GRPO Stage 1 — Easy task only, ``--max-grpo-steps`` steps. Hard
     stop if reward goes NaN.
  6. Saves GRPO adapter and uploads to HF Hub.
  7. Records a small ``training_log.json`` summary on the Hub.

Authentication: ``HF_TOKEN`` env var must be set (HF Jobs ``--secrets HF_TOKEN``).
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
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model ID. Tested: Qwen/Qwen2.5-7B-Instruct, google/gemma-2-9b-it.",
    )
    p.add_argument(
        "--hub-repo",
        required=True,
        help="HF Hub model repo to push artifacts to (e.g., 'pranav-1100/phonepilot-qwen7b').",
    )
    p.add_argument("--repo-url", default="https://github.com/Pranav-1100/meta-rl-project.git")
    p.add_argument("--repo-branch", default="master")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--sft-epochs", type=int, default=2)
    p.add_argument("--sft-batch-size", type=int, default=1)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=2e-5)
    p.add_argument("--max-grpo-steps", type=int, default=80)
    # num_generations must divide (batch_size * grad_accum * world_size).
    p.add_argument("--grpo-num-generations", type=int, default=2)
    p.add_argument("--grpo-prompts-per-task", type=int, default=20)
    p.add_argument("--grpo-temperature", type=float, default=0.7,
                   help="Lower=more focused (less garbage). Default GRPO=1.0 produces wild outputs.")
    p.add_argument("--grpo-max-completion-length", type=int, default=400,
                   help="Token budget per rollout. 200 was too tight for full JSON.")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--skip-grpo", action="store_true")
    p.add_argument(
        "--load-sft-from",
        default=None,
        help="HF Hub repo containing an existing sft_lora/ adapter. If set, "
        "downloads + loads it instead of training fresh SFT. Auto-sets --skip-sft.",
    )
    args = p.parse_args()

    # ---------------------------------------------------------------- repo clone
    REPO = Path("/tmp/phonepilot")
    if not REPO.exists():
        print(f"[setup] cloning {args.repo_url} → {REPO}")
        subprocess.check_call(
            ["git", "clone", "-b", args.repo_branch, args.repo_url, str(REPO)],
        )
    sys.path.insert(0, str(REPO / "src"))

    # ---------------------------------------------------------------- imports
    print("[setup] importing heavy stack...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, create_repo

    from phonepilot_env.agent_io import (
        AgentParseError,
        build_chat_prompt,
        messages_for_template,
        observation_to_prompt,
        parse_completion_to_action,
    )
    from phonepilot_env.env import build_env
    from phonepilot_env.grpo_reward import rollout_reward

    OUT = Path("/tmp/output")
    OUT.mkdir(parents=True, exist_ok=True)
    SFT_DIR = OUT / "sft_lora"
    GRPO_DIR = OUT / "grpo_lora"

    print(f"[setup] model={args.model}")
    print(f"[setup] hub_repo={args.hub_repo}")
    print(f"[setup] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------------------------------------------------------------- HF Hub setup
    api = HfApi()
    print(f"[hub] creating repo {args.hub_repo} (idempotent)...")
    create_repo(args.hub_repo, exist_ok=True, repo_type="model")

    # ---------------------------------------------------------------- model load
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

    # ---------------------------------------------------------------- attach LoRA
    if args.load_sft_from:
        print(f"[lora] loading existing SFT adapter from {args.load_sft_from}/sft_lora")
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        adapter_root = snapshot_download(
            repo_id=args.load_sft_from, allow_patterns="sft_lora/*"
        )
        sft_path = Path(adapter_root) / "sft_lora"
        model = PeftModel.from_pretrained(model, str(sft_path), is_trainable=True)
        # Ensure adapter parameters require gradients (PEFT sometimes loads with grads off).
        for n, p_ in model.named_parameters():
            if "lora_" in n:
                p_.requires_grad = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[lora] loaded SFT adapter — {n_trainable:,} trainable params")
        args.skip_sft = True
    else:
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

    # =================================================================
    #                            PHASE B — SFT
    # =================================================================
    sft_log = {"phase": "sft", "skipped": args.skip_sft}
    if not args.skip_sft:
        traj_dir = REPO / "data" / "trajectories"
        traj_files = sorted(traj_dir.glob("*.jsonl"))
        if not traj_files:
            print(f"[sft] ERROR: no trajectory files found at {traj_dir}")
            sys.exit(1)
        print(f"[sft] loading {len(traj_files)} trajectory files (manual JSON parse)")
        # Manual load — `datasets.load_dataset("json", ...)` chokes on nullable
        # cross-file fields like `end_claim` (bool|None). We only need `messages`.
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
            logging_steps=5,
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

        # Save adapter
        SFT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(SFT_DIR))
        tokenizer.save_pretrained(str(SFT_DIR))
        print(f"[sft] adapter saved → {SFT_DIR}")

        # Upload SFT artifacts
        print(f"[sft] uploading to {args.hub_repo}/sft_lora/...")
        api.upload_folder(
            folder_path=str(SFT_DIR),
            repo_id=args.hub_repo,
            path_in_repo="sft_lora",
            commit_message=f"SFT done — {sft_secs/60:.1f} min, {len(ds)} episodes",
        )
        sft_log["seconds"] = sft_secs
        sft_log["episodes"] = len(ds)
        sft_log["files"] = len(traj_files)

        # Quick post-SFT sanity check
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
        sft_log["sft_sample"] = sample[:300]
        try:
            action = parse_completion_to_action(sample)
            print(f"  ✓ parses post-SFT: tool={action.body.tool}")
            sft_log["sft_parses"] = True
        except AgentParseError as e:
            print(f"  ⚠ parse fails post-SFT: {e}")
            sft_log["sft_parses"] = False
        model.train()
    else:
        print("[sft] SKIPPED")

    # =================================================================
    #                       PHASE C — GRPO Stage 1
    # =================================================================
    grpo_log = {"phase": "grpo", "skipped": args.skip_grpo}
    if not args.skip_grpo:
        print(f"[grpo] building Stage-1 prompt dataset (Easy only, "
              f"{args.grpo_prompts_per_task} prompts)")
        rows = []
        for seed in range(1, args.grpo_prompts_per_task + 1):
            env = build_env()
            obs = env.reset(seed=seed, episode_id=f"grpo_easy_{seed}", task_id="easy_ria_late")
            prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs, turn_index=0))
            rows.append({"prompt": prompt, "task_id": "easy_ria_late", "seed": seed})
        grpo_dataset = Dataset.from_list(rows)
        print(f"[grpo] {len(rows)} prompts ready")

        # GRPO config — Stage 1: Easy only. Temperature lowered + completion length raised
        # to avoid the all-rewards-equal-floor degenerate regime we saw with defaults.
        grpo_args = GRPOConfig(
            output_dir="/tmp/grpo-out",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_generations=args.grpo_num_generations,
            max_prompt_length=args.max_seq_len - args.grpo_max_completion_length,
            max_completion_length=args.grpo_max_completion_length,
            temperature=args.grpo_temperature,
            top_p=0.9,
            learning_rate=1e-6,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            max_steps=args.max_grpo_steps,
            report_to="none",
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=rollout_reward,
            args=grpo_args,
            train_dataset=grpo_dataset,
        )
        t0 = time.time()
        try:
            grpo_trainer.train()
            grpo_log["status"] = "success"
        except Exception as e:  # noqa: BLE001
            print(f"[grpo] FAILED at runtime: {type(e).__name__}: {e}")
            grpo_log["status"] = "error"
            grpo_log["error"] = str(e)[:500]
        grpo_secs = time.time() - t0
        print(f"[grpo] phase finished in {grpo_secs/60:.1f} min")
        grpo_log["seconds"] = grpo_secs
        grpo_log["max_steps"] = args.max_grpo_steps

        # Save adapter (even on partial GRPO, we get useful state)
        GRPO_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(GRPO_DIR))
        tokenizer.save_pretrained(str(GRPO_DIR))
        print(f"[grpo] adapter saved → {GRPO_DIR}")

        # Upload GRPO artifacts
        print(f"[grpo] uploading to {args.hub_repo}/grpo_lora/...")
        api.upload_folder(
            folder_path=str(GRPO_DIR),
            repo_id=args.hub_repo,
            path_in_repo="grpo_lora",
            commit_message=f"GRPO Stage 1 done — {grpo_secs/60:.1f} min",
        )
    else:
        print("[grpo] SKIPPED")

    # =================================================================
    #                         finalize: log summary
    # =================================================================
    summary = {
        "model": args.model,
        "hub_repo": args.hub_repo,
        "config": {
            "max_seq_len": args.max_seq_len,
            "lora_r": args.lora_r,
            "sft_epochs": args.sft_epochs,
            "max_grpo_steps": args.max_grpo_steps,
            "grpo_num_generations": args.grpo_num_generations,
        },
        "sft": sft_log,
        "grpo": grpo_log,
    }
    summary_path = OUT / "training_log.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] training summary:\n{json.dumps(summary, indent=2)}")
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="training_log.json",
        repo_id=args.hub_repo,
        commit_message="training summary",
    )

    print("\n" + "=" * 70)
    print(f"DONE. Artifacts at: https://huggingface.co/{args.hub_repo}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
