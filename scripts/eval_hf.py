#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.5.1",
#     "transformers>=4.51,<5.0",
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
"""PhonePilot — full eval (base vs SFT) on HF Jobs, uploads results to HF Hub.

Runs the 4 baselines that don't need a GPU (random, null, scripted_easy) AND the
GPU-needed `base` (vanilla model) and `sft` (model + adapter from hub) baselines
across all 17 tasks. Generates plots + uploads everything back to the hub.

Usage::

    hf jobs run --flavor a10g-large --secrets HF_TOKEN --timeout 5400 \\
        ghcr.io/astral-sh/uv:python3.12-bookworm uv run \\
        https://raw.githubusercontent.com/.../scripts/eval_hf.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --hub-repo pranav-1100/phonepilot-qwen7b \\
        --seeds 8
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
    p.add_argument("--model", required=True,
                   help="Base model HF ID (e.g. Qwen/Qwen2.5-7B-Instruct).")
    p.add_argument("--hub-repo", required=True,
                   help="HF Hub repo with sft_lora/ adapter (e.g. pranav-1100/phonepilot-qwen7b).")
    p.add_argument("--seeds", type=int, default=8,
                   help="Episodes per (baseline, task) pair. 8 → 8 × 17 × 5 baselines = 680 episodes.")
    p.add_argument("--max-steps", type=int, default=20,
                   help="Per-episode step cap (lower = faster).")
    p.add_argument("--repo-url", default="https://github.com/Pranav-1100/meta-rl-project.git")
    p.add_argument("--repo-branch", default="master")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--skip-base", action="store_true",
                   help="Skip the GPU-base baseline (saves ~5 min).")
    p.add_argument("--skip-sft", action="store_true",
                   help="Skip the SFT baseline (only for testing).")
    p.add_argument("--lying-rate-only", action="store_true",
                   help="Only run lying-rate eval on held-out adversarial battery.")
    args = p.parse_args()

    # ---------------------------------------------------------------- repo clone
    REPO = Path("/tmp/phonepilot")
    if not REPO.exists():
        print(f"[setup] cloning {args.repo_url}")
        subprocess.check_call(
            ["git", "clone", "-b", args.repo_branch, args.repo_url, str(REPO)],
        )
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "scripts"))
    os.chdir(str(REPO))

    print("[setup] importing heavy stack...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import HfApi, snapshot_download

    from phonepilot_env.actions import PhonePilotAction
    from phonepilot_env.agent_io import (
        AgentParseError,
        build_chat_prompt,
        observation_to_prompt,
        parse_completion_to_action,
    )
    from phonepilot_env.tasks import TASK_REGISTRY, training_task_ids

    # eval.py + run_episode.py local imports
    from eval import POLICIES, evaluate_one, evaluate_lying_rate, plot_staircase
    from run_episode import POLICIES as _POLICIES_ALIAS  # noqa

    print(f"[setup] model={args.model}")
    print(f"[setup] hub_repo={args.hub_repo}")
    print(f"[setup] seeds={args.seeds}")
    print(f"[setup] CUDA: {torch.cuda.is_available()}")

    api = HfApi()

    # ---------------------------------------------------------------- load model
    print("[model] loading base in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    print("[model] base loaded ✓")

    # ---------------------------------------------------------------- model-policy factory
    def make_model_policy(model, label: str):
        """Returns a (obs, rng) -> action_dict policy that runs the given model."""
        def policy(obs, rng):  # noqa: ANN001
            prompt = build_chat_prompt(tokenizer, observation_to_prompt(obs))
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
            try:
                action = parse_completion_to_action(completion)
                return {"body": action.body.model_dump(exclude={"metadata"})}
            except AgentParseError:
                return {"body": {"tool": "wait", "minutes": 5}}
        policy.__name__ = label
        return policy

    POLICIES["base"] = make_model_policy(base_model, "base")

    if not args.skip_sft:
        print("[model] loading SFT adapter...")
        adapter_root = snapshot_download(
            repo_id=args.hub_repo, allow_patterns="sft_lora/*"
        )
        sft_path = Path(adapter_root) / "sft_lora"
        sft_model = PeftModel.from_pretrained(base_model, str(sft_path))
        sft_model.eval()
        POLICIES["sft"] = make_model_policy(sft_model, "sft")
        print("[model] SFT adapter attached ✓")

    # ---------------------------------------------------------------- which baselines
    baselines = ["random", "null", "scripted_easy"]
    if not args.skip_base:
        baselines.append("base")
    if not args.skip_sft:
        baselines.append("sft")

    OUT = Path("/tmp/eval_output")
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL_EVAL_DIR = REPO / "data" / "eval"
    LOCAL_PLOTS_DIR = REPO / "data" / "plots"
    LOCAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- main eval
    summary_rows = []
    if not args.lying_rate_only:
        print(f"\n[eval] staircase: {len(baselines)} baselines × {len(TASK_REGISTRY)} tasks × {args.seeds} seeds")
        for baseline in baselines:
            for task_id in TASK_REGISTRY:
                t0 = time.time()
                row = evaluate_one(
                    baseline, task_id, args.seeds, args.max_steps, policy_override=None
                )
                summary_rows.append(row)
                dt = time.time() - t0
                print(
                    f"  {baseline:<14} {task_id:<35} succ={row['success_rate']:.0%}  "
                    f"r={row['mean_reward']:+.2f}  ({dt:.0f}s)"
                )
        summary_path = LOCAL_EVAL_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print(f"\n[eval] summary saved → {summary_path}")

        # Plots
        plot_path = LOCAL_PLOTS_DIR / "staircase.png"
        try:
            plot_staircase(summary_rows, plot_path)
            print(f"[plot] staircase → {plot_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[plot] staircase FAILED: {e}")

    # ---------------------------------------------------------------- lying-rate eval
    print(f"\n[eval] lying-rate: {len(baselines)} baselines × held-out adversarial battery")
    lying_rows = []
    for baseline in baselines:
        t0 = time.time()
        row = evaluate_lying_rate(baseline, n_seeds=max(3, args.seeds // 2), max_steps=args.max_steps)
        lying_rows.append(row)
        dt = time.time() - t0
        print(
            f"  {baseline:<14} lying={row['overall_lying_rate']:.0%}  "
            f"honest_fail={row['honest_fail_rate']:.0%}  "
            f"no_end={row['no_end_rate']:.0%}  ({dt:.0f}s)"
        )
    lying_path = LOCAL_EVAL_DIR / "lying_rate.json"
    lying_path.write_text(json.dumps(lying_rows, indent=2))
    print(f"\n[eval] lying-rate saved → {lying_path}")

    # ---------------------------------------------------------------- run plot scripts
    print("\n[plot] running calibration + honesty-vs-capability + dashboard")
    for script in ["plot_calibration.py", "plot_honesty_vs_capability.py", "plot_capability_dashboard.py"]:
        script_path = REPO / "scripts" / script
        if not script_path.exists():
            continue
        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(REPO),
                check=True,
                timeout=120,
            )
            print(f"  ✓ {script}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ {script} failed: {e}")

    # ---------------------------------------------------------------- upload artifacts
    print(f"\n[hub] uploading eval artifacts to {args.hub_repo}/eval/")
    try:
        api.upload_folder(
            folder_path=str(LOCAL_EVAL_DIR),
            repo_id=args.hub_repo,
            path_in_repo="eval",
            commit_message=f"eval: {len(baselines)} baselines, {args.seeds} seeds",
        )
        api.upload_folder(
            folder_path=str(LOCAL_PLOTS_DIR),
            repo_id=args.hub_repo,
            path_in_repo="plots",
            commit_message="eval plots",
        )
        print("[hub] uploads complete ✓")
    except Exception as e:  # noqa: BLE001
        print(f"[hub] upload failed: {e}")

    # ---------------------------------------------------------------- summary print
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    print("\nLying rate on held-out adversarial battery (lower=better):")
    for row in lying_rows:
        print(f"  {row['baseline']:<16} lying_rate = {row['overall_lying_rate']:.0%}")

    if summary_rows:
        print("\nMean reward by baseline (across all 17 tasks):")
        from collections import defaultdict
        agg: dict[str, list[float]] = defaultdict(list)
        for r in summary_rows:
            agg[r["baseline"]].append(r["mean_reward"])
        for b, vals in sorted(agg.items()):
            print(f"  {b:<16} mean_reward = {sum(vals)/len(vals):+.3f}")

    print("\n" + "=" * 70)
    print(f"Artifacts at: https://huggingface.co/{args.hub_repo}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
