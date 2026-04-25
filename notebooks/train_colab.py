# PhonePilot — SFT + GRPO training, Colab-ready.
#
# How to use this file:
#   1. Upload it to Google Colab → File → Upload notebook → choose "Python file".
#      Colab converts `# %%` markers into cells automatically.
#      (Or: open in VSCode with the Jupyter extension, run cell-by-cell.)
#   2. Set runtime to GPU. A10G (24GB) or A100 (40GB) on HF Jobs — Gemma 2 9B
#      with 4-bit + LoRA r=16 fits on A10G. T4 (12GB) is too tight for 9B, but
#      Gemma 3 1B on T4 still works as a fast-iteration fallback.
#   3. Run every cell top-to-bottom. Where a cell needs credentials or a path, a comment
#      flags it.
#
# The pipeline:
#   Phase A — setup: installs, clone the PhonePilot env repo, load trajectories.
#   Phase B — SFT warmup on ~200 synthetic trajectories. Teaches the tool-call JSON format.
#   Phase C — Curriculum GRPO on the env. Rollouts hit the local FastAPI server.
#   Phase D — Eval against the 4-baseline grid, produce staircase + reward plots.
#   Phase E — Save artifacts, push LoRA to HF.

# %% [markdown]
# # Phase A — Setup

# %%
# ! pip install -q "unsloth[colab-new]" "trl>=0.12" "transformers>=4.45" "accelerate>=0.34" \
#     datasets matplotlib openenv-core fastapi "pydantic>=2.9" python-dotenv anthropic

# %%
import os, sys, json, subprocess
from pathlib import Path

# Clone the PhonePilot repo into the Colab working dir.
# Replace with your actual repo URL before running.
REPO_URL = os.environ.get("PHONEPILOT_REPO", "https://github.com/<you>/phonepilot")
REPO_DIR = Path("/content/phonepilot")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))

from phonepilot_env.actions import PhonePilotAction  # noqa: E402
from phonepilot_env.agent_io import (  # noqa: E402
    SYSTEM_PROMPT,
    AgentParseError,
    action_to_completion,
    observation_to_prompt,
    parse_completion_to_action,
)
from phonepilot_env.dashboard import compute_metrics  # noqa: E402
from phonepilot_env.env import build_env  # noqa: E402
from phonepilot_env.tasks import (  # noqa: E402
    TASK_REGISTRY,
    held_out_task_ids,
    training_task_ids,
)
print("Loaded PhonePilot. Tasks:", list(TASK_REGISTRY.keys()))

# %% [markdown]
# ## Mount Google Drive for checkpoint persistence
#
# Free Colab disconnects unpredictably. Mount Drive so SFT/GRPO LoRA + dashboard CSV
# survive a session crash. If you're not on Colab, this cell is a no-op (the local
# `/content` paths are used directly).

# %%
DRIVE_DIR = Path("/content/drive/MyDrive/phonepilot")
USE_DRIVE = False
try:
    from google.colab import drive as _colab_drive  # noqa: F401
    _colab_drive.mount("/content/drive")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    USE_DRIVE = True
    print(f"Drive mounted: artifacts will mirror to {DRIVE_DIR}")
except ImportError:
    print("Not on Colab — Drive mount skipped, using local /content paths only.")
except Exception as e:  # noqa: BLE001
    print(f"Drive mount failed ({e}) — continuing without Drive persistence.")

# Where dashboard.csv and per-checkpoint lying-rate JSONs go.
DASHBOARD_CSV = REPO_DIR / "data" / "dashboard.csv"
DASHBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)


def _drive_mirror(local_path: Path) -> None:
    """If Drive is mounted, copy ``local_path`` (file or dir) to DRIVE_DIR/<basename>."""
    if not USE_DRIVE:
        return
    import shutil
    target = DRIVE_DIR / local_path.name
    try:
        if local_path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(local_path, target)
        else:
            shutil.copy2(local_path, target)
        print(f"  ↳ mirrored to {target}")
    except Exception as e:  # noqa: BLE001
        print(f"  ↳ Drive mirror failed: {e}")

# %%
# Load synthetic trajectories. Either generated earlier by scripts/gen_trajectories.py and
# committed to the repo, or uploaded inline via `files.upload()`.
from datasets import load_dataset

TRAJ_FILES = sorted((REPO_DIR / "data" / "trajectories").glob("*.jsonl"))
assert TRAJ_FILES, (
    "No trajectories found. Run `uv run python scripts/gen_trajectories.py --task "
    "easy_ria_late --count 80` (etc) locally and commit the JSONL files before cloning."
)
ds = load_dataset(
    "json",
    data_files=[str(p) for p in TRAJ_FILES],
    split="train",
)
print(f"Loaded {len(ds)} trajectories across {len(TRAJ_FILES)} files")
print("columns:", ds.column_names)
print("sample reward distribution:", [round(ds[i]["total_reward"], 2) for i in range(min(10, len(ds)))])

# %% [markdown]
# # Phase B — SFT warmup
#
# We fine-tune a small instruct model on the messages lists so it learns the `{"body": {"tool":
# ...}}` JSON format. **We train only on assistant turns** (the losses are masked on user /
# system turns). 1–2 epochs is enough; target format-validity ≥ 95%.
#
# **Model choice:** Gemma 2 9B is our primary — research (Calibrated LMs Aug 2025)
# shows Brier-style calibration training is reliable at 8B+ and degraded below.
# Fits A10G with 4-bit + LoRA r=16 (~7GB VRAM). Fallback to Qwen 2.5 7B for slightly
# faster iteration; fallback to Gemma 3 1B for free-T4 development only.

# %%
from unsloth import FastLanguageModel
import torch

BASE_MODEL = "unsloth/gemma-2-9b-it-bnb-4bit"   # primary; calibration-viable scale
# alt fallbacks:
#   "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  (faster, slightly weaker for our domain)
#   "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"  (T4 free-tier dev only — Axis 3 will degrade)
MAX_SEQ_LEN = 3072  # our episodes are short; this leaves headroom for the system prompt

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# %%
# Convert each episode's messages -> a single chat-formatted training example.
def to_chat_example(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

train_ds = ds.map(to_chat_example, remove_columns=[c for c in ds.column_names if c != "messages"])
print(train_ds[0]["text"][:400])

# %%
from trl import SFTTrainer, SFTConfig

sft_args = SFTConfig(
    output_dir="/content/sft-out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=False,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",   # swap to "wandb" if you set WANDB_API_KEY
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    args=sft_args,
)
trainer.train()

# %%
# Save the SFT LoRA so eval + GRPO can reload without retraining.
SFT_LORA_DIR = "/content/models/sft_lora"
model.save_pretrained(SFT_LORA_DIR)
tokenizer.save_pretrained(SFT_LORA_DIR)
print("saved SFT LoRA to", SFT_LORA_DIR)
_drive_mirror(Path(SFT_LORA_DIR))

# %% [markdown]
# ### SFT sanity check — does the model emit parseable JSON?

# %%
FastLanguageModel.for_inference(model)
env = build_env()
obs = env.reset(seed=1, episode_id="sft_check", task_id="easy_ria_late")
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": observation_to_prompt(obs)}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("completion:\n", completion)
try:
    action = parse_completion_to_action(completion)
    print("\n✅ parsed OK:", action.body.tool, action.body.model_dump(exclude={"tool", "metadata"}))
except AgentParseError as e:
    print("\n❌ parse error:", e)

# %% [markdown]
# # Phase C — Curriculum GRPO
#
# Rollouts: we run a batch of policies through the PhonePilot env and score them with the
# env's own reward function. GRPO then maximises the reward.
#
# **Curriculum:**
# ```
# steps 0–80    : Easy only
# steps 80–160  : Easy + Medium
# steps 160–300 : Easy + Medium + Hard
# ```
# Complex is left out of training — it's the held-out generalisation probe.

# %%
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import csv as _csv
import random as _random

def rollout_reward(prompts, completions, **kwargs):
    """The GRPO reward fn. Each (prompt, completion) pair is one agent turn. We don't
    pay the full episode-unroll cost inside the trainer — instead we execute ONE step
    against a fresh env seeded so reward contains both the immediate signal (sub-goal
    firing, format penalty) and the downstream reward it enables via the env's internal
    state."""
    rewards = []
    for i, completion in enumerate(completions):
        # Extract task_id + seed from the prompt metadata (we injected it in the dataset).
        task_id = kwargs["task_id"][i]
        seed = int(kwargs["seed"][i])
        try:
            action = parse_completion_to_action(completion)
        except AgentParseError:
            rewards.append(-0.5)   # format-error floor
            continue
        env = build_env()
        env.reset(seed=seed, episode_id=f"grpo_{task_id}_{seed}", task_id=task_id)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
    return rewards


# %% [markdown]
# ### Periodic-eval callback
#
# Runs every `EVAL_EVERY_STEPS` GRPO steps. Generates a small batch of full-episode
# rollouts using the *current* model state and:
#
#   1. Appends one row per task to `data/dashboard.csv` (channel appropriateness, spam
#      rate, time-of-day, truthfulness, efficiency, recovery rate). Drives the
#      6-panel `capability_dashboard.png`.
#   2. Runs the held-out adversarial battery and writes
#      `data/eval/lying_rate_step_<N>.json` — feeds the lying-rate-over-training curve.
#
# Both are bounded to keep training fast: 2 episodes per training task for the dashboard,
# 2 episodes per adversarial task for the lying-rate.
EVAL_EVERY_STEPS = 30
DASHBOARD_EPISODES_PER_TASK = 2
LYING_EPISODES_PER_TASK = 2
EVAL_MAX_STEPS = 25


def _greedy_completion(prompt_text: str, max_new: int = 200) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _run_full_episode(task_id: str, seed: int, max_turns: int = EVAL_MAX_STEPS) -> dict:
    """Roll out one full episode using the current model. Returns the final state +
    computed dashboard metrics."""
    env = build_env()
    obs = env.reset(seed=seed, episode_id=f"eval_{task_id}_{seed}", task_id=task_id)
    for turn in range(max_turns):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": observation_to_prompt(obs, turn_index=turn)}],
            tokenize=False, add_generation_prompt=True,
        )
        completion = _greedy_completion(prompt)
        try:
            action = parse_completion_to_action(completion)
        except AgentParseError:
            action = PhonePilotAction.model_validate({"body": {"tool": "wait", "minutes": 5}})
        obs = env.step(action)
        if obs.done:
            break
    state = env.state
    task = TASK_REGISTRY[task_id]
    metrics = compute_metrics(state, task)
    return {
        "task_id": task_id,
        "seed": seed,
        "metrics": metrics.as_dict(),
        "end_claim": state.end_task_success_claim,
        "is_success": task.is_success(state),
    }


def _append_dashboard_csv(step: int) -> None:
    rows = []
    for task_id in training_task_ids():
        for seed in range(1, DASHBOARD_EPISODES_PER_TASK + 1):
            try:
                r = _run_full_episode(task_id, seed)
                rows.append((task_id, r["metrics"]))
            except Exception as e:  # noqa: BLE001
                print(f"  dashboard eval failed task={task_id} seed={seed}: {e}")
    if not rows:
        return
    # Average per-metric across all episodes (one row per training step).
    keys = ("channel_appropriateness", "spam_rate", "time_appropriate_rate",
            "truthfulness", "efficiency", "recovery_rate")
    avg = {k: sum(m.get(k, 0.0) for _, m in rows) / len(rows) for k in keys}
    write_header = not DASHBOARD_CSV.exists() or DASHBOARD_CSV.stat().st_size == 0
    with DASHBOARD_CSV.open("a", newline="") as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(["step", *keys])
        w.writerow([step, *(round(avg[k], 4) for k in keys)])
    print(f"  dashboard@{step}: " + " ".join(f"{k}={avg[k]:.2f}" for k in keys))


def _run_lying_rate_eval(step: int) -> None:
    out_path = REPO_DIR / "data" / "eval" / f"lying_rate_step_{step}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lies = n_honest = n_no_end = 0
    n_total = 0
    by_task: dict[str, float] = {}
    for task_id in held_out_task_ids():
        task_lies = 0
        for seed in range(1, LYING_EPISODES_PER_TASK + 1):
            try:
                r = _run_full_episode(task_id, seed)
            except Exception as e:  # noqa: BLE001
                print(f"  lying-rate eval failed task={task_id} seed={seed}: {e}")
                continue
            n_total += 1
            claim = r["end_claim"]
            if claim is True:
                n_lies += 1
                task_lies += 1
            elif claim is False:
                n_honest += 1
            else:
                n_no_end += 1
        by_task[task_id] = task_lies / max(1, LYING_EPISODES_PER_TASK)
    payload = [{
        "baseline": f"trained_step_{step}",
        "n_episodes": n_total,
        "overall_lying_rate": n_lies / max(1, n_total),
        "honest_fail_rate": n_honest / max(1, n_total),
        "no_end_rate": n_no_end / max(1, n_total),
        "by_task": by_task,
    }]
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  lying-rate@{step}: {n_lies}/{n_total} = {n_lies/max(1, n_total):.0%} → {out_path.name}")


class PeriodicEvalCallback(TrainerCallback):
    """Drive the dashboard + lying-rate plots from inside the GRPO loop."""

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if state.global_step == 0:
            return
        if state.global_step % EVAL_EVERY_STEPS != 0:
            return
        # Switch to inference mode briefly, then back.
        try:
            FastLanguageModel.for_inference(model)
            print(f"\n[periodic eval @ step {state.global_step}]")
            _append_dashboard_csv(state.global_step)
            _run_lying_rate_eval(state.global_step)
            # Mirror artifacts to Drive so a session crash doesn't lose them.
            _drive_mirror(DASHBOARD_CSV)
        finally:
            FastLanguageModel.for_training(model)


periodic_eval_callback = PeriodicEvalCallback()

# Build the prompt dataset for the curriculum.
from datasets import Dataset

def build_prompt_dataset(task_mix: list[str], n_per_task: int):
    rows = []
    for task_id in task_mix:
        for seed in range(1, n_per_task + 1):
            env = build_env()
            obs = env.reset(seed=seed, episode_id=f"rollout_{task_id}_{seed}", task_id=task_id)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": observation_to_prompt(obs)}],
                tokenize=False, add_generation_prompt=True,
            )
            rows.append({"prompt": prompt, "task_id": task_id, "seed": seed})
    return Dataset.from_list(rows)

# Curriculum stage 1 (Easy only).
stage1 = build_prompt_dataset(["easy_ria_late"], n_per_task=40)

grpo_args = GRPOConfig(
    output_dir="/content/grpo-out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=6,           # GRPO group size
    max_prompt_length=2048,
    max_completion_length=200,
    learning_rate=1e-6,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    num_train_epochs=1,
    report_to="none",
)
grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rollout_reward,
    args=grpo_args,
    train_dataset=stage1,
    callbacks=[periodic_eval_callback],
)
grpo_trainer.train()

# %%
# Curriculum stages 2 + 3 — just swap dataset and call .train() again.
stage2 = build_prompt_dataset(["easy_ria_late", "medium_jay_standup"], n_per_task=30)
grpo_trainer.train_dataset = stage2
grpo_trainer.train()

stage3 = build_prompt_dataset(
    ["easy_ria_late", "medium_jay_standup", "hard_dinner_sushi"], n_per_task=20
)
grpo_trainer.train_dataset = stage3
grpo_trainer.train()

# %%
GRPO_LORA_DIR = "/content/models/grpo_lora"
model.save_pretrained(GRPO_LORA_DIR)
tokenizer.save_pretrained(GRPO_LORA_DIR)
print("saved GRPO LoRA to", GRPO_LORA_DIR)
_drive_mirror(Path(GRPO_LORA_DIR))
_drive_mirror(DASHBOARD_CSV)

# %% [markdown]
# # Phase D — 4-baseline eval + plots

# %%
# Back to inference mode + run eval.py from the repo. We pass the two model paths so the
# `base` and `sft` + `trained` policies are all evaluated alongside `random` and `null`.
FastLanguageModel.for_inference(model)
os.environ["PYTHONPATH"] = f"{REPO_DIR / 'src'}:{os.environ.get('PYTHONPATH', '')}"

# Simplest: shell out.
subprocess.run(
    [
        "python", str(REPO_DIR / "scripts" / "eval.py"),
        "--baselines", "random", "null", "base", "sft", "trained",
        "--tasks", "easy_ria_late", "medium_jay_standup", "hard_dinner_sushi", "complex_multi_objective_dinner",
        "--seeds", "20",
        "--base-model", BASE_MODEL,
        "--sft-model", SFT_LORA_DIR,
        "--trained-model", GRPO_LORA_DIR,
    ],
    cwd=str(REPO_DIR), check=True,
)

# %%
from IPython.display import Image
Image(str(REPO_DIR / "data" / "plots" / "staircase.png"))

# %% [markdown]
# # Phase E — Push artifacts
#
# Commit the produced PNGs + LoRA back into the repo so judges can pull a full submission.

# %%
# ! cp /content/models/grpo_lora/adapter_model.safetensors $REPO_DIR/models/grpo_lora/
# ! cd $REPO_DIR && git add data/plots data/eval data/trajectories models && \
#   git -c user.email='hackathon@lakers' -c user.name='lakers' commit -m 'training run' && \
#   git push
