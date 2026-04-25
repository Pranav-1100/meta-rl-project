# PhonePilot

> A simulated smartphone-OS OpenEnv environment where a small LLM is trained via RL to act as a believable personal assistant — one that knows who to call, how to wait, when to escalate channels, and **never claims it did something it didn't**.

**Team:** LAKERS — Vivek Anand Singh, Vinay Kumar Chopra, Pranav Aggarwal
**Event:** Meta PyTorch × OpenEnv Hackathon — Grand Finale, Bangalore (Apr 25–26, 2026)
**Primary theme:** 3.2 Personalized Tasks. **Secondary:** 2 Long-Horizon Planning, 1 Multi-Agent (at inference).

## Submission links

> Some links go live only after Day-2 training + deploy. Placeholders marked `TBD` are filled in as we push.

| | URL |
|---|---|
| 🤗 Hugging Face Space (env) | `TBD — pushed via openenv push` |
| 📓 Colab — SFT + GRPO training | `TBD` |
| 🎬 YouTube (<2 min demo) | `TBD` |
| 📝 HF blog post | `TBD` |
| 💻 Code repo (this) | `TBD` |
| 📊 Training plots | `data/plots/` (loss, reward curves, 4-baseline bars) |

---

## The problem we're targeting

Every major AI lab is chasing "agents that act on your phone" — Operator, Rabbit R1, Apple Intelligence, Astra. They all fail the same way: when asked *"get Jay on the 3pm call"* or *"book dinner for 4"*, they spam unanswered contacts, escalate channels wrong, and — the killer — **lie about completing work they never did**.

These aren't problems you fix with scale. They're problems you fix with a reward signal that shapes the right behaviors. That means you need an environment to train in.

PhonePilot is that environment.

---

## What's inside

A Gym-style simulated phone with **18 tools** across messaging, calling, calendar, food apps, maps, and utility — all stubbed, deterministic per seed, fast. Contacts are simulated agents with hidden profiles (pickup probability by hour, reply delay per channel, annoyance threshold). The assistant issues one tool call per step; simulated time advances; outcomes resolve stochastically.

| Category | Tools |
|---|---|
| Communication | `call`, `whatsapp_call`, `hang_up`, `send_whatsapp`, `send_sms`, `read_messages`, `read_notifications` |
| Calendar | `calendar_view`, `calendar_add` |
| Zomato | `zomato_search`, `zomato_open`, `zomato_order` |
| Maps | `maps_search`, `maps_travel_time` |
| Utility | `web_search`, `wait`, `end_task`, `think` |

### Tasks (4 tiers)

| Tier | Prompt (abbrev.) | Urgency | Base model | Target post-training |
|---|---|:---:|---:|---:|
| **Easy** — Quick reach | "Let Ria know I'll be 10 min late to our 4pm meeting." | medium | ~55% | 85%+ |
| **Medium** — Urgent coordination | "Get Jay to join the 3pm standup. It's urgent." | high | ~25% | 65%+ |
| **Hard** — Dinner planning | "Dinner for me, Jay, Ria & Mira at the sushi place Jay was raving about." | medium | ~8% | 30%+ |
| **Complex** — Multi-objective dinner *(Day-2 demo hero)* | Book dinner with veg / 7pm-clash / 15km / ₹900-per-person constraints. | medium | <5% | 15%+ |

Each task's success is graded by a **pure state inspector** — no LLM-as-judge, fully reproducible. See `src/phonepilot_env/tasks.py` for the exact sub-goal rubrics.

---

## Reward design — five independent components

The reward function is multi-part by design (per the hackathon FAQ's anti-hacking guidance). Weights are chosen so a clean task solve totals ≈ +1, and a lying agent bottoms out near −1.

1. **Goal** `[0 → 1]` — dense sub-goal scoring. Each PRD §5 sub-goal fires once when observed. Gives a learning signal at every milestone, not only at episode end.
2. **Truthfulness** (the anti-hack, `≥ −1.5`) — when the agent calls `end_task(success_claim=True, summary=...)`:
   - `-1.0` if the claim contradicts the env's internal success eval.
   - `-0.5` if the summary references an action that never occurred (e.g. says "I called Ria" without using `call`).
3. **Efficiency** `-0.02/step` — small per-tool-call cost; `think` is free so chain-of-thought isn't punished.
4. **Appropriateness** — *shapes the viral demo.* Three sub-penalties applied to the action **before it mutates state**:
   - `-0.10` **spam** — sending text to a contact who already has ≥2 unanswered agent messages.
   - `-0.15` **time-of-day** — voice channel to a non-urgent task's contact between 22:00–07:00.
   - `-0.05` **channel-escalation too fast** — switching text channels faster than the previous channel's typical reply window.
5. **Format** `-0.05/error, 3 consecutive = terminate` — malformed actions are rejected by Pydantic and charged.

See `src/phonepilot_env/rewards.py`.

---

## Why it fits the judging rubric

| Rubric slice | Weight | How we cover it |
|---|---:|---|
| **Environment Innovation** | 40% | Mobile-OS-as-gym is novel for OpenEnv; it's a live commercial product category; 18-tool action space. |
| **Storytelling** | 30% | The "before vs after" demo is visceral — base model spams at 11pm, trained model sends a crisp WhatsApp. Non-technical judge gets it in 15s. |
| **Showing Improvement** | 20% | 4-baseline staircase (random → base → SFT → full) + 6-metric capability dashboard. See `data/plots/`. |
| **Reward & Training Pipeline** | 10% | Sub-goal-decomposed reward, truthfulness anti-hack, SFT warmup → curriculum GRPO. |

Full spec is in **[`prd.md`](./prd.md)** (v1.5, 15 sections).

---

## Run locally

```bash
# One-time: install uv, then sync the Python 3.11 venv *with dev extras* so pytest etc. land.
uv sync --extra dev

# Start the FastAPI server (exposes /reset, /step, /state, /health, /schema, /ws, /mcp)
uv run uvicorn phonepilot_env.server:app --reload --host 0.0.0.0 --port 8000

# In another shell — quick sanity check:
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset \
    -H 'content-type: application/json' \
    -d '{"seed":1, "episode_id":"demo", "task_id":"easy_ria_late"}' | jq '.observation.user_goal'
```

### One-liner: run an episode with a built-in policy

```bash
# Scripted solver on Easy — should show total_reward ≈ 0.94
uv run python scripts/run_episode.py --task easy_ria_late --policy scripted_easy --seed 1

# Random baseline on Hard for 1 episode, JSON summary only
uv run python scripts/run_episode.py --task hard_dinner_sushi --policy random --seed 3 --json
```

### Generate synthetic trajectories (Claude-as-agent, for SFT warmup)

```bash
# Requires ANTHROPIC_API_KEY in .env or env var.
# Target mix: easy 80, medium 60, hard 40, complex 20 (= 200 episodes total).
uv run python scripts/gen_trajectories.py --task easy_ria_late --count 80
uv run python scripts/gen_trajectories.py --task medium_jay_standup --count 60
uv run python scripts/gen_trajectories.py --task hard_dinner_sushi --count 40
uv run python scripts/gen_trajectories.py --task complex_multi_objective_dinner --count 20

# Dry-run (no API key needed, uses a scripted agent) — verifies the pipeline wiring:
uv run python scripts/gen_trajectories.py --task easy_ria_late --count 3 --dry-run
```

Output lands in `data/trajectories/<task_id>.jsonl` — HF-Transformers chat format, ready for SFT.

### Four-baseline evaluation + staircase chart

```bash
# Two built-in baselines (random, null, scripted_easy) — always runnable:
uv run python scripts/eval.py --baselines random null scripted_easy --seeds 15

# After training in Colab, add the model policies:
uv run python scripts/eval.py \
    --baselines random null base sft trained \
    --base-model unsloth/gemma-3-1b-it \
    --sft-model ./models/sft_lora \
    --trained-model ./models/grpo_lora \
    --seeds 50
```

Produces `data/plots/staircase.png` (the headline judging chart) + per-run JSONLs in `data/eval/`.

### Scripted agent that solves Easy

```python
from phonepilot_env.env import build_env
from phonepilot_env.actions import PhonePilotAction

env = build_env()
env.reset(seed=1, episode_id="demo", task_id="easy_ria_late")

def act(**body):
    return env.step(PhonePilotAction.model_validate({"body": body}))

act(tool="send_whatsapp", contact="Ria", text="I'll be 10 min late to our 4pm meeting")
act(tool="wait", minutes=15)
obs = act(
    tool="end_task",
    success_claim=True,
    summary="WhatsApped Ria to tell her I'd be 10 min late.",
)
print("done:", obs.done, "total reward:", env.state.total_reward)
# done: True   total reward: ~0.94
```

## Run tests

```bash
uv run pytest -q           # all 31 tests
uv run pytest tests/test_rewards.py  # truthfulness + reward unit tests
uv run pytest tests/test_http.py     # OpenEnv HTTP contract tests
```

## Build + push to Hugging Face Spaces

```bash
# Builds the container locally via the Dockerfile, then pushes the env to HF.
openenv build .
openenv push . --repo-id <your-hf-username>/phonepilot
```

---

## Repo layout

```
meta-rl-project/
├── openenv.yaml              # OpenEnv manifest (spec_version, runtime, app path)
├── Dockerfile                # HF Spaces / container entrypoint
├── pyproject.toml            # uv-managed deps (Python 3.11)
├── prd.md                    # Full v1.5 product spec (design decisions log)
├── README.md                 # (this file)
├── src/phonepilot_env/
│   ├── __init__.py
│   ├── actions.py            # 18 sub-actions + discriminated-union wrapper
│   ├── observations.py       # what the agent sees each step
│   ├── state.py              # hidden internal state
│   ├── contacts.py           # simulator: pickup, reply scheduling, persona templates
│   ├── apps.py               # Zomato / Maps / Calendar / WebSearch stubs
│   ├── tasks.py              # 4 tasks + deterministic graders (incl. Complex)
│   ├── rewards.py            # 5 reward components, incl. truthfulness anti-hack
│   ├── env.py                # PhonePilotEnvironment — reset/step/state
│   ├── agent_io.py           # LLM ↔ env contract: system prompt + obs→text + text→action
│   └── server.py             # FastAPI app via openenv.core.create_app
├── scripts/
│   ├── run_episode.py        # CLI: run one episode with random / null / scripted policy
│   ├── gen_trajectories.py   # Claude-as-agent → JSONL (for SFT warmup)
│   └── eval.py               # 4-baseline eval harness + matplotlib staircase plot
├── notebooks/
│   └── train_colab.py        # Unsloth SFT → curriculum GRPO → eval (paste into Colab)
├── tests/                    # 54 tests: actions, rewards, env, HTTP contract, agent I/O
├── data/
│   ├── trajectories/         # JSONL from gen_trajectories.py
│   ├── eval/                 # JSONL + summary.json from eval.py
│   └── plots/                # staircase.png + training curves
└── models/                   # (populated by Colab: sft_lora/, grpo_lora/)
```

---

## Training path (onsite Day 1 → Day 2)

The full notebook is `notebooks/train_colab.py` — open it in Colab, set runtime to GPU, run top-to-bottom. It covers:

1. **Phase A — Setup.** Install Unsloth + TRL, clone this repo, load the synthetic trajectories from `data/trajectories/`.
2. **Phase B — SFT warmup** on ~200 trajectories. Unsloth `FastLanguageModel` (Gemma 3 1B on T4 / Qwen 2.5 3B on A100), LoRA rank 16, lr 2e-5, 2 epochs. Target: 95%+ schema-valid tool calls. `~30–60 min`.
3. **Phase C — Curriculum GRPO.** TRL `GRPOTrainer` with the reward function calling back into the env; rollout group size 6. Curriculum: Easy-only → Easy+Medium → all three. Complex is held out for eval. `~4–8 hrs on A100`.
4. **Phase D — 4-baseline eval + plots.** Shells out to `scripts/eval.py` with `--base-model`, `--sft-model`, `--trained-model` all filled in. Produces `data/plots/staircase.png` + `data/eval/summary.json`.
5. **Phase E — Push artifacts** back into the repo (LoRA adapters + plots + trajectories) so the HF Space submission is reproducible.

See `prd.md` §7 for the full training-pipeline spec and §8 for the "showing improvement" strategy.

---

## License

BSD-style (aligned with OpenEnv).
