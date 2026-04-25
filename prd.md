# PhonePilot — Product Requirements Document

**Team:** LAKERS (Vivek Anand Singh, Vinay Kumar Chopra, Pranav Aggarwal)
**Event:** Meta PyTorch × OpenEnv Hackathon — Grand Finale, Scaler School of Technology, Bangalore
**Primary Theme:** 3.2 — Personalized Tasks
**Secondary Themes:** 2 (Long-Horizon Planning), 1 (Multi-Agent at inference)
**Document version:** v1.5
**Status:** Ready for build

---

## 1. TL;DR

PhonePilot is a simulated smartphone OS environment where a small LLM is trained via RL (SFT + GRPO) to act as a personal assistant. The agent completes real-world personal-assistant tasks — reaching people on the right channel, coordinating group plans, comparing prices across food delivery apps — by orchestrating a suite of simulated tools (call, WhatsApp, SMS, email, Calendar, Zomato, Swiggy, Maps, web search). Outcomes are stochastic (people don't always answer) so the agent must plan, adapt, and recover without spamming or lying about what it did.

The pitch: *"We trained a small LLM to be a believable personal assistant on a phone — it knows who to call, how to wait, when to escalate channels, and never claims it did something it didn't."*

Why this wins the hackathon: it's a literal word-for-word match to Theme 3.2's example environments, it's a live commercial product category (OpenAI Operator, Apple Intelligence, Rabbit R1), and the demo is visceral enough that a non-technical judge understands the before/after in 15 seconds.

---

## 2. Problem Statement

Every major AI lab is chasing "agents that act on your phone": OpenAI Operator, Anthropic Computer Use, Apple Intelligence, Rabbit R1's LAM, Google Astra. These products all solve variations of one problem: given a high-level human goal ("get Jay on the 3pm call," "book dinner for 4 tonight"), the agent needs to orchestrate multiple tools, handle stochastic outcomes, and recover from failure without hallucinating success.

Current LLMs do this poorly. They spam when a contact doesn't reply, they lie about completing tasks they haven't, they use the wrong channel for the urgency, they fail to read context before acting. These are not problems you fix with a bigger model — they're problems you fix with a reward signal that shapes the right behaviors, which means you need an environment to train in.

PhonePilot is that environment.

---

## 3. Hackathon Alignment

### 3.1 Theme match (Theme 3.2 — Personalized Tasks)

The Themes document lists example environments for 3.2: *"Executive Assistant Meeting Planner, Dinner and drive planning, email and message replying, shopping, etc."* PhonePilot implements the first three directly. Dinner planning is the Hard task. Meeting planning is the Medium task. Message replying is the Easy task. This mapping is explicit enough that a judge reading the PRD will immediately confirm theme fit.

### 3.2 Secondary theme match

**Theme 2 (Long-Horizon Planning):** Hard tasks require 15+ steps with multiple failure recovery points.
**Theme 1 (Multi-Agent Interactions):** Each contact is a simulated agent with its own state (availability, responsiveness, annoyance threshold). Only the assistant policy is trained, but the env has multi-agent structure at inference time.

### 3.3 Rubric alignment

| Rubric category | Weight | How PhonePilot scores |
|---|---|---|
| Environment Innovation | 40% | Novel for OpenEnv; mobile-OS-as-gym is underexplored; live commercial category |
| Storytelling & Presentation | 30% | Demo is visceral ("watch my phone do this"); non-technical-judge friendly |
| Showing Improvement in Rewards | 20% | Protected via four-baseline comparison + capability-curve dashboard (see §8) |
| Reward & Training Pipeline | 10% | Sub-goal-decomposed reward, curriculum GRPO, standard single-policy training |

### 3.4 Minimum submission requirements (from hackathon docs)

- [x] Uses OpenEnv (latest release) — FastAPI-based server
- [x] Training script via Unsloth or HF TRL, as a Colab notebook
- [x] Evidence of training: loss + reward plots committed to repo
- [x] Mini-blog on Hugging Face or <2-min YouTube video
- [x] Environment hosted on Hugging Face Space
- [x] README with all links and results
- [x] 3+ tasks with graders, scores in [0.0, 1.0]

---

## 4. Environment Specification

### 4.1 Core concept

A stepwise simulated phone. Each step the agent sees the current phone state and issues one tool call. The env advances simulated time, resolves the action stochastically where applicable, and returns a new observation. Episode ends when the agent calls `end_task()` or a time budget expires.

### 4.2 Action space (final v1.5 list)

**Communication (8 tools):**
- `call(contact)` — initiates a voice call; stochastic pickup
- `whatsapp_call(contact)` — WhatsApp voice call; stochastic
- `hang_up()` — ends active call
- `send_whatsapp(contact, text)` — WhatsApp text
- `send_sms(contact, text)` — SMS
- `send_email(contact, subject, body)` — email
- `read_messages(contact?, channel?)` — read conversation
- `read_notifications()` — check inbox

**Calendar (3 tools):**
- `calendar_view(date_range)` — list events
- `calendar_add(title, time, duration, invitees)` — create event
- `calendar_reschedule(event_id, new_time)` — move event

**Food apps — Zomato + Swiggy (6 tools, mirrored APIs):**
- `zomato_search(query, filters?)` / `swiggy_search(query, filters?)` — find restaurants
- `zomato_open(restaurant_id)` / `swiggy_open(restaurant_id)` — view menu + prices
- `zomato_order(restaurant_id, items, delivery_time)` / `swiggy_order(...)` — place order

**Maps (2 tools):**
- `maps_search(location_name)` — find locations near user
- `maps_travel_time(origin, destination)` — distance + travel duration

**Utility (4 tools):**
- `web_search(query)` — stubbed; returns canned results from dictionary
- `wait(minutes)` — advance simulated time while waiting for reply
- `end_task(success_claim: bool, summary: str)` — declare task complete
- `think(reasoning)` — internal chain-of-thought, no env effect

**Total: 23 tool signatures.** All tool calls are typed via Pydantic; malformed calls return a descriptive parsing error without consuming a step.

### 4.3 Observation space

What the agent sees at each step:
- `user_goal` (persistent across episode)
- `current_time` (simulated minutes since episode start)
- `time_budget_remaining`
- `recent_actions` (last 5 action → outcome pairs)
- `active_call_state` (if any)
- `open_app_view` (if an app is currently "open")
- `notifications` (new incoming messages/events since last step)
- `conversation_summaries` (last message per active contact-channel pair)

Explicitly **not** shown: contact responsiveness profiles, hidden difficulty tags.

### 4.4 State (internal, not fully observable)

- Current simulated time
- Full message history per (contact, channel)
- Per-contact annoyance level (increments with repeated contact without response)
- App states (calendar events, pending orders)
- Call state machine
- Episode termination flag

### 4.5 Contact simulation model

Five to ten contacts, each with a hidden profile:

```
Jay: {
  call_pickup_prob_work_hours: 0.3,       # low - busy at work
  call_pickup_prob_after_hours: 0.85,
  whatsapp_reply_median_mins: 4,
  sms_reply_median_mins: 30,
  email_reply_median_hours: 6,
  preferred_channel: "whatsapp",
  annoyance_threshold: 3  # after N unanswered msgs, response prob degrades
}
Mom: {
  call_pickup_prob_work_hours: 0.9,
  ...
}
```

Replies are generated by a frozen LLM (Claude API in dev, or a local model in production) primed with a persona snippet. The persona determines tone and content; the profile determines timing and likelihood.

---

## 5. Task Design

Four difficulty tiers. Minimum submission requires 3; we ship all four so Complex can be the Day-2 demo highlight.

### 5.1 Easy — "Quick reach"

**Prompt:** "Let Ria know I'll be 10 minutes late to our 4pm meeting."

**Success:** Ria acknowledges receipt via any channel within 5 simulated minutes.

**Expected base model success rate:** ~55%
**Target post-training:** 85%+

**Graded sub-goals:**
- Reached Ria via any appropriate channel (0.3)
- Message actually conveyed the delay + time (format check: contains "late" + "4pm" or similar) (0.2)
- Ria acknowledged (0.5)

### 5.2 Medium — "Urgent coordination"

**Prompt:** "Get Jay to join the 3pm standup call. It's urgent."

**Success:** Jay joins standup before 3:10pm simulated time.

**Expected base:** ~25%
**Target post-training:** 65%+

**Graded sub-goals:**
- Tried an appropriate first channel (call or WhatsApp call during work) (0.15)
- Waited before escalating (didn't immediately spam) (0.10)
- Escalated to a fallback channel when first failed (0.15)
- Sent a clear message conveying urgency + time (0.15)
- Jay joined in time (0.45)

### 5.3 Hard — "Dinner coordination"

**Prompt:** "Dinner tonight for me, Jay, Ria, and Mira. Jay was raving about a new sushi place last week — set that up. Make sure all three are in."

**Success:** Sushi restaurant booked, all three confirmed attending.

**Expected base:** ~8%
**Target post-training:** 30%+

**Graded sub-goals:**
- Read prior messages to find the sushi place Jay mentioned (0.15)
- Verified place exists via Zomato/Swiggy/web search (0.10)
- Checked everyone's calendar or asked availability (0.15)
- Handled at least one scheduling friction (someone busy, proposed alternative) (0.15)
- Booked restaurant (0.15)
- Received confirmation from all three contacts (0.30)

### 5.4 Complex — "Multi-objective coordination" (Day-2 demo piece)

**Prompt:** "Book dinner tonight for me + Jay + Ria + Mira. Jay is vegetarian. Ria has a 7pm call. Mira lives 15km from the rest of us. Keep it under ₹900/person including delivery. Get explicit confirmation from all three."

**Success:** Constraint-satisfying reservation with all confirmations.

**Expected base:** <5%
**Target post-training:** 15%+

**Graded sub-goals:**
- Filtered for vegetarian options (0.10)
- Checked Maps for location central enough for Mira (0.15)
- Checked Calendar for Ria's 7pm conflict, booked earlier or later (0.15)
- Used both Zomato AND Swiggy to compare prices (0.15)
- Stayed within ₹900/person budget (0.10)
- All three confirmed (0.35)

This task is deliberately hard — it exists to show the trained model handling complexity the base model can't touch, which is your Day-2 hero demo.

---

## 6. Reward Function

### 6.1 Design principles

Per the FAQ's anti-hacking guidance: multiple independent components, no single dominant term, every term is deterministic (no LLM-as-judge in the reward). Rewards sum to a scalar per step or per episode, then normalized to [-1, +1] for training stability.

### 6.2 Component 1 — Goal achievement (dense sub-goal scoring)

This is tactic 1 that you locked in. Instead of `reward = 1 if task_done else 0`, each task is decomposed into sub-goals (see §5). Sub-goal rewards fire when the env observes them achieved, not only at episode end. Weights per task are already enumerated above; they sum to 1.0 per task.

This is the single biggest unlock for training convergence — it turns a long sparse-reward task into a dense-reward task where the agent gets feedback at every meaningful progress milestone.

### 6.3 Component 2 — Truthfulness (the critical anti-hack)

When the agent calls `end_task(success_claim=True, summary=...)`:
- If `success_claim` contradicts the env's internal success evaluation: **−1.0** (large penalty)
- If `summary` references an action that never occurred in `action_history`: **−0.5**

This is the single most important reward term. Without it, RL reliably discovers the policy of lying about completion, which is exactly the failure mode the FAQ warns about. With it, the model learns that claiming success must match reality.

### 6.4 Component 3 — Efficiency

Small per-action cost: **−0.02** per tool call. Discourages excessive actions but isn't so steep that the agent skips necessary steps (e.g., checking calendar before scheduling).

### 6.5 Component 4 — Appropriateness

- Spamming penalty: `-0.1` per message sent to a contact who has unread messages already from the agent in this episode (encourages waiting before following up).
- Time-of-day penalty: `-0.15` for non-urgent contact at inappropriate hours (e.g., WhatsApp-calling at 2am for a non-time-critical task).
- Wrong-channel escalation: `-0.05` for escalating channel (e.g., email → SMS) faster than the previous channel's typical reply window.

### 6.6 Component 5 — Format validity

After SFT warmup, this is nearly free. `-0.05` for a malformed tool call that couldn't be parsed. Terminates episode after 3 consecutive format errors (safety cap).

### 6.7 Total reward formula (per episode, for logging)

```
R_total = R_goal + R_truthfulness + R_efficiency + R_appropriateness + R_format
```

For training, rewards are assigned per-step where possible (sub-goals fire when observed; format errors fire immediately) and end-of-episode where not (truthfulness only resolves at `end_task`).

---

## 7. Training Pipeline

### 7.1 Phase 1 — Synthetic trajectory generation (pre-onsite)

**Goal:** 200–500 successful task trajectories for SFT.

**Method:** Run Claude API (via LATM-style tool-user framing) against the deployed env on sampled tasks. For each trajectory, log `(observation, action, reward)` sequences. Keep:
- All successful episodes
- Partially successful episodes with interesting recovery behavior
- A small set of deliberately-generated negative examples for the truthfulness signal (episodes where the agent lied and got the large penalty) — for contrast

**Output:** a JSONL file of ~300 episodes, each averaging ~15 steps.

**Time:** ~4–6 hours (parallelizable via API).

### 7.2 Phase 2 — SFT warmup (onsite, Day 1 morning)

**Goal:** Teach the small model the tool-call format and reasonable initial behavior.

**Method:** Standard SFT via Unsloth on the synthetic trajectories. 1–2 epochs, LoRA rank 16, learning rate 2e-5. Target: model outputs schema-valid tool calls 95%+ of the time.

**Notebook to fork:** Unsloth Qwen2.5-3B fine-tuning notebook (linked from OpenEnv hackathon resources).

**Time:** 30–60 minutes on a single A100 equivalent.

### 7.3 Phase 3 — GRPO training with curriculum (onsite, Day 1 afternoon → Day 2 morning)

**Goal:** Improve actual task performance beyond SFT level.

**Method:** GRPO via Unsloth. Rollout size 4–8 per prompt. Curriculum:
- **Steps 0–80** — Easy tasks only. Expect reward climbing from ~0.3 to ~0.7.
- **Steps 80–160** — Easy + Medium mixed. Reward dips, then recovers.
- **Steps 160–300** — All three tiers. Second dip + recovery.

Log every 5 steps:
- Total reward (smoothed moving average)
- Per-component reward breakdown (5 lines on one plot)
- Task success rate per difficulty tier
- Action-validity rate
- Mean episode length

Sample rollouts every 25 steps for manual inspection. If you see the model exploiting any reward component, pause, adjust weights, resume.

**Time:** 4–8 hours on an A100.

### 7.4 Phase 4 — Evaluation (onsite, Day 2 afternoon)

Run four models on a held-out test bank of 50 task variants (15 Easy, 15 Medium, 15 Hard, 5 Complex):
- Random policy (control)
- Base model zero-shot (no fine-tuning)
- SFT-only model
- Full trained (SFT + GRPO)

Produce:
- Success-rate bar chart (4 models × 4 difficulty tiers)
- Reward curve from training
- Capability curve dashboard (see §8.2)
- Reliability diagram if tracking confidence
- Example trajectory video (see §8.3)

---

## 8. "Showing Improvement" Strategy — the 20% score

This is the category where PhonePilot is weakest if we don't plan for it. Here's the plan.

### 8.1 Four-baseline staircase

Four bars per task tier, all in one chart. Expected shape: random ≪ base ≪ SFT < trained. The *staircase* is the evidence of learning; each gap is a different kind of improvement (format, behavior, task-solving).

### 8.2 Capability curve dashboard

Six metrics, each tracked every 10 training steps, plotted on one dashboard:

1. **Channel-ladder appropriateness** — did agent escalate in the right order?
2. **Spam rate** — average messages per contact before waiting. Should decrease.
3. **Time-appropriate behavior** — fraction of non-urgent actions at reasonable hours.
4. **Truthfulness** — `end_task(success=True)` was actually true.
5. **Efficiency** — mean actions per successful episode.
6. **Recovery rate** — when first channel failed, did agent successfully adapt?

Each is an independent learning curve. Even if main reward is noisy on a given day, 3–4 of these will show clean improvement.

### 8.3 Qualitative before/after (the viral demo clip)

60-second side-by-side video on the same Medium or Hard task:
- Left: base model. Spams SMS three times. Calls at 11pm. Declares task complete. Jay never responded.
- Right: trained model. Tries call. No answer. Sends crisp WhatsApp ("Jay — quick one, 3pm standup, can you hop on?"). Jay responds. Task complete.

This clip goes in the README, the pitch, and the submission video. It wins Storytelling and makes Improvement visceral.

### 8.4 Capability probes

10 small standalone probes run every 20 training steps, e.g. "send a one-line hi to Ria," "find a pizza place in Koramangala," "check what's on my calendar tomorrow." Each deterministic pass/fail. Plot: "probes passed out of 10, over training." Clean monotonic curve.

---

## 9. Technology Stack

### 9.1 Why Python is required

The entire hackathon stack is Python-native:

- **OpenEnv Core** — Python + FastAPI. The framework is defined in Python; environments must be Python classes.
- **TRL** — Python. The training library is Hugging Face Transformers-based.
- **Unsloth** — Python / Jupyter notebooks. All example recipes in the hackathon FAQ are Colab notebooks.
- **Pre-submission validator** — Shell script that runs `pip install openenv-core` and `openenv validate` on the submitted repo.
- **Hugging Face Spaces** — default to Python + FastAPI for ML spaces.

There is no submission path that avoids Python for the env + training. Attempting to go through TS/JS would require reimplementing OpenEnv's interface, which is out of scope for a hackathon and will cost more time than it saves.

### 9.2 Stack components

| Component | Tool |
|---|---|
| Environment server | Python 3.10+, FastAPI, Pydantic |
| OpenEnv compliance | openenv-core package |
| LLM inference during SFT data gen | Anthropic API (Claude) or OpenAI API |
| SFT training | Unsloth + HF Transformers |
| GRPO training | Unsloth + TRL |
| Metrics logging | Weights & Biases (free tier) |
| Deployment | Hugging Face Spaces (Dockerfile) |
| Demo video | OBS / screen recording |
| Optional demo frontend | React/Next (if desired, not required) |

### 9.3 Where other languages can fit

If your team wants to build a polished demo UI instead of just showing terminal output: write a small React app that consumes the env's REST API and visualizes the agent's actions as a phone-screen animation. This is purely for the submission video / pitch — the grader doesn't care, but it makes the demo more arresting. This part can be TS/JS and owned by whoever on the team prefers web.

---

## 10. Team Roles

### 10.1 Vivek (lead) — Environment Owner
- OpenEnv scaffold (action types, observation types, reward hooks)
- Contact simulator (profiles, response generation via frozen LLM)
- App stubs (Calendar, Zomato, Swiggy, Maps, web search)
- State management + time advancement
- FastAPI server + openenv.yaml + Dockerfile
- HF Space deployment
- README (technical section)

### 10.2 Vinay — Tasks + Rewards Owner
- The 4 task graders (deterministic success evaluators per task)
- Reward function implementation (all 5 components)
- Synthetic trajectory generation script (Claude API → JSONL)
- Eval harness (runs the 4 baselines, produces the charts)
- Capability probes

### 10.3 Pranav — Training Owner
- **Start today:** get the Unsloth Qwen2.5-3B GRPO notebook running on a dummy env. This is the skill that takes longest to learn, and it's the critical path. Don't wait for the real env to be ready.
- SFT pipeline from the synthetic trajectories
- GRPO training with the curriculum schedule
- WandB logging + plot generation
- Model checkpoint management

Late-stage all three merge into: demo video recording, README polish, pitch prep.

---

## 11. Timeline

### 11.1 Pre-onsite (depends on actual gap — will tighten once confirmed)

**Days -N through -3 (Vivek + Vinay in parallel; Pranav on training prep):**
- Spec locked ✓ (this document)
- OpenEnv skeleton committed to repo
- Contact simulator working with 5 contacts
- Calendar + Zomato + Maps stubs implemented (Swiggy is mirror of Zomato)
- Easy + Medium tasks implemented with graders
- Reward function v1 implemented
- Unit tests for reward function (especially truthfulness)
- Deployed to HF Space

**Days -2 to -1:**
- Hard + Complex tasks implemented
- Synthetic trajectory generation run, ~300 trajectories saved
- Baseline metrics logged for the base model (no training)
- Pranav has Unsloth GRPO running on a toy env
- Demo video scaffolded (first side-by-side attempt)

### 11.2 Onsite Day 1

- Morning: SFT warmup run on real trajectories. Check format-validity rate.
- Afternoon: First GRPO run on Easy-only curriculum. Target visible reward climb by end of session.
- Evening mentor round: get feedback. Debug reward hacking if any observed.

### 11.3 Onsite Day 2

- Morning: Second GRPO run with full curriculum. Let it cook during breakfast + early session.
- Midday: Eval run. Generate all four baselines. Produce charts.
- Afternoon: Demo video recording. README finalization.
- **5pm: submission deadline.**

---

## 12. Scope Management

### 12.1 In scope (v1.5)

- 23 tools across 4 app categories + messaging
- 4 tasks (Easy / Medium / Hard / Complex)
- 5-component reward function with sub-goal decomposition
- SFT + curriculum GRPO training
- 4-baseline comparison
- 6-metric capability dashboard
- 60-second before/after demo clip
- HF Space deployment + README

### 12.2 Out of scope

- Voice I/O (speech-to-text, text-to-speech)
- Real browser/app integrations (everything is stubbed)
- Multimodal / screen parsing
- Multi-agent RL training (other contacts are frozen; only assistant is trained)
- More than one round of RL training with different hyperparameters
- Fancy custom evaluation UI (beyond default HF Space)
- Group chats, voice notes, media messages
- Amazon / Flipkart / shopping apps (Day-2 stretch only)

### 12.3 Cut order under pressure

If Day 1 evening metrics show problems, cut in this order:

1. **First cut:** Swiggy. Zomato-only. Complex task loses its "compare food apps" component but otherwise survives.
2. **Second cut:** Complex task. Ship Easy + Medium + Hard only. Still satisfies 3-task minimum.
3. **Third cut:** Maps app. Hard task loses the location-reasoning component; becomes a pure messaging task.
4. **Fourth cut:** Appropriateness and efficiency reward components. Keep only goal + truthfulness + format.
5. **Never cut:** SFT warmup, truthfulness penalty, at least one GRPO run with logged before/after metrics, 60-second demo clip.

---

## 13. Submission Checklist

From the hackathon docs:

- [ ] Hugging Face Space URL — env deploys and responds to reset()
- [ ] Colab Notebook link — Unsloth training script, re-runnable
- [ ] Code repository link — GitHub with README, Dockerfile, openenv.yaml
- [ ] YouTube video OR HF blog post URL — 2-minute explainer
- [ ] All URLs included in README
- [ ] Reward curves and loss plots committed as PNGs in repo
- [ ] Four-baseline comparison chart committed
- [ ] Example trajectory video committed or linked

---

## 14. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GRPO doesn't converge in onsite window | Medium | High | Curriculum schedule starts with Easy only; sub-goal reward gives dense signal; fall back to SFT+Easy only submission |
| Action space too large for small model | Medium | Medium | Start with Gemma 3 1B; upgrade to Qwen 2.5 3B only if 1B converges |
| Reward hacking on truthfulness | Low | High | Truthfulness is the biggest penalty; extensively unit-tested |
| HF Space deployment fails onsite | Low | High | Deploy pre-onsite; test with external curl requests |
| Team member unavailable Day 2 | Low | High | Roles are independent enough that any one can be absorbed by the other two |
| Demo video fails to compile | Low | Medium | Start video work Day 1 evening, not Day 2 afternoon |

---

## 15. Appendix

### A. Hackathon rubric reference

From "Apr '26 OpenEnv Hackathon Themes & Judging Criteria":

- Environment Innovation — 40%
- Storytelling & Presentation — 30%
- Showing Improvement in Rewards — 20%
- Reward & Training Pipeline — 10%

### B. Key references

- OpenEnv Core: https://github.com/meta-pytorch/OpenEnv
- Unsloth notebooks: linked from the hackathon FAQ
- TRL GRPO docs: Hugging Face Transformers Reinforcement Learning library
- OpenAI Operator: live commercial product in the agent-for-phone category
- Anthropic Computer Use: live commercial product in adjacent category

### C. Design decisions log

- **Why phone-OS simulation vs. computer-use simulation:** Phone context is more consumer-relatable for demo; smaller action space than full browser; native fit to Theme 3.2.
- **Why SFT + GRPO vs. pure GRPO:** With a 23-tool action space, pure GRPO from base model wastes most rollouts on malformed tool calls; SFT warmup on synthetic trajectories fixes format quickly.
- **Why Zomato + Swiggy vs. single food app:** Enables cross-app price-comparison behavior (compelling demo), shares same API schema (low incremental cost), teaches the model that app categories have abstractions.
- **Why only food apps, not shopping apps too:** Shopping introduces a second task family with different reward shape; training distribution gets hard to balance; insufficient incremental demo value vs. cost.

---

*End of PRD.*