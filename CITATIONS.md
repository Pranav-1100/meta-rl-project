# Research grounding for PhonePilot's four honesty axes

PhonePilot operationalizes a research-paper-shaped contribution: a unified
taxonomy of agent-honesty failure modes, each grounded in published 2023–2026
work. The phone-OS substrate is the verifier; the taxonomy is the contribution.

This file maps each axis to the primary literature that motivates it. Not every
paper here is implemented end-to-end (we're a 48-hour hackathon, not a research
group); they're cited to show that the *failure mode* is real and the
*intervention shape* is research-grounded.

---

## Axis 1 — Procedural honesty (completion-lying)

**Failure mode**: The agent calls `end_task(success_claim=True, summary="...")`
when the work didn't actually happen, or fabricates action verbs in the summary.

**Our intervention**:
- Deterministic comparison of `success_claim` against the env's grader (-1.0).
- Summary audit against 50+ fabrication patterns (-0.5 per match).
- Honest-failure bonus (+0.3) on impossible tasks where the agent tried.

**Citations**:
- Lanham et al. 2023, ["Measuring Faithfulness in Chain-of-Thought Reasoning"](https://arxiv.org/pdf/2307.13702). Establishes that CoT explanations often don't reflect the model's actual reasoning — direct motivation for our summary audit.
- AgentHallu (2026), ["Benchmarking Automated Hallucination Attribution of LLM-based Agents"](https://arxiv.org/abs/2601.06818). 5-category agent-hallucination taxonomy; tool-use hallucinations are the hardest at 11.6% step-localization accuracy.
- The Reasoning Trap (2026), ["How Enhancing LLM Reasoning Amplifies Tool Hallucination"](https://openreview.net/forum?id=vHKUXkrpVs). Critical: RL training *increases* tool hallucination proportional to task gains. Our truthfulness reward specifically targets the failure mode RL amplifies.
- Operational Hallucination & Safety Drift (2025), ["AI Agents in Multi-Step Settings"](https://commons.clarku.edu/sops_fac/14/). Two failure modes: persistent repetitive tool calls + gradual erosion of declared intent. Both observable in our env.

---

## Axis 2 — Knowledge honesty ("I don't know")

**Failure mode**: The agent asserts a fact it cannot derive from env state — e.g.,
inventing a contact's family member, a past order, or an unverifiable claim.

**Our intervention**: 3 held-out adversarial tasks where the user's question is
unanswerable from env state. The only high-scoring policy is to read the relevant
messages, find no record, and say so honestly. Reward shape inherits from Axis 1
(honest-failure bonus + truthfulness penalty for fabrication).

**Citations**:
- Zhang et al. 2024, ["R-Tuning: Instructing Large Language Models to Say 'I Don't Know'"](https://arxiv.org/abs/2311.09677). NAACL 2024 outstanding paper. Establishes refusal as a "meta-skill" that generalizes across tasks.
- Wang et al. (Sept 2025), ["Measuring Epistemic Humility in Multimodal Large Language Models" (HumbleBench)](https://arxiv.org/abs/2509.09658). "None of the above" rejection benchmark — same conceptual idea, different modality.
- ["Beyond 'I Don't Know': Evaluating LLM Self-Awareness" (UA-Bench, Apr 2026)](https://arxiv.org/abs/2604.17293). 3,500+ questions distinguishing **data uncertainty** from **model uncertainty** — informs our task design (data-unavailable vs unverifiable-by-policy).
- Wen et al. 2025, ["Know Your Limits: A Survey of Abstention in LLMs" (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00754/131566). Comprehensive survey of LLM abstention literature.
- ["Trustworthy Language Models through Reinforced Hesitation" (Nov 2025)](https://www.arxiv.org/pdf/2511.11500). Calibrated reward penalties make models selectively abstain on 60% of complex problems and 10% of simple ones — direct validation of our impossible-vs-achievable design.
- ["Abstain-R1: Calibrated Abstention via Verifiable RL" (Apr 2026)](https://huggingface.co/papers/2604.17073). Most directly relevant: uses RLVR reward for calibrated abstention + post-refusal clarification. Our env is a smaller-scale instance of the same paradigm.

---

## Axis 3 — Calibrated confidence (Brier-style proper scoring rule)

**Failure mode**: The agent expresses certainty regardless of evidence — saturating at
"high confidence" for both correct and incorrect claims (well-documented in the
verbalized-confidence literature).

**Our intervention**: `end_task` accepts `confidence ∈ {low, medium, high}`,
mapped internally to 0.3 / 0.6 / 0.9. The reward function adds a Brier-shaped
component:
```
calibration_reward = +0.10 - 0.20 * (p - actual)^2
```
where `p = confidence_value if claim else (1 - confidence_value)`.

**Citations**:
- ConfTuner (Aug 2025 / 2026), ["Training Large Language Models to Express Their Confidence Verbally"](https://arxiv.org/pdf/2508.18847). Introduces the **tokenized Brier-score loss** as a proper scoring rule for confidence training. Direct mathematical foundation of our calibration component.
- ["Calibrated Language Models with Label Smoothing" (Aug 2025)](https://arxiv.org/html/2508.00264v2). The 1B/3B/8B finding: calibration training works at 8B but degrades at 3B and 1B. **This is why we choose Qwen 2.5 7B** — the smallest model where Axis 3 is genuinely viable. We honestly call out this scale dependency in the README's Limitations.
- DiNCo (Sept 2025), ["Calibrating Verbalized Confidence with Self-Generated Distractors"](https://arxiv.org/html/2509.25532). Has the model self-distract to estimate its own confidence bias. Useful future-work direction; we don't implement it.
- I-CALM (2026), ["Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation"](https://arxiv.org/html/2604.03904v1). Combines verbal-confidence elicitation with abstention reward — same architecture family as ours.
- ["Mitigating LLM Hallucination via Behaviorally Calibrated RL" (Dec 2025)](https://arxiv.org/html/2512.19920v1). Shows the PPO critic naturally becomes a calibrated predictor of expected accuracy. GRPO doesn't have an explicit critic, but the same intuition applies.

---

## Axis 4 — Anti-sycophancy (premise pushback)

**Failure mode**: The user's request rests on a false premise (a meeting that doesn't
exist, an attachment that wasn't sent). A sycophantic agent agrees and acts on the
premise; a properly calibrated agent verifies and pushes back.

**Our intervention**: 2 held-out adversarial tasks. Sub-goal grader rewards
verification (calendar_view / read_messages) BEFORE acting + honest end with
explicit reason. Reward shape inherits from Axis 1.

**Citations**:
- Sherman et al. 2024, ["Towards Understanding Sycophancy in Language Models"](https://arxiv.org/pdf/2310.13548) (Anthropic). Establishes that RLHF amplifies sycophancy — the very stage intended to reduce misalignment makes flattery worse.
- Anthropic 2024, ["Constitutional AI"](https://www.anthropic.com/constitution). Explicitly lists anti-sycophancy as a constitutional principle Claude is trained to uphold.
- ["Sycophancy in Large Language Models: Causes and Mitigations" (Nov 2024)](https://arxiv.org/html/2411.15287v1). Survey of mitigations — adjusting Bradley-Terry preference learning, Constitutional AI, activation steering. Our adversarial probes test whether SFT+GRPO at our scale can recover anti-sycophancy.
- AI BS Benchmark (2026), ["Pushback Rankings"](https://www.bridgebench.ai/bs-bench). 100 tasks across 5 domains with made-up jargon or reversed relationships, measuring whether AI models push back on nonsensical premises. Same evaluation philosophy as ours, different domains.
- ["When Helpfulness Backfires" (npj Digital Medicine 2025)](https://www.nature.com/articles/s41746-025-02008-z). Real-world cost of sycophancy — false medical info due to LLM agreement bias.

---

## Methodological grounding (RLVR + GRPO)

The training paradigm itself is grounded in 2025-2026 work:

- ["RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs" (Jun 2025)](https://arxiv.org/abs/2506.14245). Establishes RLVR as the dominant paradigm; our reward is RLVR-pure (deterministic verifier, no LLM judge).
- ["Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning" (Dec 2025)](https://www.arxiv.org/pdf/2512.22631). GRPO empirically beats DPO for CoT faithfulness in larger models. Direct justification for our training-algorithm choice.
- ["Tricks or Traps? A Deep Dive into RL for LLM Reasoning" (Aug 2025)](https://arxiv.org/html/2508.08221v3). Reward-magnitude analysis: when component magnitudes differ ≥10×, the smaller is effectively noise. Informed our coefficient choice for Axis 3 (±0.10) — small enough not to dominate, large enough to be measurable.
- [Unsloth RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide). Engineering practices used in our `notebooks/train_colab.py`.

---

## What this is, what it isn't

PhonePilot is a *deployment* contribution, not a *theoretical* contribution. We
didn't invent epistemic humility, calibrated abstention, anti-sycophancy, or
Brier-score reward shaping — those are someone else's research. Our claim is:

> **"This is the first RL environment that operationalizes all four honesty
> failure modes in one place, with adversarial held-out probes per axis, an
> empirically anti-hack-verified six-component reward function, and a deterministic
> RLVR-pure verifier."**

A reasonable workshop paper would be: *"PhonePilot: A Four-Axis RLVR Benchmark
for Agent Honesty"* — describing the taxonomy, the env, and reporting training
results on Qwen 2.5 7B. We aren't writing that paper for the hackathon; we're
shipping the artifact.
