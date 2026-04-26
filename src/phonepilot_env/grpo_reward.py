"""GRPO rollout reward function — extracted from the training notebook for unit testing.

The TRL ``GRPOTrainer`` calls ``rollout_reward(prompts, completions, **kwargs)`` for
each batch of K generated completions. We unpack each completion into a
``PhonePilotAction`` (or assign the format-error floor on parse failure), step the env
once with the parsed action, and return the per-step reward.

This single-step rollout choice (vs full episode unroll inside the reward fn) is a
deliberate cost trade — full episodes would 10x training compute. The env's per-step
reward already includes:

  * the goal sub-goal increment for THIS step,
  * any format / efficiency / appropriateness penalties,
  * the truthfulness + calibration components when the action is ``end_task``.

So even a one-step rollout carries meaningful reward signal — just episode-truncated.

The constants live alongside :data:`FORMAT_FLOOR_REWARD` so they can be tuned without
edit-and-retrain.
"""

from __future__ import annotations

from typing import Any

from .actions import PhonePilotAction  # noqa: F401  (used by callers via the schema)
from .agent_io import AgentParseError, parse_completion_to_action
from .env import build_env


# Reward assigned to a completion that fails to parse as a PhonePilotAction. Lower than
# the worst legitimate per-step reward so format errors are unambiguously dispreferred,
# but not so negative that one bad rollout dominates the group-relative advantage.
FORMAT_FLOOR_REWARD: float = -0.5


def _shaped_format_bonus(completion: str) -> float:
    """Tiny partial-credit reward to BREAK the all-equal-rewards tie.

    GRPO needs reward variance within a group to compute advantages. If 100% of
    rollouts hit FORMAT_FLOOR_REWARD (parse failure), advantages are zero and the
    gradient vanishes. By giving graded credit for format adherence we ensure that
    "more JSON-like" completions get slightly higher reward, providing a signal for
    the model to climb.

    Bonus components (max +0.20 total, kept small relative to the -0.5 floor):
      * +0.05 if completion contains a fence (```)
      * +0.05 if it specifically contains the JSON fence (```json)
      * +0.05 if it contains both '{' and '}' (JSON-like braces)
      * +0.05 if it contains '"tool"' or '"body"' (PhonePilot schema hints)
    """
    bonus = 0.0
    if "```" in completion:
        bonus += 0.05
        if "```json" in completion:
            bonus += 0.05
    if "{" in completion and "}" in completion:
        bonus += 0.05
    if '"tool"' in completion or '"body"' in completion:
        bonus += 0.05
    return bonus


def rollout_reward(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """The GRPO reward function.

    Args:
        prompts: list of prompt strings (unused — TRL passes them but the env doesn't
            need them since seed + task_id determine the state).
        completions: list of model-generated completion strings (one per group-relative
            sample). Each is expected to contain a fenced or bare JSON tool call.
        kwargs: TRL passes through any extra columns from the training dataset.
            We rely on:
              * ``task_id`` — list[str], one per completion. Identifies which task to
                seed the env with.
              * ``seed``    — list[int], one per completion. Determines stochastic
                outcomes (call pickup, reply scheduling).

    Returns:
        list[float] of length ``len(completions)``. Each is either:
          * the env's ``obs.reward`` after stepping the parsed action, OR
          * :data:`FORMAT_FLOOR_REWARD` if the completion couldn't be parsed.
    """
    rewards: list[float] = []
    task_ids = kwargs.get("task_id", [None] * len(completions))
    seeds = kwargs.get("seed", [0] * len(completions))

    for completion, task_id, seed in zip(completions, task_ids, seeds):
        try:
            action = parse_completion_to_action(completion)
        except AgentParseError:
            # Shaped floor: FORMAT_FLOOR_REWARD + small format-adherence bonus.
            # Provides intra-group reward variance so GRPO can compute advantages.
            rewards.append(FORMAT_FLOOR_REWARD + _shaped_format_bonus(completion))
            continue

        env = build_env()
        env.reset(
            seed=int(seed),
            episode_id=f"grpo_{task_id}_{seed}",
            task_id=task_id or "easy_ria_late",
        )
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))

    return rewards


__all__ = ["rollout_reward", "FORMAT_FLOOR_REWARD"]
