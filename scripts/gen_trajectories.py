#!/usr/bin/env python3
"""Generate synthetic trajectories for SFT warmup by running Claude as the agent.

Output is one JSONL file per task id under ``data/trajectories/<task_id>.jsonl``. Each
line is one complete episode in HF-Transformers-chat-format::

    {
      "task_id": "easy_ria_late",
      "seed": 3,
      "total_reward": 0.94,
      "reward_components": {...},
      "terminated": true,
      "messages": [
        {"role": "system", "content": "<SYSTEM_PROMPT>"},
        {"role": "user", "content": "<turn-0 observation text>"},
        {"role": "assistant", "content": "```json\\n{...}\\n```"},
        {"role": "user", "content": "<turn-1 observation text>"},
        ...
      ]
    }

Usage
-----

    # needs ANTHROPIC_API_KEY either in env or in .env
    uv run python scripts/gen_trajectories.py \\
        --task easy_ria_late --count 50 --seed-start 1 --seed-end 50

    # or do a dry-run that uses the scripted_easy policy (no API calls) — verifies the
    # pipeline end-to-end:
    uv run python scripts/gen_trajectories.py --task easy_ria_late --count 5 --dry-run

The synthetic-trajectory budget we're aiming at for SFT is ~200 total across tasks. A
good starting split:
    easy_ria_late          : 80
    medium_jay_standup     : 60
    hard_dinner_sushi      : 40
    complex_multi_...      : 20
Because the Complex task often fails even for Claude, we keep its count low and SFT
still benefits from the partial-progress sub-goal firings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Make the source tree importable without `uv run -m` magic.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phonepilot_env.actions import PhonePilotAction  # noqa: E402
from phonepilot_env.agent_io import (  # noqa: E402
    SYSTEM_PROMPT,
    AgentParseError,
    action_to_completion,
    observation_to_prompt,
    parse_completion_to_action,
)
from phonepilot_env.env import build_env  # noqa: E402
from phonepilot_env.observations import PhonePilotObservation  # noqa: E402


TRAJ_DIR = Path(__file__).resolve().parent.parent / "data" / "trajectories"
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@dataclass
class AnthropicAgent:
    """Claude as the agent. Initialised lazily so --dry-run doesn't require the SDK."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 400
    temperature: float = 0.6
    _client: object = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "anthropic SDK not installed — uv sync should have installed it"
                ) from e
            # Load .env if present.
            try:
                from dotenv import load_dotenv

                load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
            except ImportError:
                pass
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Put it in .env or export it before running."
                )
            self._client = Anthropic()
        return self._client

    def turn(self, messages: list[dict]) -> str:
        """Given the ongoing chat history, return one assistant completion string."""
        client = self._ensure_client()
        # anthropic SDK wants system separately + only user/assistant roles in messages.
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        chat = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=chat,
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


@dataclass
class ScriptedAgent:
    """A trivial fallback agent for --dry-run. Handles the Easy task cleanly and falls
    back to ``wait → end_task(False)`` on harder tasks so the pipeline doesn't hang."""

    def turn(self, messages: list[dict]) -> str:
        # Infer turn index from count of prior assistant messages.
        turn = sum(1 for m in messages if m["role"] == "assistant")
        user_text = messages[-1]["content"].lower()
        if "let ria know" in user_text and turn == 0:
            return action_to_completion(
                PhonePilotAction.model_validate(
                    {
                        "body": {
                            "tool": "send_whatsapp",
                            "contact": "Ria",
                            "text": "I'll be 10 min late to our 4pm meeting",
                        }
                    }
                )
            )
        if "let ria know" in user_text and turn == 1:
            return action_to_completion(
                PhonePilotAction.model_validate({"body": {"tool": "wait", "minutes": 15}})
            )
        if "let ria know" in user_text and turn >= 2:
            return action_to_completion(
                PhonePilotAction.model_validate(
                    {
                        "body": {
                            "tool": "end_task",
                            "success_claim": True,
                            "summary": "WhatsApped Ria about the 10-min delay to the 4pm meeting.",
                        }
                    }
                )
            )
        # For other tasks, waste a couple of turns then give up honestly.
        if turn < 2:
            return action_to_completion(
                PhonePilotAction.model_validate({"body": {"tool": "wait", "minutes": 5}})
            )
        return action_to_completion(
            PhonePilotAction.model_validate(
                {
                    "body": {
                        "tool": "end_task",
                        "success_claim": False,
                        "summary": "Could not complete within budget.",
                    }
                }
            )
        )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_one_episode(
    task_id: str,
    seed: int,
    agent,
    max_turns: int = 25,
    verbose: bool = False,
) -> dict:
    env = build_env()
    obs = env.reset(seed=seed, episode_id=f"synth_{task_id}_{seed}", task_id=task_id)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in range(max_turns):
        user_msg = observation_to_prompt(obs, turn_index=turn)
        messages.append({"role": "user", "content": user_msg})

        # Two retries on parse errors — on the third we just inject a wait action.
        completion: str | None = None
        action: PhonePilotAction | None = None
        for retry in range(3):
            try:
                completion = agent.turn(messages)
                action = parse_completion_to_action(completion)
                break
            except AgentParseError as e:
                if verbose:
                    print(f"[turn {turn}] parse error (retry {retry}): {e}")
                if retry == 2:
                    action = PhonePilotAction.model_validate(
                        {"body": {"tool": "wait", "minutes": 5}}
                    )
                    completion = action_to_completion(action)

        # Replace the agent's (possibly unparseable) text with the canonical serialised
        # form so SFT training always sees well-formed completions.
        assert action is not None
        canonical = action_to_completion(action)
        messages.append({"role": "assistant", "content": canonical})

        obs = env.step(action)
        if verbose:
            print(
                f"[turn {turn}] {action.body.tool} → reward {obs.reward:.3f} "
                f"fired {obs.info.get('sub_goals_fired')}"
            )
        if obs.done:
            break

    return {
        "task_id": task_id,
        "seed": seed,
        "total_reward": env.state.total_reward,
        "reward_components": dict(env.state.reward_components),
        "terminated": env.state.terminated,
        "end_claim": env.state.end_task_success_claim,
        "end_summary": env.state.end_task_summary,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        required=True,
        choices=[
            "easy_ria_late",
            "medium_jay_standup",
            "hard_dinner_sushi",
            "complex_multi_objective_dinner",
        ],
    )
    p.add_argument("--count", type=int, default=10, help="number of episodes to generate")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--max-turns", type=int, default=25)
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--dry-run", action="store_true", help="skip Claude, use a scripted agent")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--min-reward", type=float, default=-100.0,
                   help="discard episodes with total_reward below this (after running)")
    args = p.parse_args()

    agent = ScriptedAgent() if args.dry_run else AnthropicAgent(model=args.model)

    out_path = TRAJ_DIR / f"{args.task}.jsonl"
    kept = 0
    skipped_low_reward = 0

    t0 = time.time()
    with out_path.open("a") as f:
        for i in range(args.count):
            seed = args.seed_start + i
            try:
                result = run_one_episode(
                    args.task, seed, agent, max_turns=args.max_turns, verbose=args.verbose
                )
            except Exception as e:  # noqa: BLE001
                print(f"seed={seed} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if result["total_reward"] < args.min_reward:
                skipped_low_reward += 1
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            kept += 1
            print(
                f"[{i+1}/{args.count}] seed={seed}  total_reward={result['total_reward']:.2f}  "
                f"fired={len([k for k,v in result['reward_components'].items() if v>0])}"
            )

    print(
        f"\nDone. kept {kept} / {args.count} episodes "
        f"(skipped {skipped_low_reward} below --min-reward={args.min_reward}) "
        f"in {time.time()-t0:.1f}s → {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
