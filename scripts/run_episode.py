#!/usr/bin/env python3
"""Run a single PhonePilot episode against a pluggable policy and print the transcript.

Three built-in policies:

  * ``random`` — picks uniformly from the 18 tools with plausible defaults. Baseline #1
    for the 4-way comparison chart.
  * ``scripted_easy`` — deterministic solver for the Easy task (sanity check that a
    well-behaved agent hits > 0.9 total reward).
  * ``null`` — does nothing but ``wait`` until time budget expires. Useful as a floor.

Used two ways during the hackathon:

  1. Quick manual eyeballing of env behaviour:
        uv run python scripts/run_episode.py --task easy_ria_late --policy scripted_easy
  2. As the "random baseline" half of the 4-model comparison chart:
        for SEED in 1..50; do uv run python scripts/run_episode.py --task easy_ria_late \\
            --policy random --seed $SEED --json >> data/eval/random_easy.jsonl; done
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

# Allow running the script directly without `uv run` setting PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phonepilot_env.actions import PhonePilotAction, TOOL_NAMES  # noqa: E402
from phonepilot_env.agent_io import observation_to_prompt  # noqa: E402
from phonepilot_env.env import build_env  # noqa: E402
from phonepilot_env.observations import PhonePilotObservation  # noqa: E402


Policy = Callable[[PhonePilotObservation, random.Random], dict]


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


def null_policy(obs: PhonePilotObservation, rng: random.Random) -> dict:
    return {"body": {"tool": "wait", "minutes": 10}}


def random_policy(obs: PhonePilotObservation, rng: random.Random) -> dict:
    """Uniformly pick a tool and fill in plausible arguments."""
    contacts = ["Jay", "Ria", "Mira", "Mom"]
    tool = rng.choice(list(TOOL_NAMES))
    c = rng.choice(contacts)
    text = rng.choice(["hi", "quick question", "are you around?", "running late", "call me?"])
    body: dict = {"tool": tool}
    if tool in ("call", "whatsapp_call"):
        body["contact"] = c
    elif tool == "hang_up":
        pass
    elif tool in ("send_whatsapp", "send_sms"):
        body["contact"] = c
        body["text"] = text
    elif tool == "read_messages":
        body["contact"] = c
    elif tool == "read_notifications":
        pass
    elif tool == "calendar_view":
        body["date"] = "today"
    elif tool == "calendar_add":
        body["title"] = "Dinner"
        body["start_time"] = "20:00"
        body["duration_min"] = 60
        body["invitees"] = [c]
    elif tool == "zomato_search":
        body["query"] = rng.choice(["sushi", "pizza", "biryani"])
    elif tool == "zomato_open":
        body["restaurant_id"] = rng.choice(
            ["z_sushi_haven", "z_sakura_sushi", "z_pizza_place", "z_biryani_house"]
        )
    elif tool == "zomato_order":
        body["restaurant_id"] = "z_sushi_haven"
        body["items"] = ["Veg Maki Platter"]
    elif tool == "maps_search":
        body["query"] = rng.choice(["sushi", "coffee", "biryani"])
    elif tool == "maps_travel_time":
        body["origin"] = "Koramangala"
        body["destination"] = rng.choice(["Indiranagar", "Whitefield", "Jayanagar"])
    elif tool == "web_search":
        body["query"] = rng.choice(["sushi bangalore", "best dinner spot"])
    elif tool == "wait":
        body["minutes"] = rng.choice([5, 10, 15])
    elif tool == "think":
        body["reasoning"] = "considering options"
    elif tool == "end_task":
        body["success_claim"] = rng.random() < 0.5
        body["summary"] = "attempted the task"
    return {"body": body}


def scripted_easy_policy(obs: PhonePilotObservation, rng: random.Random) -> dict:
    """Deterministic Easy-task solver. Uses the turn index implied by recent_actions."""
    n_actions = len(obs.recent_actions)
    if n_actions == 0:
        return {
            "body": {
                "tool": "send_whatsapp",
                "contact": "Ria",
                "text": "I'll be 10 min late to our 4pm meeting",
            }
        }
    if n_actions == 1:
        return {"body": {"tool": "wait", "minutes": 15}}
    return {
        "body": {
            "tool": "end_task",
            "success_claim": True,
            "summary": "WhatsApped Ria to say I'd be 10 min late to our 4pm meeting.",
        }
    }


POLICIES: dict[str, Policy] = {
    "null": null_policy,
    "random": random_policy,
    "scripted_easy": scripted_easy_policy,
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    task_id: str,
    policy_name: str,
    seed: int,
    max_steps: int = 40,
    verbose: bool = True,
) -> dict:
    policy = POLICIES[policy_name]
    env = build_env()
    obs = env.reset(seed=seed, episode_id=f"{policy_name}_{task_id}_{seed}", task_id=task_id)
    rng = random.Random(seed * 1000 + 17)  # separate from env's rng

    steps: list[dict] = []
    for turn in range(max_steps):
        if verbose:
            print(observation_to_prompt(obs, turn_index=turn))
        action_dict = policy(obs, rng)
        if verbose:
            print(">>> ACTION:", json.dumps(action_dict))

        try:
            action = PhonePilotAction.model_validate(action_dict)
        except Exception as e:
            if verbose:
                print(f"!!! policy emitted invalid action: {e}")
            # Count as a format error — env will penalise; just continue.
            action_dict = {"body": {"tool": "wait", "minutes": 1}}
            action = PhonePilotAction.model_validate(action_dict)

        obs = env.step(action)
        steps.append(
            {
                "turn": turn,
                "action": action_dict,
                "reward": obs.reward,
                "done": obs.done,
                "sub_goals_fired": list(obs.info.get("sub_goals_fired", [])),
            }
        )
        if verbose:
            print(f"<<< reward={obs.reward}  done={obs.done}  fired={obs.info.get('sub_goals_fired')}\n")
        if obs.done:
            break

    return {
        "task_id": task_id,
        "policy": policy_name,
        "seed": seed,
        "total_reward": env.state.total_reward,
        "reward_components": dict(env.state.reward_components),
        "steps_taken": len(steps),
        "terminated": env.state.terminated,
        "end_claim": env.state.end_task_success_claim,
        "end_summary": env.state.end_task_summary,
        "steps": steps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        default="easy_ria_late",
        choices=[
            "easy_ria_late",
            "medium_jay_standup",
            "hard_dinner_sushi",
            "complex_multi_objective_dinner",
        ],
    )
    p.add_argument("--policy", default="scripted_easy", choices=sorted(POLICIES))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--json", action="store_true", help="emit summary as JSON only (no transcript)")
    args = p.parse_args()

    result = run_episode(args.task, args.policy, args.seed, args.max_steps, verbose=not args.json)

    if args.json:
        # Strip steps for compactness in eval logs.
        compact = {k: v for k, v in result.items() if k != "steps"}
        print(json.dumps(compact))
    else:
        print("=" * 60)
        print(f"TASK {args.task} via {args.policy} (seed {args.seed})")
        print(f"total_reward: {result['total_reward']:.3f}")
        print(f"reward_components: {result['reward_components']}")
        print(f"steps: {result['steps_taken']}  terminated: {result['terminated']}")


if __name__ == "__main__":
    main()
