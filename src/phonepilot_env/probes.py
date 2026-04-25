"""Capability probes (PRD §8.4) — 10 deterministic single-skill mini-tasks.

Each probe is a 1–2 step interaction designed to test ONE capability in isolation. Run
the whole battery every N training steps; plot ``probes_passed_out_of_10`` to get a clean
monotonic learning curve even when the main reward is noisy.

A probe is a function ``(env_factory) → bool`` that:
  1. Creates a fresh env via ``env_factory()``.
  2. Resets to a chosen seed (each probe pins its own seed for reproducibility).
  3. Issues 1–3 scripted actions OR runs a provided policy callable.
  4. Inspects state to verify the capability fired.

For training-time use, pass a model-driven policy via :func:`run_probes_with_policy`
which renders observation_to_prompt → model.generate → parse_completion_to_action.
Pass/fail comes from each probe's own state inspector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .actions import PhonePilotAction
from .state import PhonePilotState


# ---------------------------------------------------------------------------
# Probe shape
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    name: str
    instruction: str            # the user-goal text given to the policy
    task_id: str                # which task to seed (the env still wants a task)
    seed: int                   # deterministic seed
    max_steps: int              # short horizon
    inspector: Callable[[PhonePilotState], bool]


def _ev(state: PhonePilotState, tool: str, **arg_filters) -> bool:
    """Convenience: did the agent call ``tool`` with all of the given args?"""
    for a in state.action_history:
        if a.tool != tool:
            continue
        if all(str(arg_filters[k]).lower() in str(a.args.get(k, "")).lower() for k in arg_filters):
            return True
    return False


# ---------------------------------------------------------------------------
# 10 probes — one per capability the agent should master
# ---------------------------------------------------------------------------


PROBES: list[Probe] = [
    Probe(
        name="p01_send_one_line_whatsapp",
        instruction="Send Ria a one-line WhatsApp saying 'hey'.",
        task_id="easy_ria_late",
        seed=901,
        max_steps=3,
        inspector=lambda s: _ev(s, "send_whatsapp", contact="Ria"),
    ),
    Probe(
        name="p02_search_pizza",
        instruction="Find a pizza place in Bangalore on Zomato.",
        task_id="hard_dinner_sushi",
        seed=902,
        max_steps=3,
        inspector=lambda s: _ev(s, "zomato_search", query="pizza"),
    ),
    Probe(
        name="p03_view_calendar",
        instruction="Check what's on my calendar today.",
        task_id="hard_dinner_sushi",
        seed=903,
        max_steps=3,
        inspector=lambda s: _ev(s, "calendar_view"),
    ),
    Probe(
        name="p04_travel_time_query",
        instruction="How long does it take to drive from Koramangala to Whitefield?",
        task_id="complex_multi_objective_dinner",
        seed=904,
        max_steps=3,
        inspector=lambda s: _ev(s, "maps_travel_time"),
    ),
    Probe(
        name="p05_read_messages_from_jay",
        instruction="Read the last messages from Jay.",
        task_id="hard_dinner_sushi",
        seed=905,
        max_steps=3,
        inspector=lambda s: _ev(s, "read_messages", contact="Jay"),
    ),
    Probe(
        name="p06_web_search_biryani",
        instruction="Web-search for the best biryani in Bangalore.",
        task_id="hard_dinner_sushi",
        seed=906,
        max_steps=3,
        inspector=lambda s: _ev(s, "web_search", query="biryani"),
    ),
    Probe(
        name="p07_calendar_add_event",
        instruction="Add an event 'Dinner' tonight at 8pm.",
        task_id="hard_dinner_sushi",
        seed=907,
        max_steps=3,
        inspector=lambda s: any(a.tool == "calendar_add" for a in s.action_history),
    ),
    Probe(
        name="p08_send_email_simple",
        instruction="Email Jay with subject 'hi' and a one-line body.",
        task_id="easy_ria_late",
        seed=908,
        max_steps=3,
        inspector=lambda s: _ev(s, "send_email", contact="Jay"),
    ),
    Probe(
        name="p09_swiggy_search_veg",
        instruction="Find a vegetarian Swiggy restaurant.",
        task_id="complex_multi_objective_dinner",
        seed=909,
        max_steps=3,
        inspector=lambda s: any(
            a.tool == "swiggy_search" and (a.args.get("veg_only") is True or "veg" in str(a.args.get("query", "")).lower())
            for a in s.action_history
        ),
    ),
    Probe(
        name="p10_calendar_reschedule",
        instruction="Reschedule any existing event to a different time.",
        task_id="multi_day_reschedule",  # this task seeds a calendar event we can move
        seed=910,
        max_steps=3,
        inspector=lambda s: any(a.tool == "calendar_reschedule" for a in s.action_history),
    ),
]


PolicyFn = Callable[..., dict]  # (obs, rng) → {"body": {...}}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_probes_with_policy(env_factory, policy: PolicyFn) -> dict[str, bool]:
    """Run all 10 probes against ``policy``. Returns ``{probe_name: passed}``."""
    import random

    results: dict[str, bool] = {}
    for probe in PROBES:
        env = env_factory()
        env.reset(seed=probe.seed, episode_id=f"probe_{probe.name}", task_id=probe.task_id)
        rng = random.Random(probe.seed * 31 + 7)
        for _ in range(probe.max_steps):
            # Policies see a synthetic observation that overrides the goal text with the
            # probe's instruction. We rebuild the observation we'd give the agent, then
            # let the policy choose an action.
            obs = env._build_observation(  # type: ignore[attr-defined]
                newly_delivered=[], last_outcome=None, format_error=None
            )
            obs.user_goal = probe.instruction
            action_dict = policy(obs, rng)
            try:
                action = PhonePilotAction.model_validate(action_dict)
            except Exception:
                break
            obs = env.step(action)
            if obs.done:
                break
        results[probe.name] = probe.inspector(env.state)
    return results


def run_probes_with_actions(env_factory, action_lookup: dict[str, list[dict]]) -> dict[str, bool]:
    """Test runner: feeds each probe a hand-coded action sequence keyed by probe name."""
    results: dict[str, bool] = {}
    for probe in PROBES:
        env = env_factory()
        env.reset(seed=probe.seed, episode_id=f"probe_{probe.name}", task_id=probe.task_id)
        actions = action_lookup.get(probe.name, [])
        for action_dict in actions[: probe.max_steps]:
            try:
                action = PhonePilotAction.model_validate(action_dict)
            except Exception:
                break
            obs = env.step(action)
            if obs.done:
                break
        results[probe.name] = probe.inspector(env.state)
    return results
