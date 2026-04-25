"""Drama injector — stochastic mid-episode curveballs that test recovery.

This is the project's headline-novelty. Most RL-environment-for-LLMs work treats episodes
as deterministic-modulo-policy-noise; PhonePilot can fire **named events** mid-rollout to
simulate the kind of real-world surprises an actual phone-OS agent has to recover from:

  * ``contact_dropout`` — a contact's responsiveness drops to ~0 for the rest of the
    episode (e.g. they put their phone in airplane mode).
  * ``phone_low_battery`` — voice calls fail (drop) for the rest of the episode; messaging
    still works.
  * ``restaurant_unavailable`` — a specific restaurant becomes unbookable (orders to it
    return error).
  * ``traffic_jam`` — ``maps_travel_time`` results double.
  * ``new_constraint`` — a contact pings the agent with an extra requirement
    ("Mira: btw I'm allergic to fish") that wasn't in the original task prompt.

Each event has a probability of firing per step and a `trigger_after_step` floor so it
doesn't fire on step 0. The env opts into drama per-task via ``Task.use_drama``.

The dashboard's ``recovery_rate`` is the metric judges should watch when drama is on:
trained models recover (escalate channel, reorder elsewhere), base models give up or lie.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .state import PhonePilotState


@dataclass
class DramaEvent:
    name: str
    probability_per_step: float = 0.05
    trigger_after_step: int = 3
    fired: bool = False  # ensures it only fires once per episode
    apply_fn: Callable[["PhonePilotState"], str] = lambda s: ""


@dataclass
class DramaConfig:
    enabled: bool = False
    events: list[DramaEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event apply-functions
# ---------------------------------------------------------------------------


def _apply_contact_dropout(state: "PhonePilotState") -> str:
    """Pick a non-Mom contact and drop their responsiveness near zero. Mom is excluded
    because she's the most responsive baseline and we don't want all 4 contacts to be
    unreachable at once."""
    candidates = [c for c in state.contacts.keys() if c != "Mom"]
    if not candidates:
        return ""
    target = candidates[len(state.action_history) % len(candidates)]  # deterministic
    profile = state.contacts[target]
    profile.call_pickup_prob_work_hours = 0.05
    profile.call_pickup_prob_after_hours = 0.05
    profile.whatsapp_reply_median_min = 240
    profile.sms_reply_median_min = 240
    return f"DRAMA: {target}'s phone went silent (no pickups, very slow text replies)."


def _apply_phone_low_battery(state: "PhonePilotState") -> str:
    """All voice tools fail for the rest of the episode. We model this by zeroing pickup
    probabilities across all contacts — effectively a phone-side outage."""
    for profile in state.contacts.values():
        profile.call_pickup_prob_work_hours = 0.0
        profile.call_pickup_prob_after_hours = 0.0
    return "DRAMA: your phone battery is critical — voice calls now fail."


def _apply_restaurant_unavailable(state: "PhonePilotState") -> str:
    """Mark Sushi Haven as unavailable by tagging it on state.extras."""
    extras = getattr(state, "model_extra", None) or {}
    extras["unavailable_restaurants"] = list(extras.get("unavailable_restaurants", [])) + [
        "z_sushi_haven"
    ]
    # Pydantic with extra="allow" stores extras into model_extra.
    # We also drop a synthetic notification.
    return "DRAMA: Sushi Haven just went out of stock for the night."


def _apply_traffic_jam(state: "PhonePilotState") -> str:
    extras = getattr(state, "model_extra", None) or {}
    extras["traffic_multiplier"] = 2.0
    return "DRAMA: heavy traffic spotted — travel times will be doubled."


def _apply_new_constraint(state: "PhonePilotState") -> str:
    """A late-breaking message from Mira adds an ad-hoc dietary constraint."""
    from .state import MessageEvent

    state.messages.append(
        MessageEvent(
            sender="Mira",
            recipient="user",
            channel="whatsapp",
            text="btw don't pick anywhere with seafood — I'm allergic, sorry forgot to mention",
            sent_at_min=state.current_time_min,
        )
    )
    return "DRAMA: Mira just added a no-seafood constraint."


DEFAULT_EVENT_LIBRARY: dict[str, Callable[["PhonePilotState"], str]] = {
    "contact_dropout": _apply_contact_dropout,
    "phone_low_battery": _apply_phone_low_battery,
    "restaurant_unavailable": _apply_restaurant_unavailable,
    "traffic_jam": _apply_traffic_jam,
    "new_constraint": _apply_new_constraint,
}


# ---------------------------------------------------------------------------
# Step hook
# ---------------------------------------------------------------------------


def maybe_fire_drama(
    state: "PhonePilotState",
    config: DramaConfig,
    rng: random.Random,
    step_idx: int,
) -> str | None:
    """Called by the env at the start of each step. Returns a notification string if an
    event fires this step, else ``None``. Each event fires at most once per episode."""
    if not config.enabled or not config.events:
        return None
    for event in config.events:
        if event.fired:
            continue
        if step_idx < event.trigger_after_step:
            continue
        if rng.random() < event.probability_per_step:
            event.fired = True
            return event.apply_fn(state)
    return None


def make_default_drama_config(rng: random.Random | None = None) -> DramaConfig:
    """Pick 2 random events out of the library — we don't want every episode to fire all
    of them, that's noise."""
    rng = rng or random.Random(42)
    chosen = rng.sample(list(DEFAULT_EVENT_LIBRARY.keys()), k=2)
    return DramaConfig(
        enabled=True,
        events=[
            DramaEvent(
                name=name,
                probability_per_step=0.20,
                trigger_after_step=2,
                apply_fn=DEFAULT_EVENT_LIBRARY[name],
            )
            for name in chosen
        ],
    )
