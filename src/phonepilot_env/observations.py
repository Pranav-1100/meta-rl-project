"""PhonePilot observation — what the agent sees each step.

Hidden from the agent: contact profiles, task's internal sub-goal state, reward component
breakdown. Those live in :class:`~phonepilot_env.state.PhonePilotState`.
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core import Observation
from pydantic import Field

from .actions import TOOL_NAMES


class Notification(Observation.__base__):  # type: ignore[misc]
    """A single new alert surfaced to the agent since the last step."""

    kind: Literal["message", "call_incoming", "call_missed", "calendar_reminder", "system"]
    channel: Literal["whatsapp", "sms", "email", "call", "calendar", "system"] | None = None
    contact: str | None = None
    preview: str = ""
    timestamp: str = Field(description="HH:MM of the simulated clock")


class ActionOutcome(Observation.__base__):  # type: ignore[misc]
    """Compact summary of a recent action and the env's response — for the agent's scratchpad."""

    tool: str
    arg_summary: str = ""
    outcome: str = ""
    at_time: str = ""


class PhonePilotObservation(Observation):
    """Per-step view the agent receives.

    Inherits ``done: bool`` and ``reward: float | None`` from :class:`openenv.core.Observation`.
    The reward field is populated by the environment's step() after running the reward function.
    """

    user_goal: str = Field(description="The task prompt for this episode.")
    current_time: str = Field(description="Simulated wall-clock time, e.g. '14:35'.")
    time_budget_remaining_min: int = Field(
        ge=0, description="Simulated minutes left before the episode auto-terminates."
    )

    available_tools: tuple[str, ...] = Field(
        default=TOOL_NAMES,
        description="Names of all callable tools. Emit actions as {'body':{'tool':<name>,...}}.",
    )

    recent_actions: list[ActionOutcome] = Field(
        default_factory=list,
        description="Last 5 (action, outcome) pairs.",
    )

    active_call: str | None = Field(
        default=None,
        description="'ringing Jay', 'connected to Ria', or None.",
    )

    open_app_view: dict[str, Any] | None = Field(
        default=None,
        description="Payload from the last app-opening action (e.g. zomato_open result).",
    )

    notifications: list[Notification] = Field(
        default_factory=list,
        description="New incoming messages/calls/reminders since the last step.",
    )

    conversation_summaries: dict[str, str] = Field(
        default_factory=dict,
        description="Contact → short string of the last message in the most recent channel.",
    )

    error: str | None = Field(
        default=None,
        description="Human-readable parse/validation error from the previous action, if any.",
    )

    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Ad-hoc diagnostic fields (reward breakdown, sub-goal keys fired, etc.)."
        " Training code may read this; agent should not rely on it.",
    )
