"""Internal environment state — NOT shown to the agent.

Holds contact profiles, full message history, calendar events, simulated clock, reward
bookkeeping, and sub-goal flags. The Observation exposes only a curated subset.
"""

from __future__ import annotations

from typing import Any, Literal

from openenv.core import State
from pydantic import Field


Channel = Literal["whatsapp", "sms", "email", "call"]


class ContactProfile(State.__base__):  # type: ignore[misc]
    """Hidden responsiveness profile driving stochastic pickup + reply delays."""

    name: str
    call_pickup_prob_work_hours: float = 0.5
    call_pickup_prob_after_hours: float = 0.8
    whatsapp_reply_median_min: int = 10
    sms_reply_median_min: int = 30
    email_reply_median_min: int = 360
    preferred_channel: Channel = "whatsapp"
    annoyance_threshold: int = 3

    # Runtime counters (reset each episode):
    unanswered_agent_messages: int = 0
    will_attend_dinner: bool | None = None
    location: str = "Koramangala"  # used for maps demo
    dietary: Literal["any", "vegetarian", "vegan"] = "any"


class MessageEvent(State.__base__):  # type: ignore[misc]
    sender: str  # "user" (the assistant on behalf of the user) or a contact name
    recipient: str  # contact name or "user"
    channel: Channel
    text: str
    sent_at_min: int  # simulated minutes since episode start


class CalendarEvent(State.__base__):  # type: ignore[misc]
    event_id: str
    title: str
    start_min: int  # minutes since episode-start day 00:00
    duration_min: int = 60
    invitees: list[str] = Field(default_factory=list)


class Order(State.__base__):  # type: ignore[misc]
    order_id: str
    restaurant_id: str
    items: list[str]
    delivery_time: str
    placed_at_min: int
    price_per_person: int = 0


class ActionRecord(State.__base__):  # type: ignore[misc]
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    at_min: int = 0


class PendingReply(State.__base__):  # type: ignore[misc]
    """A reply that the contact simulator has scheduled. Fires when clock reaches ``at_min``."""

    from_contact: str
    channel: Channel
    text: str
    at_min: int


class PhonePilotState(State):
    """Complete hidden state for one episode."""

    # Simulated time
    start_clock_min: int = Field(default=14 * 60, description="Minutes-of-day the episode begins.")
    current_time_min: int = Field(default=0, description="Simulated minutes since episode start.")
    time_budget_min: int = 120

    # The task
    active_task_id: str = ""

    # World
    contacts: dict[str, ContactProfile] = Field(default_factory=dict)
    messages: list[MessageEvent] = Field(default_factory=list)
    calendar: list[CalendarEvent] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)
    active_call: dict[str, Any] | None = None  # {"contact": str, "connected": bool, "since_min": int}

    # Agent trajectory
    action_history: list[ActionRecord] = Field(default_factory=list)

    # Scheduled events that fire when clock advances
    pending_replies: list[PendingReply] = Field(default_factory=list)
    delivered_notifications_after_min: int = 0  # last-seen watermark for notifications

    # Reward bookkeeping
    sub_goals_fired: set[str] = Field(default_factory=set)
    total_reward: float = 0.0
    reward_components: dict[str, float] = Field(
        default_factory=lambda: {
            "goal": 0.0,
            "truthfulness": 0.0,
            "efficiency": 0.0,
            "appropriateness": 0.0,
            "format": 0.0,
            "calibration": 0.0,
        }
    )

    # Safety counters
    format_error_streak: int = 0
    terminated: bool = False
    end_task_success_claim: bool | None = None
    end_task_summary: str = ""

    # ------------------------------------------------------------------ helpers

    def clock_hhmm(self, offset_min: int = 0) -> str:
        total = (self.start_clock_min + self.current_time_min + offset_min) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"

    def is_work_hours(self) -> bool:
        hour = ((self.start_clock_min + self.current_time_min) // 60) % 24
        return 9 <= hour < 18

    def advance_time(self, minutes: int) -> None:
        self.current_time_min += max(0, minutes)
