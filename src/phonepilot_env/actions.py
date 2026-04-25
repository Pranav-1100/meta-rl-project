"""PhonePilot action space.

OpenEnv's FastAPI server calls ``action_cls.model_validate(data)`` with a single concrete
class. To expose 18 different tools, we wrap a discriminated Pydantic union in a top-level
:class:`PhonePilotAction` whose only payload field is ``body``. The JSON an agent emits
looks like::

    {"body": {"tool": "send_whatsapp", "contact": "Ria", "text": "I'll be 10 min late"}}

The ``tool`` literal on each sub-action discriminates the union.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from openenv.core import Action
from pydantic import Field


# ---------------------------------------------------------------------------
# Sub-actions (one Pydantic model per tool)
# ---------------------------------------------------------------------------


# --- Communication (7) ---


class CallAction(Action):
    tool: Literal["call"] = "call"
    contact: str


class WhatsAppCallAction(Action):
    tool: Literal["whatsapp_call"] = "whatsapp_call"
    contact: str


class HangUpAction(Action):
    tool: Literal["hang_up"] = "hang_up"


class SendWhatsAppAction(Action):
    tool: Literal["send_whatsapp"] = "send_whatsapp"
    contact: str
    text: str


class SendSMSAction(Action):
    tool: Literal["send_sms"] = "send_sms"
    contact: str
    text: str


class SendEmailAction(Action):
    tool: Literal["send_email"] = "send_email"
    contact: str
    subject: str
    body: str


class ReadMessagesAction(Action):
    tool: Literal["read_messages"] = "read_messages"
    contact: str | None = None
    channel: Literal["whatsapp", "sms", "email"] | None = None


class ReadNotificationsAction(Action):
    tool: Literal["read_notifications"] = "read_notifications"


# --- Calendar (3) ---


class CalendarViewAction(Action):
    tool: Literal["calendar_view"] = "calendar_view"
    date: str = Field(default="today", description="ISO date or 'today'/'tomorrow'.")


class CalendarAddAction(Action):
    tool: Literal["calendar_add"] = "calendar_add"
    title: str
    start_time: str = Field(description="ISO datetime or 'HH:MM' (assumed today)")
    duration_min: int = Field(default=60, ge=1, le=720)
    invitees: list[str] = Field(default_factory=list)


class CalendarRescheduleAction(Action):
    tool: Literal["calendar_reschedule"] = "calendar_reschedule"
    event_id: str
    new_start_time: str = Field(description="HH:MM (today) or 'tomorrow HH:MM' / 'YYYY-MM-DD HH:MM'.")


# --- Zomato (3) ---


class ZomatoSearchAction(Action):
    tool: Literal["zomato_search"] = "zomato_search"
    query: str
    cuisine: str | None = None
    veg_only: bool = False
    max_price_per_person: int | None = None


class ZomatoOpenAction(Action):
    tool: Literal["zomato_open"] = "zomato_open"
    restaurant_id: str


class ZomatoOrderAction(Action):
    tool: Literal["zomato_order"] = "zomato_order"
    restaurant_id: str
    items: list[str]
    delivery_time: str = Field(default="ASAP", description="'ASAP' or 'HH:MM'")


# --- Swiggy (3) — mirror of Zomato. Different catalog so the agent has to compare. ---


class SwiggySearchAction(Action):
    tool: Literal["swiggy_search"] = "swiggy_search"
    query: str
    cuisine: str | None = None
    veg_only: bool = False
    max_price_per_person: int | None = None


class SwiggyOpenAction(Action):
    tool: Literal["swiggy_open"] = "swiggy_open"
    restaurant_id: str


class SwiggyOrderAction(Action):
    tool: Literal["swiggy_order"] = "swiggy_order"
    restaurant_id: str
    items: list[str]
    delivery_time: str = Field(default="ASAP", description="'ASAP' or 'HH:MM'")


# --- Maps (2) ---


class MapsSearchAction(Action):
    tool: Literal["maps_search"] = "maps_search"
    query: str


class MapsTravelTimeAction(Action):
    tool: Literal["maps_travel_time"] = "maps_travel_time"
    origin: str
    destination: str


# --- Utility (4) ---


class WebSearchAction(Action):
    tool: Literal["web_search"] = "web_search"
    query: str


class WaitAction(Action):
    """Advance simulated time while waiting for a reply."""

    tool: Literal["wait"] = "wait"
    minutes: int = Field(ge=1, le=240)


class EndTaskAction(Action):
    """Declare the task complete.

    The ``success_claim`` is compared against the env's own internal success evaluation;
    mismatches trigger the truthfulness penalty. ``summary`` is audited for references
    to actions that never occurred.
    """

    tool: Literal["end_task"] = "end_task"
    success_claim: bool
    summary: str


class ThinkAction(Action):
    """Internal chain-of-thought. No env effect, no simulated-time cost."""

    tool: Literal["think"] = "think"
    reasoning: str


# ---------------------------------------------------------------------------
# Discriminated union + top-level wrapper
# ---------------------------------------------------------------------------

SUB_ACTION_CLASSES: tuple[type[Action], ...] = (
    CallAction,
    WhatsAppCallAction,
    HangUpAction,
    SendWhatsAppAction,
    SendSMSAction,
    SendEmailAction,
    ReadMessagesAction,
    ReadNotificationsAction,
    CalendarViewAction,
    CalendarAddAction,
    CalendarRescheduleAction,
    ZomatoSearchAction,
    ZomatoOpenAction,
    ZomatoOrderAction,
    SwiggySearchAction,
    SwiggyOpenAction,
    SwiggyOrderAction,
    MapsSearchAction,
    MapsTravelTimeAction,
    WebSearchAction,
    WaitAction,
    EndTaskAction,
    ThinkAction,
)


SubAction = Annotated[
    Union[
        CallAction,
        WhatsAppCallAction,
        HangUpAction,
        SendWhatsAppAction,
        SendSMSAction,
        SendEmailAction,
        ReadMessagesAction,
        ReadNotificationsAction,
        CalendarViewAction,
        CalendarAddAction,
        CalendarRescheduleAction,
        ZomatoSearchAction,
        ZomatoOpenAction,
        ZomatoOrderAction,
        SwiggySearchAction,
        SwiggyOpenAction,
        SwiggyOrderAction,
        MapsSearchAction,
        MapsTravelTimeAction,
        WebSearchAction,
        WaitAction,
        EndTaskAction,
        ThinkAction,
    ],
    Field(discriminator="tool"),
]


class PhonePilotAction(Action):
    """Top-level action wrapper. One required ``body`` field, which is the discriminated union."""

    body: SubAction


ACTION_REGISTRY: dict[str, type[Action]] = {
    cls.model_fields["tool"].default: cls for cls in SUB_ACTION_CLASSES  # type: ignore[misc]
}

TOOL_NAMES: tuple[str, ...] = tuple(ACTION_REGISTRY.keys())


__all__ = [
    "PhonePilotAction",
    "SubAction",
    "ACTION_REGISTRY",
    "TOOL_NAMES",
    "CallAction",
    "WhatsAppCallAction",
    "HangUpAction",
    "SendWhatsAppAction",
    "SendSMSAction",
    "SendEmailAction",
    "ReadMessagesAction",
    "ReadNotificationsAction",
    "CalendarViewAction",
    "CalendarAddAction",
    "CalendarRescheduleAction",
    "ZomatoSearchAction",
    "ZomatoOpenAction",
    "ZomatoOrderAction",
    "SwiggySearchAction",
    "SwiggyOpenAction",
    "SwiggyOrderAction",
    "MapsSearchAction",
    "MapsTravelTimeAction",
    "WebSearchAction",
    "WaitAction",
    "EndTaskAction",
    "ThinkAction",
]
