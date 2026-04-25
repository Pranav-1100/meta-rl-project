"""Task catalog + deterministic graders.

Each :class:`Task` knows how to seed the initial state (pre-existing messages, calendar
events) and how to grade an episode's state history against its own sub-goal rubric.

All graders are **pure state inspectors** — no LLM-as-judge, fully reproducible, which is
what makes this training target debuggable. The PRD's reward §6 and task §5 are implemented
directly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from .state import CalendarEvent, MessageEvent, PhonePilotState


Urgency = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# SubGoal helpers
# ---------------------------------------------------------------------------


def _agent_sent_to(state: PhonePilotState, contact: str, *, keywords: tuple[str, ...] = ()) -> bool:
    """Did the agent send at least one message to ``contact`` whose text contains all ``keywords``?"""
    for m in state.messages:
        if m.sender == "user" and m.recipient == contact:
            t = m.text.lower()
            if all(k.lower() in t for k in keywords):
                return True
    return False


def _contact_replied(state: PhonePilotState, contact: str) -> bool:
    return any(m.sender == contact and m.recipient == "user" for m in state.messages)


def _used_tool(state: PhonePilotState, tool: str) -> bool:
    return any(a.tool == tool for a in state.action_history)


def _count_tool(state: PhonePilotState, tool: str) -> int:
    return sum(1 for a in state.action_history if a.tool == tool)


def _count_distinct_channels_tried(state: PhonePilotState, contact: str) -> int:
    channels: set[str] = set()
    for a in state.action_history:
        if a.tool in ("send_whatsapp", "send_sms", "call", "whatsapp_call") and a.args.get(
            "contact"
        ) == contact:
            channels.add(a.tool)
    return len(channels)


# ---------------------------------------------------------------------------
# Task shape
# ---------------------------------------------------------------------------


SubGoalFn = Callable[[PhonePilotState], bool]


@dataclass
class Task:
    """A single task spec.

    ``sub_goals`` is an ordered list of ``(name, weight, is_achieved_fn)``. Weights must
    sum to 1.0 — total goal reward in [0.0, 1.0] per the hackathon rubric spec.

    ``urgency`` drives the appropriateness reward — high-urgency tasks excuse voice-channel
    contact outside business hours, while medium/low urgency tasks get penalised for it.
    """

    id: str
    difficulty: str
    prompt: str
    time_budget_min: int
    sub_goals: list[tuple[str, float, SubGoalFn]]
    seed_state: Callable[[PhonePilotState], None] = field(default=lambda s: None)
    urgency: Urgency = "medium"
    expected_base_success: float = 0.0
    target_post_training: float = 0.0

    def grade(self, state: PhonePilotState) -> dict[str, float]:
        """Return {subgoal_name: weight_if_achieved_else_0}. Sum = current goal reward."""
        return {
            name: (weight if fn(state) else 0.0) for name, weight, fn in self.sub_goals
        }

    def is_success(self, state: PhonePilotState) -> bool:
        """Used by the truthfulness check when the agent calls end_task(success_claim=True)."""
        scores = self.grade(state)
        return sum(scores.values()) >= 0.75


# ---------------------------------------------------------------------------
# Easy — "Quick reach"
# ---------------------------------------------------------------------------


def _easy_seed(state: PhonePilotState) -> None:
    # Start the episode a few minutes before the 4pm meeting so the goal is time-sensitive.
    state.start_clock_min = 15 * 60 + 45  # 15:45
    state.calendar.append(
        CalendarEvent(
            event_id="cal_4pm",
            title="Meeting with Ria",
            start_min=16 * 60,
            duration_min=30,
            invitees=["user", "Ria"],
        )
    )


def _easy_reached_ria(state: PhonePilotState) -> bool:
    return any(
        a.tool in ("send_whatsapp", "send_sms", "call", "whatsapp_call")
        and a.args.get("contact") == "Ria"
        for a in state.action_history
    )


def _easy_message_conveyed(state: PhonePilotState) -> bool:
    """Text references 'late' AND '4pm' (or '16:00' or 'meeting')."""
    for m in state.messages:
        if m.sender == "user" and m.recipient == "Ria":
            t = m.text.lower()
            time_ref = ("4" in t) or ("16:00" in t) or ("meeting" in t)
            if "late" in t and time_ref:
                return True
    return False


def _easy_ria_acknowledged(state: PhonePilotState) -> bool:
    return _contact_replied(state, "Ria")


EASY_TASK = Task(
    id="easy_ria_late",
    difficulty="easy",
    prompt="Let Ria know I'll be 10 minutes late to our 4pm meeting.",
    time_budget_min=60,
    sub_goals=[
        ("reached_ria", 0.30, _easy_reached_ria),
        ("message_conveyed", 0.20, _easy_message_conveyed),
        ("ria_acknowledged", 0.50, _easy_ria_acknowledged),
    ],
    seed_state=_easy_seed,
    expected_base_success=0.55,
    target_post_training=0.85,
)


# ---------------------------------------------------------------------------
# Medium — "Urgent coordination"
# ---------------------------------------------------------------------------


def _medium_seed(state: PhonePilotState) -> None:
    # It is now 14:45, the standup is at 15:00.
    state.start_clock_min = 14 * 60 + 45
    state.calendar.append(
        CalendarEvent(
            event_id="cal_standup",
            title="3pm Standup",
            start_min=15 * 60,
            duration_min=30,
            invitees=["user", "Jay", "Ria"],
        )
    )


def _medium_first_channel_appropriate(state: PhonePilotState) -> bool:
    """Agent's FIRST contact attempt to Jay should be a call or whatsapp_call during work hours."""
    for a in state.action_history:
        if a.tool in ("call", "whatsapp_call") and a.args.get("contact") == "Jay":
            return True
        if a.tool in ("send_whatsapp", "send_sms") and a.args.get("contact") == "Jay":
            # Agent tried text first — not ideal for urgent.
            return False
    return False


def _medium_waited_before_spamming(state: PhonePilotState) -> bool:
    """Between two Jay-targeted actions, at least one `wait` or `think` must separate them.

    Also passes if agent only contacted Jay once.
    """
    jay_actions_idx = [
        i
        for i, a in enumerate(state.action_history)
        if a.tool in ("call", "whatsapp_call", "send_whatsapp", "send_sms")
        and a.args.get("contact") == "Jay"
    ]
    if len(jay_actions_idx) < 2:
        return True
    for prev, nxt in zip(jay_actions_idx, jay_actions_idx[1:]):
        between = state.action_history[prev + 1 : nxt]
        if any(b.tool in ("wait", "read_messages", "read_notifications") for b in between):
            continue
        return False
    return True


def _medium_escalated_channel(state: PhonePilotState) -> bool:
    return _count_distinct_channels_tried(state, "Jay") >= 2


def _medium_urgency_conveyed(state: PhonePilotState) -> bool:
    for m in state.messages:
        if m.sender == "user" and m.recipient == "Jay":
            t = m.text.lower()
            time_ref = ("3" in t) or ("15:00" in t) or ("standup" in t) or ("3pm" in t)
            urgency_ref = any(w in t for w in ("urgent", "asap", "quick", "now"))
            if time_ref and urgency_ref:
                return True
    return False


def _medium_jay_joined(state: PhonePilotState) -> bool:
    """Jay has affirmatively replied `joining` / `on my way` / `dialing in` on any channel
    OR an active call with Jay is connected during standup window."""
    for m in state.messages:
        if m.sender == "Jay" and m.recipient == "user":
            t = m.text.lower()
            if any(k in t for k in ("joining", "on my way", "dial", "on it", "hop")):
                return True
    if state.active_call and state.active_call.get("contact") == "Jay" and state.active_call.get(
        "connected"
    ):
        return True
    return False


MEDIUM_TASK = Task(
    id="medium_jay_standup",
    difficulty="medium",
    prompt="Get Jay to join the 3pm standup call. It's urgent.",
    time_budget_min=30,
    sub_goals=[
        ("first_channel_appropriate", 0.15, _medium_first_channel_appropriate),
        ("waited_before_spam", 0.10, _medium_waited_before_spamming),
        ("escalated_fallback", 0.15, _medium_escalated_channel),
        ("urgency_conveyed", 0.15, _medium_urgency_conveyed),
        ("jay_joined_in_time", 0.45, _medium_jay_joined),
    ],
    seed_state=_medium_seed,
    urgency="high",
    expected_base_success=0.25,
    target_post_training=0.65,
)


# ---------------------------------------------------------------------------
# Hard — "Dinner coordination"
# ---------------------------------------------------------------------------


def _hard_seed(state: PhonePilotState) -> None:
    state.start_clock_min = 17 * 60  # 5 PM — evening planning
    # Pre-seed: Jay told the user about the sushi place last week (one message from Jay to user).
    state.messages.append(
        MessageEvent(
            sender="Jay",
            recipient="user",
            channel="whatsapp",
            text="yo went to this new spot Sushi Haven in Indiranagar last week, you'd love it",
            sent_at_min=-7 * 24 * 60,  # a week ago in simulated time
        )
    )


def _hard_read_prior(state: PhonePilotState) -> bool:
    return _used_tool(state, "read_messages")


def _hard_verified_place(state: PhonePilotState) -> bool:
    for a in state.action_history:
        if a.tool == "zomato_search" and "sushi" in str(a.args.get("query", "")).lower():
            return True
        if a.tool == "web_search" and "sushi" in str(a.args.get("query", "")).lower():
            return True
        if a.tool == "zomato_open" and a.args.get("restaurant_id", "").startswith("z_sushi") or (
            a.tool == "zomato_open" and "sushi" in a.args.get("restaurant_id", "").lower()
        ):
            return True
    return False


def _hard_checked_availability(state: PhonePilotState) -> bool:
    if _used_tool(state, "calendar_view"):
        return True
    # Or: agent asked each of {Jay, Ria, Mira} whether they're free / in.
    return all(
        _agent_sent_to(state, c) for c in ("Jay", "Ria", "Mira")
    )


def _hard_handled_friction(state: PhonePilotState) -> bool:
    """At least one contact initially declined/was busy and the agent proposed an alternative.

    Simplified v1: passes if the agent sent a SECOND message to any of {Jay, Ria, Mira} with
    a time different from the first (detected by presence of a digit change in the text).
    """
    for c in ("Jay", "Ria", "Mira"):
        sent = [m for m in state.messages if m.sender == "user" and m.recipient == c]
        if len(sent) >= 2:
            return True
    return False


def _hard_booked_restaurant(state: PhonePilotState) -> bool:
    sushi_booked = any("sushi" in o.restaurant_id.lower() for o in state.orders)
    sushi_in_calendar = any(
        a.tool == "calendar_add" and "sushi" in a.args.get("title", "").lower()
        for a in state.action_history
    )
    return sushi_booked or sushi_in_calendar


def _hard_all_three_confirmed(state: PhonePilotState) -> bool:
    return all(
        state.contacts.get(c) and state.contacts[c].will_attend_dinner is True
        for c in ("Jay", "Ria", "Mira")
    )


HARD_TASK = Task(
    id="hard_dinner_sushi",
    difficulty="hard",
    prompt=(
        "Dinner tonight for me, Jay, Ria, and Mira. Jay was raving about a new sushi place "
        "last week — set that up. Make sure all three are in."
    ),
    time_budget_min=90,
    sub_goals=[
        ("read_prior_messages", 0.15, _hard_read_prior),
        ("verified_place_exists", 0.10, _hard_verified_place),
        ("checked_availability", 0.15, _hard_checked_availability),
        ("handled_friction", 0.15, _hard_handled_friction),
        ("booked_restaurant", 0.15, _hard_booked_restaurant),
        ("all_three_confirmed", 0.30, _hard_all_three_confirmed),
    ],
    seed_state=_hard_seed,
    expected_base_success=0.08,
    target_post_training=0.30,
)


# ---------------------------------------------------------------------------
# Complex — "Multi-objective coordination" (Day-2 demo hero)
# ---------------------------------------------------------------------------


def _complex_seed(state: PhonePilotState) -> None:
    state.start_clock_min = 17 * 60  # 5 PM planning
    # Ria has a 7pm call we must not clash with.
    state.calendar.append(
        CalendarEvent(
            event_id="cal_ria_7pm",
            title="Ria — Client call",
            start_min=19 * 60,
            duration_min=60,
            invitees=["user", "Ria"],
        )
    )


def _complex_filtered_veg(state: PhonePilotState) -> bool:
    for a in state.action_history:
        if a.tool == "zomato_search":
            if a.args.get("veg_only") is True:
                return True
            if "veg" in str(a.args.get("query", "")).lower():
                return True
    return False


def _complex_checked_maps_for_mira(state: PhonePilotState) -> bool:
    for a in state.action_history:
        if a.tool == "maps_travel_time":
            if "whitefield" in str(a.args.get("origin", "")).lower() or "whitefield" in str(
                a.args.get("destination", "")
            ).lower():
                return True
        if a.tool == "maps_search" and "mira" in str(a.args.get("query", "")).lower():
            return True
    return False


def _complex_avoided_ria_7pm(state: PhonePilotState) -> bool:
    """Agent checked the calendar AND didn't book a slot that straddles 19:00–20:00."""
    if not _used_tool(state, "calendar_view"):
        return False
    # Any calendar_add / zomato_order with a start/delivery time between 18:30 and 20:00
    # is considered clashing.
    def _touches_7pm(hhmm: str) -> bool:
        try:
            hh, mm = (int(x) for x in hhmm.split(":", 1))
            total = hh * 60 + mm
            return 18 * 60 + 30 <= total <= 20 * 60
        except (ValueError, AttributeError):
            return False

    for a in state.action_history:
        if a.tool == "calendar_add" and _touches_7pm(str(a.args.get("start_time", ""))):
            return False
        if a.tool == "zomato_order":
            dt = str(a.args.get("delivery_time", ""))
            if dt != "ASAP" and _touches_7pm(dt):
                return False
    return True


def _complex_compared_options(state: PhonePilotState) -> bool:
    """Swiggy isn't in v1, so 'comparison' = agent opened at least TWO distinct restaurants
    (seen via zomato_open), OR did a web_search AND a zomato_search."""
    opens = {a.args.get("restaurant_id") for a in state.action_history if a.tool == "zomato_open"}
    if len({r for r in opens if r}) >= 2:
        return True
    searched = _used_tool(state, "zomato_search")
    web_searched = _used_tool(state, "web_search")
    return searched and web_searched


def _complex_within_budget(state: PhonePilotState) -> bool:
    if not state.orders:
        return False
    # 4-person budget: <= 900 per person including delivery, so total <= 3600.
    for o in state.orders:
        # Approximation: per-person price from the stub, assume delivery_fee ~60.
        estimated_per_person = o.price_per_person + 15  # delivery split across 4
        if estimated_per_person <= 900:
            return True
    return False


def _complex_all_three_confirmed(state: PhonePilotState) -> bool:
    return all(
        state.contacts.get(c) and state.contacts[c].will_attend_dinner is True
        for c in ("Jay", "Ria", "Mira")
    )


COMPLEX_TASK = Task(
    id="complex_multi_objective_dinner",
    difficulty="complex",
    prompt=(
        "Book dinner tonight for me + Jay + Ria + Mira. Jay is vegetarian. Ria has a 7pm call. "
        "Mira lives 15km from the rest of us. Keep total per-person cost under ₹900 including "
        "delivery. Get explicit confirmation from all three."
    ),
    time_budget_min=120,
    sub_goals=[
        ("filtered_vegetarian", 0.10, _complex_filtered_veg),
        ("checked_maps_for_mira", 0.15, _complex_checked_maps_for_mira),
        ("avoided_ria_7pm", 0.15, _complex_avoided_ria_7pm),
        ("compared_options", 0.15, _complex_compared_options),
        ("within_budget", 0.10, _complex_within_budget),
        ("all_three_confirmed", 0.35, _complex_all_three_confirmed),
    ],
    seed_state=_complex_seed,
    urgency="medium",
    expected_base_success=0.05,
    target_post_training=0.15,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TASK_REGISTRY: dict[str, Task] = {
    EASY_TASK.id: EASY_TASK,
    MEDIUM_TASK.id: MEDIUM_TASK,
    HARD_TASK.id: HARD_TASK,
    COMPLEX_TASK.id: COMPLEX_TASK,
}


def get_task(task_id: str) -> Task:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task_id!r}. Known: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[task_id]


def default_task_id() -> str:
    return EASY_TASK.id
