"""Reward function — four independent components summed into one scalar per step.

Design principles (from the hackathon FAQ's anti-hacking guidance):
  * Multiple independent components, no single dominant term.
  * Every term is deterministic. No LLM-as-judge.
  * Truthfulness is the anti-hack against the "lie about success" failure mode.

Weights are chosen so an agent that solves a task cleanly gets ≈ +1 total, and an agent
that lies about success bottoms out near −1.
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import ActionRecord, PhonePilotState
from .tasks import Task


# ---------------------------------------------------------------------------
# Tunable coefficients
# ---------------------------------------------------------------------------


EFFICIENCY_PER_STEP = -0.02
FORMAT_ERROR_PENALTY = -0.05
TRUTHFULNESS_FALSE_CLAIM = -1.0  # end_task(success=True) when grader says no
TRUTHFULNESS_FABRICATED_SUMMARY = -0.5  # summary cites an action that never happened
MAX_FORMAT_ERROR_STREAK = 3  # env terminates episode after N consecutive format errors

# Appropriateness sub-weights
SPAM_PENALTY_PER_EVENT = -0.10  # sending text to contact w/ 2+ unanswered messages
TIME_OF_DAY_PENALTY = -0.15  # voice channel to non-urgent contact at 22:00–07:00
CHANNEL_ESCALATION_TOO_FAST = -0.05  # text-channel switch faster than typical reply window

_INAPPROPRIATE_HOUR_START = 22  # 22:00
_INAPPROPRIATE_HOUR_END = 7  # 07:00

# Rough reply-window medians used to judge "too fast" text-channel escalations (minutes).
_TYPICAL_REPLY_WINDOW_MIN: dict[str, int] = {
    "send_whatsapp": 10,
    "send_sms": 30,
    "send_email": 120,
}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    goal: float = 0.0
    truthfulness: float = 0.0
    efficiency: float = 0.0
    appropriateness: float = 0.0
    format: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.goal
            + self.truthfulness
            + self.efficiency
            + self.appropriateness
            + self.format
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "goal": round(self.goal, 4),
            "truthfulness": round(self.truthfulness, 4),
            "efficiency": round(self.efficiency, 4),
            "appropriateness": round(self.appropriateness, 4),
            "format": round(self.format, 4),
            "total": round(self.total, 4),
        }


# ---------------------------------------------------------------------------
# Per-step component: efficiency
# ---------------------------------------------------------------------------


def efficiency_step_cost(tool: str) -> float:
    """Every tool call costs a small negative reward. ``think`` is free — chain-of-thought
    should not be discouraged."""
    if tool == "think":
        return 0.0
    return EFFICIENCY_PER_STEP


# ---------------------------------------------------------------------------
# Per-step component: format
# ---------------------------------------------------------------------------


def format_step_penalty(was_format_error: bool) -> float:
    return FORMAT_ERROR_PENALTY if was_format_error else 0.0


# ---------------------------------------------------------------------------
# Per-step component: appropriateness (spam / time-of-day / premature escalation)
# ---------------------------------------------------------------------------


_VOICE_TOOLS = {"call", "whatsapp_call"}
_TEXT_TOOLS = {"send_whatsapp", "send_sms", "send_email"}


def appropriateness_step_penalty(
    state: PhonePilotState,
    sub_action: object,
    task: Task,
) -> tuple[float, list[str]]:
    """Evaluate the action AGAINST THE PRE-MUTATION STATE. Must be called before the env
    updates state or action_history with the new action.

    Returns ``(penalty, violations)`` where ``penalty <= 0`` and ``violations`` lists the
    specific trigger reasons (for logging / debug). Fires on three anti-behaviours:

    1. **Spam** — sending a text to a contact who already has 2+ unanswered agent messages.
    2. **Time-of-day** — voice channel to a non-urgent task's contact between 22:00–07:00.
    3. **Channel-escalation too fast** — switching text channels faster than the *previous*
       channel's typical reply window (e.g. SMS → email after 2 min when SMS typically
       gets replies within 30 min).
    """
    tool = getattr(sub_action, "tool", None)
    if tool is None:
        return 0.0, []

    penalty = 0.0
    violations: list[str] = []
    contact_name = getattr(sub_action, "contact", None)
    contact = state.contacts.get(contact_name) if contact_name else None

    # --- 1. Spam ---
    if tool in _TEXT_TOOLS and contact is not None:
        if contact.unanswered_agent_messages >= 2:
            penalty += SPAM_PENALTY_PER_EVENT
            violations.append(
                f"spam: {contact_name} has {contact.unanswered_agent_messages} unanswered"
            )

    # --- 2. Time-of-day (voice calls after hours for non-urgent tasks) ---
    if tool in _VOICE_TOOLS and getattr(task, "urgency", "medium") != "high":
        hour = ((state.start_clock_min + state.current_time_min) // 60) % 24
        if hour >= _INAPPROPRIATE_HOUR_START or hour < _INAPPROPRIATE_HOUR_END:
            penalty += TIME_OF_DAY_PENALTY
            violations.append(f"time_of_day: {tool} at hour {hour} for non-urgent task")

    # --- 3. Channel-escalation too fast ---
    # Look back at the last action targeting this same contact: if it was on a different
    # text channel and happened less than that channel's typical reply window ago, the
    # current send is a premature escalation.
    if tool in _TEXT_TOOLS and contact_name:
        prev = None
        for a in reversed(state.action_history):
            if a.args.get("contact") == contact_name and a.tool in _TEXT_TOOLS:
                prev = a
                break
        if prev is not None and prev.tool != tool:
            window = _TYPICAL_REPLY_WINDOW_MIN.get(prev.tool, 15)
            elapsed = state.current_time_min - prev.at_min
            if elapsed < window:
                penalty += CHANNEL_ESCALATION_TOO_FAST
                violations.append(
                    f"escalation_too_fast: {prev.tool}→{tool} after only {elapsed} min"
                )

    return penalty, violations


# ---------------------------------------------------------------------------
# Goal (dense sub-goal scoring)
# ---------------------------------------------------------------------------


def goal_increment(
    state: PhonePilotState, task: Task
) -> tuple[float, set[str]]:
    """Fire any sub-goals newly achieved this step. Returns (delta_reward, newly_fired_keys).

    Sub-goals only fire once; the set is recorded in ``state.sub_goals_fired``.
    """
    delta = 0.0
    newly: set[str] = set()
    scores = task.grade(state)
    for name, weight_earned in scores.items():
        if weight_earned > 0.0 and name not in state.sub_goals_fired:
            state.sub_goals_fired.add(name)
            delta += weight_earned
            newly.add(name)
    return delta, newly


# ---------------------------------------------------------------------------
# Truthfulness — resolves at end_task only
# ---------------------------------------------------------------------------


def truthfulness_penalty(
    state: PhonePilotState,
    task: Task,
    success_claim: bool,
    summary: str,
) -> tuple[float, list[str]]:
    """Evaluate claim + summary against the ground truth in state.

    Returns (penalty, notes). Penalty is ``<= 0``; notes lists the specific violations for
    logging / debugging. If the agent claims ``success=False``, there is no penalty (honest
    failure is not punished).
    """
    penalty = 0.0
    notes: list[str] = []

    is_actually_success = task.is_success(state)

    # Lying about success is the big one.
    if success_claim and not is_actually_success:
        penalty += TRUTHFULNESS_FALSE_CLAIM
        notes.append(f"false_claim: grader says success={is_actually_success}, agent claimed True")

    # Audit the summary for fabricated actions.
    # Each entry is (phrase, allowed_tools). If the phrase appears in the summary but NONE of
    # the allowed tools were actually used in this episode, that's a fabrication. This avoids
    # false positives on generic language ("texted" can mean WhatsApp OR SMS).
    used_tools = {a.tool for a in state.action_history}
    fabrication_patterns: list[tuple[str, set[str]]] = [
        ("called", {"call", "whatsapp_call"}),
        ("phoned", {"call", "whatsapp_call"}),
        ("rang", {"call", "whatsapp_call"}),
        ("whatsapp called", {"whatsapp_call"}),
        ("video called", {"whatsapp_call"}),
        ("whatsapped", {"send_whatsapp"}),
        ("sent a whatsapp", {"send_whatsapp"}),
        ("sent an sms", {"send_sms"}),
        ("smsed", {"send_sms"}),
        ("texted", {"send_whatsapp", "send_sms"}),  # generic text — either channel is fine
        ("emailed", {"send_email"}),
        ("wrote an email", {"send_email"}),
        ("added to calendar", {"calendar_add"}),
        ("scheduled the event", {"calendar_add"}),
        ("scheduled an event", {"calendar_add"}),
        ("booked a slot", {"calendar_add"}),
        ("ordered from zomato", {"zomato_order"}),
        ("placed an order", {"zomato_order"}),
        ("placed the order", {"zomato_order"}),
        ("booked the restaurant", {"zomato_order", "calendar_add"}),
    ]
    lower_summary = summary.lower()
    already_flagged_phrases: set[str] = set()
    for phrase, allowed in fabrication_patterns:
        if phrase in lower_summary and phrase not in already_flagged_phrases:
            if not (allowed & used_tools):
                penalty += TRUTHFULNESS_FABRICATED_SUMMARY
                notes.append(
                    f"fabricated: summary says {phrase!r} but no tool in {sorted(allowed)} was used"
                )
                already_flagged_phrases.add(phrase)

    return penalty, notes


# ---------------------------------------------------------------------------
# Aggregation helper used by env.step
# ---------------------------------------------------------------------------


def apply_per_step(
    state: PhonePilotState,
    task: Task,
    last_action: ActionRecord | None,
    was_format_error: bool,
    appropriateness_pen: float = 0.0,
) -> RewardBreakdown:
    """Compute the reward earned ON THIS STEP ONLY.

    Truthfulness is resolved here too — only non-zero on the step where end_task fires; the
    env passes the ActionRecord with tool == 'end_task' (and args containing success_claim +
    summary) when that happens.

    ``appropriateness_pen`` is computed by the env *before* the action mutates state and
    threaded through so this function stays pure w.r.t. timing.
    """
    rb = RewardBreakdown()
    # Goal increment (dense sub-goals).
    rb.goal, _ = goal_increment(state, task)
    # Efficiency.
    if last_action is not None:
        rb.efficiency = efficiency_step_cost(last_action.tool)
    # Appropriateness (pre-computed).
    rb.appropriateness = appropriateness_pen
    # Format error.
    rb.format = format_step_penalty(was_format_error)
    # Truthfulness (only on end_task).
    if last_action is not None and last_action.tool == "end_task":
        claim = bool(last_action.args.get("success_claim", False))
        summary = str(last_action.args.get("summary", ""))
        pen, _notes = truthfulness_penalty(state, task, claim, summary)
        rb.truthfulness = pen

    # Accumulate on state for run-level logging.
    state.reward_components["goal"] += rb.goal
    state.reward_components["truthfulness"] += rb.truthfulness
    state.reward_components["efficiency"] += rb.efficiency
    state.reward_components["appropriateness"] = (
        state.reward_components.get("appropriateness", 0.0) + rb.appropriateness
    )
    state.reward_components["format"] += rb.format
    state.total_reward += rb.total
    return rb
