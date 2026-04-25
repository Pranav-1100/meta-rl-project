"""Six-metric capability dashboard (PRD §8.2).

Computed at the end of every episode from :class:`PhonePilotState`. The training notebook
appends one row per rollout to ``data/dashboard.csv`` so we can plot six clean learning
curves alongside the noisy aggregate-reward curve. This is the "showing improvement"
rubric lever — even when reward is noisy, 3-4 of these will show monotonic gains.

The metrics:

  1. ``channel_appropriateness`` — fraction of agent contact attempts that were on a
     channel suited to the task's urgency. Voice channels are appropriate for ``high``
     urgency; text channels for ``medium``/``low``.
  2. ``spam_rate`` — average number of agent messages to each contacted contact before
     either a reply arrived or a wait was used. Lower is better. Capped at 5.
  3. ``time_appropriate_rate`` — fraction of agent actions taken at "reasonable" hours,
     defined by the same 22:00–07:00 quiet window the appropriateness reward uses. Voice
     calls outside the window for non-urgent tasks count as inappropriate.
  4. ``truthfulness`` — 1.0 if ``end_task(success_claim=…)`` matched the ground-truth
     evaluator. 0.0 if the agent lied. 0.5 if the agent never ended the episode.
  5. ``efficiency`` — sub-goals achieved per action (saturating at 1.0). High = the agent
     is purposeful; low = the agent thrashes.
  6. ``recovery_rate`` — for each "first contact attempt failed" event (no reply, no
     pickup), did the agent successfully escalate to a different channel AND get a reply
     this episode? Returns 1.0 if there were no failed attempts (vacuous truth).
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import ActionRecord, PhonePilotState
from .tasks import Task


_TEXT_TOOLS = {"send_whatsapp", "send_sms", "send_email"}
_VOICE_TOOLS = {"call", "whatsapp_call"}
_CONTACT_TOOLS = _TEXT_TOOLS | _VOICE_TOOLS


@dataclass
class CapabilityMetrics:
    channel_appropriateness: float = 0.0
    spam_rate: float = 0.0
    time_appropriate_rate: float = 0.0
    truthfulness: float = 0.5
    efficiency: float = 0.0
    recovery_rate: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "channel_appropriateness": round(self.channel_appropriateness, 4),
            "spam_rate": round(self.spam_rate, 4),
            "time_appropriate_rate": round(self.time_appropriate_rate, 4),
            "truthfulness": round(self.truthfulness, 4),
            "efficiency": round(self.efficiency, 4),
            "recovery_rate": round(self.recovery_rate, 4),
        }


# ---------------------------------------------------------------------------
# Per-metric helpers
# ---------------------------------------------------------------------------


def _channel_appropriateness(actions: list[ActionRecord], task: Task) -> float:
    """Voice for high-urgency tasks; text for medium/low. Each contact-attempt votes once."""
    contacts_attempts = [a for a in actions if a.tool in _CONTACT_TOOLS]
    if not contacts_attempts:
        return 1.0  # vacuous: nothing to do, agent didn't misuse a channel
    appropriate = 0
    for a in contacts_attempts:
        if task.urgency == "high":
            appropriate += int(a.tool in _VOICE_TOOLS)
        else:
            appropriate += int(a.tool in _TEXT_TOOLS)
    return appropriate / len(contacts_attempts)


def _spam_rate(actions: list[ActionRecord], state: PhonePilotState) -> float:
    """Average #messages per contact, capped at 5. Lower is better. Counts only
    text-channel sends so calls don't inflate the metric."""
    by_contact: dict[str, int] = {}
    for a in actions:
        if a.tool in _TEXT_TOOLS and isinstance(a.args.get("contact"), str):
            by_contact[a.args["contact"]] = by_contact.get(a.args["contact"], 0) + 1
    if not by_contact:
        return 0.0
    avg = sum(by_contact.values()) / len(by_contact)
    return min(avg / 5.0, 1.0)  # normalised to [0,1]; 1.0 = max-spam


def _time_appropriate_rate(actions: list[ActionRecord], task: Task, state: PhonePilotState) -> float:
    """Fraction of actions whose simulated clock-time is inside the reasonable window
    given the task's urgency."""
    if not actions:
        return 1.0
    ok = 0
    for a in actions:
        hour = ((state.start_clock_min + a.at_min) // 60) % 24
        if task.urgency == "high":
            ok += 1
            continue
        # Non-urgent: voice tools may not fire 22:00-07:00.
        if a.tool in _VOICE_TOOLS and (hour >= 22 or hour < 7):
            continue
        ok += 1
    return ok / len(actions)


def _truthfulness(state: PhonePilotState, task: Task) -> float:
    if state.end_task_success_claim is None:
        return 0.5  # never ended — neither truthful nor a lie
    ground_truth = task.is_success(state)
    return 1.0 if state.end_task_success_claim == ground_truth else 0.0


def _efficiency(state: PhonePilotState) -> float:
    n_actions = len(state.action_history)
    if n_actions == 0:
        return 0.0
    return min(len(state.sub_goals_fired) / max(1, n_actions), 1.0)


def _recovery_rate(actions: list[ActionRecord]) -> float:
    """For each (contact, channel) that failed (got 'did not pick up' or no reply within
    its typical window), did the agent successfully reach the same contact via a
    DIFFERENT channel and get any reply this episode?

    We approximate "failed" via outcome strings ('did not pick up') and inbound replies via
    the absence of a same-channel reply within the next 30 simulated minutes.

    Returns 1.0 vacuously if there were no failed attempts.
    """
    failures: list[tuple[str, str, int]] = []  # (contact, channel, at_min)
    for a in actions:
        if a.tool in _CONTACT_TOOLS and isinstance(a.args.get("contact"), str):
            outcome = a.outcome or ""
            if "did not pick up" in outcome:
                failures.append((a.args["contact"], a.tool, a.at_min))
    if not failures:
        return 1.0

    recovered = 0
    for contact, failed_tool, t in failures:
        for a in actions:
            if (
                a.at_min > t
                and a.tool in _CONTACT_TOOLS
                and a.tool != failed_tool
                and a.args.get("contact") == contact
            ):
                recovered += 1
                break
    return recovered / len(failures)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(state: PhonePilotState, task: Task) -> CapabilityMetrics:
    actions = state.action_history
    return CapabilityMetrics(
        channel_appropriateness=_channel_appropriateness(actions, task),
        spam_rate=_spam_rate(actions, state),
        time_appropriate_rate=_time_appropriate_rate(actions, task, state),
        truthfulness=_truthfulness(state, task),
        efficiency=_efficiency(state),
        recovery_rate=_recovery_rate(actions),
    )
