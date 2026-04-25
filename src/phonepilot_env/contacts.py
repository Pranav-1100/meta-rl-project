"""Contact simulator — stochastic pickups, delayed replies, template-driven persona voice.

Drives two distinct things:

1. **Timing / likelihood** of a contact responding — governed by their ``ContactProfile``.
2. **Text content** of the reply — template-based in v1 (deterministic, free, unit-testable).
   A ``ClaudeReplyGenerator`` hook exists for v2 when we want richer demo dialogue.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .state import ContactProfile, MessageEvent, PendingReply

if TYPE_CHECKING:
    from .state import PhonePilotState


# ---------------------------------------------------------------------------
# Default personas
# ---------------------------------------------------------------------------


def default_contacts() -> dict[str, ContactProfile]:
    """Four personas spanning the scenarios the PRD's tasks need."""
    return {
        "Jay": ContactProfile(
            name="Jay",
            call_pickup_prob_work_hours=0.30,
            call_pickup_prob_after_hours=0.85,
            whatsapp_reply_median_min=4,
            sms_reply_median_min=25,
            email_reply_median_min=360,
            preferred_channel="whatsapp",
            annoyance_threshold=3,
            dietary="vegetarian",
            location="Indiranagar",
        ),
        "Ria": ContactProfile(
            name="Ria",
            call_pickup_prob_work_hours=0.55,
            call_pickup_prob_after_hours=0.70,
            whatsapp_reply_median_min=3,
            sms_reply_median_min=15,
            email_reply_median_min=240,
            preferred_channel="whatsapp",
            annoyance_threshold=4,
            dietary="any",
            location="Koramangala",
        ),
        "Mira": ContactProfile(
            name="Mira",
            call_pickup_prob_work_hours=0.40,
            call_pickup_prob_after_hours=0.60,
            whatsapp_reply_median_min=8,
            sms_reply_median_min=40,
            email_reply_median_min=480,
            preferred_channel="whatsapp",
            annoyance_threshold=3,
            dietary="any",
            location="Whitefield",  # ~15km from Koramangala — used by Complex task
        ),
        "Mom": ContactProfile(
            name="Mom",
            call_pickup_prob_work_hours=0.90,
            call_pickup_prob_after_hours=0.95,
            whatsapp_reply_median_min=6,
            sms_reply_median_min=12,
            email_reply_median_min=180,
            preferred_channel="call",
            annoyance_threshold=5,
            dietary="vegetarian",
            location="Jayanagar",
        ),
    }


# ---------------------------------------------------------------------------
# Call pickup probability
# ---------------------------------------------------------------------------


def pickup_probability(profile: ContactProfile, is_work_hours: bool) -> float:
    base = (
        profile.call_pickup_prob_work_hours
        if is_work_hours
        else profile.call_pickup_prob_after_hours
    )
    # Annoyed contacts are less likely to engage.
    if profile.unanswered_agent_messages >= profile.annoyance_threshold:
        base *= 0.4
    return max(0.0, min(1.0, base))


def roll_pickup(profile: ContactProfile, is_work_hours: bool, rng: random.Random) -> bool:
    return rng.random() < pickup_probability(profile, is_work_hours)


# ---------------------------------------------------------------------------
# Message reply scheduling
# ---------------------------------------------------------------------------


# Expressed as a jitter band around the median — reply shows up at a random time in
# ``[0.5*median, 2.0*median]`` to give RL a bit of stochasticity without being wild.
_JITTER_LO = 0.5
_JITTER_HI = 2.0


def _reply_median_for(profile: ContactProfile, channel: str) -> int:
    return {
        "whatsapp": profile.whatsapp_reply_median_min,
        "sms": profile.sms_reply_median_min,
        "email": profile.email_reply_median_min,
    }.get(channel, profile.whatsapp_reply_median_min)


def schedule_reply(
    state: "PhonePilotState",
    profile: ContactProfile,
    channel: str,
    incoming_text: str,
    rng: random.Random,
) -> PendingReply | None:
    """Decide whether + when this contact replies, and with what text.

    Returns the scheduled reply (also appends it to ``state.pending_replies``), or ``None``
    if the contact won't reply at all (high annoyance, out of band, etc.).
    """
    # If the contact has been pinged too many times with no response from them, they stop.
    profile.unanswered_agent_messages += 1
    will_ignore = (
        profile.unanswered_agent_messages > profile.annoyance_threshold
        and rng.random() < 0.6
    )
    if will_ignore:
        return None

    median = _reply_median_for(profile, channel)
    jitter = rng.uniform(_JITTER_LO, _JITTER_HI)
    delay_min = max(1, int(round(median * jitter)))

    reply_text = _render_reply(profile, channel, incoming_text, state, rng)
    pending = PendingReply(
        from_contact=profile.name,
        channel=channel,  # type: ignore[arg-type]
        text=reply_text,
        at_min=state.current_time_min + delay_min,
    )
    state.pending_replies.append(pending)
    return pending


# ---------------------------------------------------------------------------
# Template reply generator
# ---------------------------------------------------------------------------


def _render_reply(
    profile: ContactProfile,
    channel: str,
    incoming_text: str,
    state: "PhonePilotState",
    rng: random.Random,
) -> str:
    """Generate persona-consistent reply text using keyword-driven templates."""
    lower = incoming_text.lower()

    # --- Acknowledgement of "I'll be late" style ---
    if any(w in lower for w in ("late", "delay", "delayed", "running behind", "held up")):
        options = {
            "Jay": ["got it, no rush", "np, see you soon", "all good, take your time"],
            "Ria": ["ok, thanks for the heads up!", "got it", "cool, see you when you're here"],
            "Mira": ["ok", "no worries", "sure"],
            "Mom": ["okay beta, drive safe", "thanks for telling me", "no problem"],
        }.get(profile.name, ["ok", "got it"])
        return rng.choice(options)

    # --- "Can you join / hop on the standup / call?" ---
    if any(w in lower for w in ("standup", "call", "meeting", "join", "hop on", "dial in")):
        if profile.name == "Jay" and "urgent" in lower:
            return rng.choice(["on my way", "joining in 2", "yes dialing in now"])
        if profile.name == "Jay":
            return rng.choice(["sure, one sec", "give me 5 min", "ok joining"])
        return rng.choice(["yes, joining", "on it", "ok"])

    # --- Dinner invites / confirmations ---
    if any(w in lower for w in ("dinner", "sushi", "restaurant", "eat", "meal", "drinks")):
        profile.will_attend_dinner = True
        return rng.choice(["sounds good, I'm in!", "yes count me in", "confirmed — see you there"])

    # --- Greetings / check-ins ---
    if any(w in lower for w in ("hey", "hi ", "hello", "yo")):
        return rng.choice(["hey!", "hi :)", "yo"])

    # --- Fallback (acknowledgement) ---
    return rng.choice(["ok", "got it", "sure", "noted"])


# ---------------------------------------------------------------------------
# Fire pending replies that are now due
# ---------------------------------------------------------------------------


def flush_due_replies(state: "PhonePilotState") -> list[MessageEvent]:
    """Move every :class:`PendingReply` whose ``at_min`` ≤ now into ``state.messages``.

    Returns the newly-delivered :class:`MessageEvent` list so the env can surface them as
    notifications and update conversation summaries. Also resets the sender's
    ``unanswered_agent_messages`` counter (a reply is proof they're not ignoring).
    """
    now = state.current_time_min
    delivered: list[MessageEvent] = []
    still_pending: list[PendingReply] = []
    for pr in state.pending_replies:
        if pr.at_min <= now:
            ev = MessageEvent(
                sender=pr.from_contact,
                recipient="user",
                channel=pr.channel,
                text=pr.text,
                sent_at_min=pr.at_min,
            )
            state.messages.append(ev)
            delivered.append(ev)
            if pr.from_contact in state.contacts:
                state.contacts[pr.from_contact].unanswered_agent_messages = 0
        else:
            still_pending.append(pr)
    state.pending_replies = still_pending
    return delivered
