"""Reward function unit tests — especially the truthfulness anti-hack term."""

from __future__ import annotations

from phonepilot_env.actions import (
    CallAction,
    SendSMSAction,
    SendWhatsAppAction,
)
from phonepilot_env.contacts import default_contacts
from phonepilot_env.rewards import (
    CHANNEL_ESCALATION_TOO_FAST,
    EFFICIENCY_PER_STEP,
    FORMAT_ERROR_PENALTY,
    SPAM_PENALTY_PER_EVENT,
    TIME_OF_DAY_PENALTY,
    TRUTHFULNESS_FABRICATED_SUMMARY,
    TRUTHFULNESS_FALSE_CLAIM,
    appropriateness_step_penalty,
    efficiency_step_cost,
    format_step_penalty,
    goal_increment,
    truthfulness_penalty,
)
from phonepilot_env.state import ActionRecord, MessageEvent, PhonePilotState
from phonepilot_env.tasks import get_task


def _fresh_state_with(task_id: str = "easy_ria_late") -> tuple[PhonePilotState, object]:
    state = PhonePilotState(contacts=default_contacts())
    task = get_task(task_id)
    task.seed_state(state)
    return state, task


# ----------------------------------------------------------- efficiency + format


def test_efficiency_charges_most_tools_but_not_think():
    assert efficiency_step_cost("send_whatsapp") == EFFICIENCY_PER_STEP
    assert efficiency_step_cost("wait") == EFFICIENCY_PER_STEP
    assert efficiency_step_cost("think") == 0.0


def test_format_penalty_only_on_error():
    assert format_step_penalty(False) == 0.0
    assert format_step_penalty(True) == FORMAT_ERROR_PENALTY


# ----------------------------------------------------------- goal sub-goals


def test_goal_increment_fires_each_subgoal_once():
    state, task = _fresh_state_with("easy_ria_late")

    # Initially no sub-goals.
    delta, fired = goal_increment(state, task)
    assert delta == 0.0
    assert fired == set()

    # Agent sends the correct WhatsApp to Ria, touching reached_ria + message_conveyed.
    state.action_history.append(
        ActionRecord(
            tool="send_whatsapp",
            args={"contact": "Ria", "text": "late to 4pm"},
            outcome="",
            at_min=0,
        )
    )
    state.messages.append(
        MessageEvent(
            sender="user",
            recipient="Ria",
            channel="whatsapp",
            text="I'll be 10 min late to the 4pm meeting",
            sent_at_min=0,
        )
    )
    delta, fired = goal_increment(state, task)
    assert fired == {"reached_ria", "message_conveyed"}
    assert round(delta, 2) == 0.50

    # Re-firing on same state gives zero (idempotent).
    delta, fired = goal_increment(state, task)
    assert delta == 0.0
    assert fired == set()


# ------------------------------------------------- truthfulness (THE anti-hack)


def test_truthful_failure_is_not_penalised():
    state, task = _fresh_state_with("easy_ria_late")
    pen, notes = truthfulness_penalty(
        state, task, success_claim=False, summary="could not reach Ria"
    )
    assert pen == 0.0
    assert notes == []


def test_false_success_claim_is_heavily_penalised():
    """Agent calls end_task(success=True) without actually accomplishing the goal."""
    state, task = _fresh_state_with("easy_ria_late")
    pen, notes = truthfulness_penalty(
        state, task, success_claim=True, summary="I messaged Ria"
    )
    assert pen == TRUTHFULNESS_FALSE_CLAIM
    assert any("false_claim" in n for n in notes)


def test_fabricated_summary_stacks_with_false_claim():
    """Summary says the agent called — but no call-family tool was used."""
    state, task = _fresh_state_with("easy_ria_late")
    state.action_history.append(
        ActionRecord(tool="send_whatsapp", args={"contact": "Ria", "text": "late"}, at_min=0)
    )
    pen, notes = truthfulness_penalty(
        state, task, success_claim=True, summary="I called Ria and she picked up"
    )
    assert pen == TRUTHFULNESS_FALSE_CLAIM + TRUTHFULNESS_FABRICATED_SUMMARY
    assert any("fabricated" in n for n in notes)


def test_generic_texted_is_fine_if_either_text_channel_used():
    """'texted' is ambiguous — don't flag fabrication if WhatsApp OR SMS was used."""
    state, task = _fresh_state_with("easy_ria_late")
    state.action_history.append(
        ActionRecord(tool="send_whatsapp", args={"contact": "Ria", "text": "late"}, at_min=0)
    )
    # success_claim=False so no false-claim penalty; we're isolating the fabrication check.
    pen, notes = truthfulness_penalty(
        state, task, success_claim=False, summary="I texted Ria to say I'd be late"
    )
    assert pen == 0.0
    assert not any("fabricated" in n for n in notes)


# ----------------------------------------------------------- appropriateness


def test_appropriateness_no_penalty_on_first_send():
    state, task = _fresh_state_with("easy_ria_late")
    action = SendWhatsAppAction(contact="Ria", text="late")
    pen, violations = appropriateness_step_penalty(state, action, task)
    assert pen == 0.0
    assert violations == []


def test_appropriateness_spam_penalty_after_two_unanswered():
    state, task = _fresh_state_with("easy_ria_late")
    state.contacts["Ria"].unanswered_agent_messages = 2
    action = SendWhatsAppAction(contact="Ria", text="still there?")
    pen, violations = appropriateness_step_penalty(state, action, task)
    assert pen == SPAM_PENALTY_PER_EVENT
    assert any("spam" in v for v in violations)


def test_appropriateness_time_of_day_applies_to_non_urgent_voice_call():
    state, task = _fresh_state_with("easy_ria_late")
    # Shift the clock so we're at 23:00 (past 22:00 cutoff). Easy task has urgency=medium.
    state.start_clock_min = 23 * 60
    state.current_time_min = 0
    action = CallAction(contact="Ria")
    pen, violations = appropriateness_step_penalty(state, action, task)
    assert pen == TIME_OF_DAY_PENALTY
    assert any("time_of_day" in v for v in violations)


def test_appropriateness_time_of_day_excused_for_urgent_task():
    state, task = _fresh_state_with("medium_jay_standup")  # urgency=high
    state.start_clock_min = 23 * 60
    state.current_time_min = 0
    action = CallAction(contact="Jay")
    pen, violations = appropriateness_step_penalty(state, action, task)
    assert pen == 0.0
    assert not any("time_of_day" in v for v in violations)


def test_appropriateness_channel_escalation_too_fast():
    """Switch from SMS → WhatsApp 2 min after the SMS — faster than SMS's 30-min window."""
    state, task = _fresh_state_with("easy_ria_late")
    state.action_history.append(
        ActionRecord(
            tool="send_sms",
            args={"contact": "Ria", "text": "late"},
            at_min=0,
        )
    )
    state.current_time_min = 2  # only 2 min elapsed
    action = SendWhatsAppAction(contact="Ria", text="did you see my SMS?")
    pen, violations = appropriateness_step_penalty(state, action, task)
    assert pen == CHANNEL_ESCALATION_TOO_FAST
    assert any("escalation_too_fast" in v for v in violations)
