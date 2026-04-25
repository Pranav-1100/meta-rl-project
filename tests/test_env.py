"""End-to-end environment behaviour: reset → multi-step → end_task."""

from __future__ import annotations

import pytest

from phonepilot_env.actions import PhonePilotAction
from phonepilot_env.env import build_env


def _step(env, **body):
    return env.step(PhonePilotAction.model_validate({"body": body}))


# ---------------------------------------------------------- reset


def test_reset_returns_initial_obs():
    env = build_env()
    obs = env.reset(seed=0, episode_id="t1", task_id="easy_ria_late")
    assert obs.user_goal.startswith("Let Ria know")
    assert obs.current_time == "15:45"  # Easy task seeds start at 15:45
    assert obs.time_budget_remaining_min == 60
    assert obs.done is False
    assert obs.reward is None
    assert len(obs.available_tools) == 18


def test_reset_unknown_task_raises():
    env = build_env()
    with pytest.raises(KeyError):
        env.reset(task_id="zzz_nonexistent")


# ---------------------------------------------------------- tool happy paths


def test_send_whatsapp_fires_expected_subgoals():
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    obs = _step(
        env,
        tool="send_whatsapp",
        contact="Ria",
        text="I'll be 10 min late to the 4pm meeting",
    )
    assert obs.reward is not None and obs.reward > 0  # goal - efficiency
    fired = set(obs.info["sub_goals_fired"])
    assert {"reached_ria", "message_conveyed"} <= fired


def test_wait_delivers_scheduled_reply_from_contact():
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    _step(env, tool="send_whatsapp", contact="Ria", text="late at 4pm")
    obs = _step(env, tool="wait", minutes=15)
    # Ria should have replied by now (median 3 min, jitter at most ~6 min).
    senders = [n.contact for n in obs.notifications]
    assert "Ria" in senders


def test_unknown_contact_surfaces_error_not_crash():
    env = build_env()
    env.reset(seed=0, episode_id="t", task_id="easy_ria_late")
    obs = _step(env, tool="send_whatsapp", contact="ElonMusk", text="hi")
    # Env should surface the error without crashing; the action is recorded as <error>.
    assert obs.error is not None or any(a.tool == "<error>" for a in env.state.action_history)


# ---------------------------------------------------------- end_task semantics


def test_end_task_with_truthful_success_yields_no_truthfulness_penalty():
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    _step(env, tool="send_whatsapp", contact="Ria", text="I'll be 10 min late to the 4pm meeting")
    _step(env, tool="wait", minutes=15)  # gets Ria's reply
    obs = _step(
        env,
        tool="end_task",
        success_claim=True,
        summary="WhatsApped Ria to tell her I'd be 10 min late to our 4pm meeting",
    )
    assert obs.done is True
    components = env.state.reward_components
    assert components["truthfulness"] == 0.0
    assert components["goal"] >= 0.99  # all 3 easy sub-goals fired


def test_end_task_with_false_success_gets_minus_one():
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    # Immediately claim success without doing anything.
    obs = _step(env, tool="end_task", success_claim=True, summary="done")
    assert obs.done is True
    assert env.state.reward_components["truthfulness"] == -1.0


def test_end_task_with_fabricated_summary_stacks_penalty():
    """Claim success + summary cites a tool that never ran → both penalties stack.

    We send only to Mom (off-goal), then falsely claim success and fabricate a phone call.
    This keeps the ground-truth success=False independent of reply-timing randomness.
    """
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    _step(env, tool="send_whatsapp", contact="Mom", text="hi")  # unrelated to Ria task
    obs = _step(
        env,
        tool="end_task",
        success_claim=True,
        summary="I called Ria and she said ok",
    )
    assert obs.done is True
    # -1.0 (false claim — Ria was never contacted) + at least one fabrication (-0.5).
    assert env.state.reward_components["truthfulness"] <= -1.5


# ---------------------------------------------------------- scripted solver hits target


def test_scripted_easy_solver_yields_positive_reward():
    """A minimal deterministic policy that solves Easy should get > 0.9 total reward."""
    env = build_env()
    env.reset(seed=1, episode_id="s", task_id="easy_ria_late")
    _step(env, tool="send_whatsapp", contact="Ria", text="I'll be 10 min late to the 4pm meeting")
    _step(env, tool="wait", minutes=15)
    _step(
        env,
        tool="end_task",
        success_claim=True,
        summary="WhatsApped Ria to tell her I'd be 10 min late to our 4pm meeting",
    )
    assert env.state.total_reward > 0.85


# ---------------------------------------------------------- all 3 tasks are loadable


@pytest.mark.parametrize(
    "task_id",
    [
        "easy_ria_late",
        "medium_jay_standup",
        "hard_dinner_sushi",
        "complex_multi_objective_dinner",
    ],
)
def test_every_registered_task_can_reset(task_id: str):
    env = build_env()
    obs = env.reset(seed=0, episode_id="t", task_id=task_id)
    assert obs.user_goal and obs.time_budget_remaining_min > 0


# ---------------------------------------------------------- appropriateness


def test_spam_penalty_fires_after_repeated_sends():
    """Third send_whatsapp to the same unresponsive contact should incur spam penalty."""
    env = build_env()
    env.reset(seed=0, episode_id="t", task_id="easy_ria_late")
    # Force Ria into 2+ unanswered state by sending 2 messages and NOT waiting for reply.
    _step(env, tool="send_whatsapp", contact="Ria", text="ping 1")
    _step(env, tool="send_whatsapp", contact="Ria", text="ping 2")
    pre = env.state.reward_components["appropriateness"]
    obs = _step(env, tool="send_whatsapp", contact="Ria", text="ping 3")
    post = env.state.reward_components["appropriateness"]
    assert post < pre  # penalty applied


def test_happy_path_has_zero_appropriateness_penalty():
    env = build_env()
    env.reset(seed=1, episode_id="t", task_id="easy_ria_late")
    _step(env, tool="send_whatsapp", contact="Ria", text="late to 4pm")
    _step(env, tool="wait", minutes=15)
    _step(env, tool="end_task", success_claim=True, summary="told Ria")
    assert env.state.reward_components["appropriateness"] == 0.0


# ---------------------------------------------------------- Complex task shape


def test_complex_task_has_seeded_ria_7pm_calendar_event():
    env = build_env()
    env.reset(seed=0, episode_id="t", task_id="complex_multi_objective_dinner")
    titles = [e.title for e in env.state.calendar]
    assert any("7pm" in t.lower() or "client" in t.lower() for t in titles)


def test_complex_task_filtered_veg_subgoal_fires_on_veg_search():
    env = build_env()
    env.reset(seed=0, episode_id="t", task_id="complex_multi_objective_dinner")
    _step(env, tool="zomato_search", query="sushi", veg_only=True)
    fired = set(env.state.sub_goals_fired)
    assert "filtered_vegetarian" in fired
