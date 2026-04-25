"""Agent I/O round-trip + robustness tests.

This file is important: SFT and GRPO training both depend on every assistant completion
being parseable back into a :class:`PhonePilotAction`. If the round-trip ever breaks
silently, we waste hours of GPU time on a model that learns to emit garbage.
"""

from __future__ import annotations

import pytest

from phonepilot_env.actions import (
    CallAction,
    EndTaskAction,
    PhonePilotAction,
    SendWhatsAppAction,
    WaitAction,
    ZomatoOrderAction,
)
from phonepilot_env.agent_io import (
    SYSTEM_PROMPT,
    AgentParseError,
    action_to_completion,
    observation_to_prompt,
    parse_completion_to_action,
)
from phonepilot_env.env import build_env


# ---------------------------------------------------------- round-trip


@pytest.mark.parametrize(
    "sub",
    [
        CallAction(contact="Jay"),
        SendWhatsAppAction(contact="Ria", text="I'll be 10 min late"),
        WaitAction(minutes=10),
        EndTaskAction(success_claim=True, summary="done"),
        ZomatoOrderAction(
            restaurant_id="z_sushi_haven",
            items=["Veg Maki Platter"],
            delivery_time="20:00",
        ),
    ],
)
def test_action_to_completion_round_trip(sub):
    action = PhonePilotAction(body=sub)
    completion = action_to_completion(action)
    parsed = parse_completion_to_action(completion)
    assert type(parsed.body) is type(sub)
    assert parsed.body.model_dump(exclude={"metadata"}) == sub.model_dump(
        exclude={"metadata"}
    )


# ---------------------------------------------------------- robustness of parser


def test_parser_accepts_bare_body_shape():
    action = parse_completion_to_action(
        '```json\n{"body": {"tool": "wait", "minutes": 5}}\n```'
    )
    assert action.body.tool == "wait"


def test_parser_auto_wraps_bare_sub_action():
    # Sometimes small models skip the {"body": ...} wrapper.
    action = parse_completion_to_action('```json\n{"tool": "wait", "minutes": 5}\n```')
    assert action.body.tool == "wait"


def test_parser_accepts_unfenced_json():
    action = parse_completion_to_action(
        'Thinking first...\n{"body": {"tool": "wait", "minutes": 1}}'
    )
    assert action.body.tool == "wait"


def test_parser_rejects_no_json():
    with pytest.raises(AgentParseError):
        parse_completion_to_action("I'm going to call Ria now.")


def test_parser_rejects_malformed_json():
    with pytest.raises(AgentParseError):
        parse_completion_to_action('```json\n{"tool": "wait", "minutes":}\n```')


def test_parser_rejects_unknown_tool():
    with pytest.raises(AgentParseError):
        parse_completion_to_action(
            '```json\n{"body": {"tool": "summon_uber", "destination": "moon"}}\n```'
        )


# ---------------------------------------------------------- observation rendering


def test_observation_to_prompt_contains_goal_and_clock():
    env = build_env()
    obs = env.reset(seed=0, episode_id="t", task_id="easy_ria_late")
    rendered = observation_to_prompt(obs, turn_index=0)
    assert "GOAL:" in rendered
    assert "Let Ria know" in rendered
    assert "15:45" in rendered  # Easy task starts at 15:45
    assert "Respond with exactly one JSON" in rendered


def test_system_prompt_mentions_all_tools():
    # Every tool name should be referenced in the system prompt so the model knows about it.
    for tool in (
        "call",
        "send_whatsapp",
        "wait",
        "end_task",
        "zomato_search",
        "maps_travel_time",
        "think",
    ):
        assert tool in SYSTEM_PROMPT
