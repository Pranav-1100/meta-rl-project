"""Action parsing + discriminated-union validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phonepilot_env.actions import (
    ACTION_REGISTRY,
    EndTaskAction,
    PhonePilotAction,
    SendWhatsAppAction,
    TOOL_NAMES,
)


def test_tool_count_matches_prd_full_scope():
    # Phase 2 brings us to the PRD §4.2 full 23-tool surface area.
    assert len(TOOL_NAMES) == 23
    # Spot-check representative tools from each category.
    for t in (
        "send_whatsapp",
        "send_email",
        "call",
        "calendar_add",
        "calendar_reschedule",
        "zomato_order",
        "swiggy_order",
        "end_task",
        "wait",
        "think",
    ):
        assert t in TOOL_NAMES


def test_parse_send_whatsapp():
    act = PhonePilotAction.model_validate(
        {"body": {"tool": "send_whatsapp", "contact": "Ria", "text": "late"}}
    )
    assert isinstance(act.body, SendWhatsAppAction)
    assert act.body.contact == "Ria"


def test_parse_end_task():
    act = PhonePilotAction.model_validate(
        {"body": {"tool": "end_task", "success_claim": True, "summary": "done"}}
    )
    assert isinstance(act.body, EndTaskAction)
    assert act.body.success_claim is True
    # Confidence defaults to 'medium' for backward compat with pre-Phase-1 callers.
    assert act.body.confidence == "medium"


def test_parse_end_task_with_confidence():
    """Phase 1: end_task accepts a confidence ∈ {low, medium, high}."""
    for level in ("low", "medium", "high"):
        act = PhonePilotAction.model_validate(
            {"body": {
                "tool": "end_task",
                "success_claim": False,
                "summary": "couldn't reach",
                "confidence": level,
            }}
        )
        assert isinstance(act.body, EndTaskAction)
        assert act.body.confidence == level


def test_rejects_invalid_confidence_value():
    """Phase 1: confidence must be one of low/medium/high; 'sure' or 0.9 should reject."""
    for bad in ("sure", "very_high", "0.9", "", 0.9):
        with pytest.raises(ValidationError):
            PhonePilotAction.model_validate(
                {"body": {
                    "tool": "end_task",
                    "success_claim": True,
                    "summary": "done",
                    "confidence": bad,
                }}
            )


def test_rejects_unknown_tool():
    with pytest.raises(ValidationError):
        PhonePilotAction.model_validate({"body": {"tool": "summon_uber"}})


def test_rejects_missing_required_fields():
    # send_whatsapp requires both contact and text.
    with pytest.raises(ValidationError):
        PhonePilotAction.model_validate({"body": {"tool": "send_whatsapp", "contact": "Jay"}})


def test_registry_covers_all_tools():
    assert set(ACTION_REGISTRY.keys()) == set(TOOL_NAMES)
