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


def test_tool_count_matches_prd_v1_scope():
    assert len(TOOL_NAMES) == 18
    # Spot-check representative tools from each category.
    for t in ("send_whatsapp", "call", "calendar_add", "zomato_order", "end_task", "wait", "think"):
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


def test_rejects_unknown_tool():
    with pytest.raises(ValidationError):
        PhonePilotAction.model_validate({"body": {"tool": "summon_uber"}})


def test_rejects_missing_required_fields():
    # send_whatsapp requires both contact and text.
    with pytest.raises(ValidationError):
        PhonePilotAction.model_validate({"body": {"tool": "send_whatsapp", "contact": "Jay"}})


def test_registry_covers_all_tools():
    assert set(ACTION_REGISTRY.keys()) == set(TOOL_NAMES)
