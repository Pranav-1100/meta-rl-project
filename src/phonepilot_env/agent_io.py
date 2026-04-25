"""Agent ↔ environment text contract.

The LLM we train speaks text. The environment speaks Pydantic. This module is the bridge:

  * :data:`SYSTEM_PROMPT` — the constant system message shown at the top of every rollout,
    describing all 18 tools and the JSON action format the model must emit.
  * :func:`observation_to_prompt` — renders a :class:`PhonePilotObservation` as the plain
    text the model sees each turn.
  * :func:`parse_completion_to_action` — extracts the JSON object from the model's text
    completion and validates it against :class:`PhonePilotAction`.

The invariant: a base model SFT-tuned on ``observation_to_prompt(obs) → completion`` pairs,
where every completion round-trips through ``parse_completion_to_action`` into a valid
:class:`PhonePilotAction`, will emit schema-valid actions at inference time. This is the
warm-start that makes GRPO productive instead of wasting rollouts on format errors.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .actions import PhonePilotAction, TOOL_NAMES
from .observations import PhonePilotObservation


# ---------------------------------------------------------------------------
# System prompt (the training-time + inference-time invariant)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are PhonePilot, a personal assistant running on a simulated smartphone OS. Your job
is to complete the user's request by issuing one tool call per turn. You are talking to
a machine, not the user — every turn you must emit exactly one JSON object describing
the tool you want to run.

## Output format (strict)

Respond with a single JSON object on a single line, wrapped in a ```json code fence:

```json
{"body": {"tool": "<tool_name>", ...args}}
```

No prose outside the code fence. The JSON must match the schema of exactly one tool.
If you want to think before acting, use the `think` tool — it's free and has no side
effects.

## Available tools

**Messaging / calls:**
- `{"tool":"call","contact":"<name>"}` — voice call. Stochastic pickup; low probability
  during work hours.
- `{"tool":"whatsapp_call","contact":"<name>"}` — WhatsApp voice call.
- `{"tool":"hang_up"}` — end the active call.
- `{"tool":"send_whatsapp","contact":"<name>","text":"<message>"}` — WhatsApp text.
  Fastest async channel (median reply ~5 min).
- `{"tool":"send_sms","contact":"<name>","text":"<message>"}` — SMS. Slower (~30 min).
- `{"tool":"read_messages","contact":"<name>","channel":"whatsapp|sms|email"}` — history.
  Both args are optional; leaving them null reads everything.
- `{"tool":"read_notifications"}` — show new incoming messages.

**Calendar:**
- `{"tool":"calendar_view","date":"today"}` — list events.
- `{"tool":"calendar_add","title":"<str>","start_time":"HH:MM","duration_min":60,"invitees":[]}`

**Zomato (food delivery):**
- `{"tool":"zomato_search","query":"<str>","cuisine":null,"veg_only":false,"max_price_per_person":null}`
- `{"tool":"zomato_open","restaurant_id":"<id>"}` — full menu + prices.
- `{"tool":"zomato_order","restaurant_id":"<id>","items":["..."],"delivery_time":"HH:MM|ASAP"}`

**Maps:**
- `{"tool":"maps_search","query":"<str>"}`
- `{"tool":"maps_travel_time","origin":"<str>","destination":"<str>"}`

**Utility:**
- `{"tool":"web_search","query":"<str>"}` — small canned result set.
- `{"tool":"wait","minutes":<1..240>}` — advance simulated time. Use this after sending a
  message to give the contact a chance to reply. DO NOT immediately follow up — contacts
  get annoyed by spam.
- `{"tool":"think","reasoning":"<str>"}` — internal scratchpad. Free, no side effects.
- `{"tool":"end_task","success_claim":<bool>,"summary":"<str>"}` — declare the episode
  finished. Be honest: if you did not actually complete the goal, set `success_claim`
  to false. The environment compares your claim against its internal success check and
  heavily penalises false claims and summaries that cite actions you never took.

## Behaviour rules

1. Pick an appropriate first channel. Urgent tasks → call / whatsapp_call during work
   hours. Non-urgent → whatsapp / sms.
2. After sending a message, WAIT before following up. Spamming is penalised.
3. Escalate channel only after the previous one's typical reply window has passed.
4. Do not call people between 22:00 and 07:00 unless the task is genuinely urgent.
5. Never lie in `end_task`. If the goal was not met, say so.
6. Respond with exactly one JSON object per turn, nothing else.
"""


# Available tool names as plain text (used in some shorter prompt variants).
TOOL_LIST_INLINE = ", ".join(TOOL_NAMES)


# ---------------------------------------------------------------------------
# Observation → text
# ---------------------------------------------------------------------------


def observation_to_prompt(obs: PhonePilotObservation, turn_index: int | None = None) -> str:
    """Render the agent-visible portion of an observation as concise text.

    Keeps the representation short — a small model has a limited context budget, and
    verbose prose wastes tokens. Only fields the agent *needs* to make the next decision
    are shown.
    """
    lines: list[str] = []
    header = f"TURN {turn_index}" if turn_index is not None else "TURN"
    lines.append(f"# {header}  (clock {obs.current_time}, budget left {obs.time_budget_remaining_min} min)")
    lines.append("")
    lines.append(f"GOAL: {obs.user_goal}")
    lines.append("")

    if obs.active_call:
        lines.append(f"ACTIVE_CALL: {obs.active_call}")

    if obs.notifications:
        lines.append("NEW_NOTIFICATIONS:")
        for n in obs.notifications:
            contact = n.contact or "?"
            ch = n.channel or "?"
            lines.append(f"  [{ch}] {contact} @ {n.timestamp}: {n.preview}")
        lines.append("")

    if obs.recent_actions:
        lines.append("RECENT_ACTIONS (most recent last):")
        for a in obs.recent_actions:
            lines.append(f"  {a.at_time}  {a.tool}({a.arg_summary}) → {a.outcome}")
        lines.append("")

    if obs.conversation_summaries:
        # Keep it focused — last message per contact (other than our own echo).
        focused = {k: v for k, v in obs.conversation_summaries.items() if ":you" not in k}
        if focused:
            lines.append("CONVERSATIONS (last msg per contact):")
            for contact, msg in list(focused.items())[:8]:
                lines.append(f"  {contact}: {msg}")
            lines.append("")

    if obs.open_app_view:
        # Compact one-line summary (full dict can be huge).
        app = obs.open_app_view.get("app", "?") if isinstance(obs.open_app_view, dict) else "?"
        lines.append(f"OPEN_APP: {app} ({_compact_dict(obs.open_app_view, max_len=220)})")
        lines.append("")

    if obs.error:
        lines.append(f"ERROR (previous step): {obs.error}")
        lines.append("")

    lines.append("Respond with exactly one JSON tool call inside a ```json fence.")
    return "\n".join(lines)


def _compact_dict(d: Any, max_len: int = 200) -> str:
    s = json.dumps(d, ensure_ascii=False, default=str)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Text → action
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{(?:[^{}]|\{[^{}]*\})*\})", re.DOTALL)


class AgentParseError(ValueError):
    """Raised when the model's completion can't be coerced into a valid action."""


def parse_completion_to_action(completion: str) -> PhonePilotAction:
    """Extract a single JSON object from the model's completion and validate it.

    Accepts three forms (most to least strict):
      1. `` ```json\\n{...}\\n``` `` — canonical fenced block.
      2. ``{...}`` — bare JSON object (last one in the string wins if multiple).
      3. Trailing best-effort: if the string starts with ``{`` and ends with ``}``, try it
         as-is.

    Raises :class:`AgentParseError` with a message suitable to log alongside the bad
    completion.
    """
    if completion is None:
        raise AgentParseError("empty completion")

    raw_json: str | None = None

    fenced = _JSON_FENCE_RE.findall(completion)
    if fenced:
        raw_json = fenced[-1].strip()
    else:
        bare = _BARE_JSON_RE.findall(completion)
        if bare:
            raw_json = bare[-1].strip()

    if raw_json is None:
        raise AgentParseError("no JSON object found in completion")

    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise AgentParseError(f"invalid JSON: {e.msg}") from e

    # Auto-upgrade bare sub-action shape: {"tool": "..."} → {"body": {"tool": "..."}}.
    if isinstance(obj, dict) and "body" not in obj and "tool" in obj:
        obj = {"body": obj}

    try:
        return PhonePilotAction.model_validate(obj)
    except ValidationError as e:
        raise AgentParseError(f"schema validation failed: {e.errors()[:2]}") from e


# ---------------------------------------------------------------------------
# Action → training-completion text (round-trip)
# ---------------------------------------------------------------------------


def action_to_completion(action: PhonePilotAction) -> str:
    """Serialise a :class:`PhonePilotAction` back to the exact text the model should emit.

    Used by the synthetic-trajectory generator so every training example's completion is
    parseable by :func:`parse_completion_to_action`.
    """
    body = action.body.model_dump(exclude={"metadata"})
    return "```json\n" + json.dumps({"body": body}, ensure_ascii=False) + "\n```"
