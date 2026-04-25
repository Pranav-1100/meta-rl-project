"""PhonePilot environment — the Gym-style heart of the OpenEnv submission.

Implements :class:`openenv.core.Environment[PhonePilotAction, PhonePilotObservation,
PhonePilotState]`:

  * ``reset(seed, episode_id, task_id=None)`` — seeds state + returns the first observation.
  * ``step(action)`` — dispatches to the right tool handler, advances simulated time, fires
    any replies whose delay has elapsed, computes a reward, and returns the next observation.
  * ``state`` — read-only snapshot of hidden state (used by the server's /state route and by
    tests).

Kept intentionally linear: one file, clear handler functions, reward resolved via the
helpers in :mod:`phonepilot_env.rewards`. Stochastic pickups/reply delays are driven by a
seedable :class:`random.Random` so episodes are reproducible per ``episode_id``.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from openenv.core import Environment

from .actions import (
    CalendarAddAction,
    CalendarViewAction,
    CallAction,
    EndTaskAction,
    HangUpAction,
    MapsSearchAction,
    MapsTravelTimeAction,
    PhonePilotAction,
    ReadMessagesAction,
    ReadNotificationsAction,
    SendSMSAction,
    SendWhatsAppAction,
    ThinkAction,
    WaitAction,
    WebSearchAction,
    WhatsAppCallAction,
    ZomatoOpenAction,
    ZomatoOrderAction,
    ZomatoSearchAction,
)
from .apps import (
    calendar_add,
    calendar_view,
    maps_search,
    maps_travel_time,
    web_search,
    zomato_open,
    zomato_order,
    zomato_search,
)
from .contacts import (
    default_contacts,
    flush_due_replies,
    roll_pickup,
    schedule_reply,
)
from .observations import ActionOutcome, Notification, PhonePilotObservation
from .rewards import (
    MAX_FORMAT_ERROR_STREAK,
    RewardBreakdown,
    apply_per_step,
    appropriateness_step_penalty,
)
from .state import ActionRecord, CalendarEvent, MessageEvent, Order, PhonePilotState
from .tasks import TASK_REGISTRY, Task, default_task_id, get_task


class PhonePilotEnvironment(
    Environment[PhonePilotAction, PhonePilotObservation, PhonePilotState]
):
    """The PhonePilot simulated phone."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        # Lazy — real state is created in reset().
        self._state: PhonePilotState = PhonePilotState()
        self._task: Task = get_task(default_task_id())
        self._rng: random.Random = random.Random(0)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> PhonePilotState:
        return self._state

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> PhonePilotObservation:
        task_id = kwargs.get("task_id") or default_task_id()
        self._task = get_task(task_id)
        self._rng = random.Random(seed if seed is not None else abs(hash(episode_id or "default")))

        self._state = PhonePilotState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            contacts=default_contacts(),
            time_budget_min=self._task.time_budget_min,
            active_task_id=task_id,
        )
        # Task-specific seeding (calendar entries, pre-existing messages, etc).
        self._task.seed_state(self._state)

        return self._build_observation(newly_delivered=[], last_outcome=None, format_error=None)

    # ------------------------------------------------------------------ step

    def step(
        self,
        action: PhonePilotAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> PhonePilotObservation:
        self._state.step_count += 1

        # Dispatch to handler. Format errors shouldn't happen here because Pydantic
        # already validated at the server boundary, but we guard anyway.
        sub = getattr(action, "body", None)
        if sub is None:
            return self._handle_format_error("action.body is missing")

        # Compute appropriateness penalty BEFORE the action mutates state — spam etc. are
        # judged against what the agent saw at decision time.
        appr_pen, _violations = appropriateness_step_penalty(self._state, sub, self._task)

        try:
            outcome, advance_min = self._dispatch(sub)
        except ValueError as e:
            # Handler-reported semantic error — treat as a non-terminal soft failure (no format
            # penalty, but surface the message to the agent).
            return self._surface_error(str(e))

        # Record the action.
        args = sub.model_dump(exclude={"tool", "metadata"})
        rec = ActionRecord(
            tool=sub.tool,
            args=args,
            outcome=outcome,
            at_min=self._state.current_time_min,
        )
        self._state.action_history.append(rec)
        self._state.format_error_streak = 0

        # Advance simulated time (except for `think`).
        self._state.advance_time(advance_min)

        # Flush any contact replies that are now due. Their MessageEvents will become
        # notifications on the next observation.
        newly_delivered = flush_due_replies(self._state)

        # If the tool was end_task, terminate the episode.
        if isinstance(sub, EndTaskAction):
            self._state.terminated = True
            self._state.end_task_success_claim = sub.success_claim
            self._state.end_task_summary = sub.summary

        # Check budget.
        if self._state.current_time_min >= self._state.time_budget_min and not self._state.terminated:
            self._state.terminated = True

        # Compute reward for this step.
        breakdown = apply_per_step(
            self._state,
            self._task,
            last_action=rec,
            was_format_error=False,
            appropriateness_pen=appr_pen,
        )

        return self._build_observation(
            newly_delivered=newly_delivered,
            last_outcome=rec,
            format_error=None,
            breakdown=breakdown,
        )

    # ----------------------------------------------------------- tool dispatch

    def _dispatch(self, sub: Any) -> tuple[str, int]:
        """Return (outcome_string, simulated_minutes_consumed)."""
        if isinstance(sub, CallAction):
            return self._do_call(sub.contact, via="call")
        if isinstance(sub, WhatsAppCallAction):
            return self._do_call(sub.contact, via="whatsapp_call")
        if isinstance(sub, HangUpAction):
            return self._do_hang_up()
        if isinstance(sub, SendWhatsAppAction):
            return self._do_send(sub.contact, sub.text, channel="whatsapp")
        if isinstance(sub, SendSMSAction):
            return self._do_send(sub.contact, sub.text, channel="sms")
        if isinstance(sub, ReadMessagesAction):
            return self._do_read_messages(sub.contact, sub.channel)
        if isinstance(sub, ReadNotificationsAction):
            return self._do_read_notifications()
        if isinstance(sub, CalendarViewAction):
            return self._do_calendar_view(sub.date)
        if isinstance(sub, CalendarAddAction):
            return self._do_calendar_add(sub)
        if isinstance(sub, ZomatoSearchAction):
            return self._do_zomato_search(sub)
        if isinstance(sub, ZomatoOpenAction):
            return self._do_zomato_open(sub.restaurant_id)
        if isinstance(sub, ZomatoOrderAction):
            return self._do_zomato_order(sub)
        if isinstance(sub, MapsSearchAction):
            return f"places={maps_search(query=sub.query)}", 1
        if isinstance(sub, MapsTravelTimeAction):
            return (
                f"travel={maps_travel_time(origin=sub.origin, destination=sub.destination)}",
                1,
            )
        if isinstance(sub, WebSearchAction):
            return f"results={web_search(query=sub.query)}", 1
        if isinstance(sub, WaitAction):
            return f"waited {sub.minutes} min", sub.minutes
        if isinstance(sub, EndTaskAction):
            return f"ended task (claim={sub.success_claim})", 1
        if isinstance(sub, ThinkAction):
            # No env effect, no time cost — just echo a compact form.
            truncated = sub.reasoning if len(sub.reasoning) <= 140 else sub.reasoning[:137] + "..."
            return f"thought: {truncated}", 0
        raise ValueError(f"Unsupported sub-action: {type(sub).__name__}")

    # ----------------------------------------------------------- tool handlers

    def _require_contact(self, name: str) -> str:
        if name not in self._state.contacts:
            raise ValueError(
                f"Unknown contact {name!r}. Known: {sorted(self._state.contacts.keys())}"
            )
        return name

    def _do_call(self, contact: str, *, via: str) -> tuple[str, int]:
        self._require_contact(contact)
        profile = self._state.contacts[contact]
        is_work = self._state.is_work_hours()
        picked_up = roll_pickup(profile, is_work, self._rng)
        self._state.active_call = {
            "contact": contact,
            "channel": via,
            "connected": picked_up,
            "since_min": self._state.current_time_min,
        }
        if picked_up:
            # Pickup resets annoyance counter; any active "call pickup counts as acknowledgement".
            profile.unanswered_agent_messages = 0
            # For tasks that check "Jay joined", a connected call is acceptance — record an
            # incoming MessageEvent so graders see it.
            self._state.messages.append(
                MessageEvent(
                    sender=contact,
                    recipient="user",
                    channel="call",
                    text="(picked up the call, I'm here)",
                    sent_at_min=self._state.current_time_min,
                )
            )
            return f"{contact} picked up the call", 2
        return f"{contact} did not pick up", 1

    def _do_hang_up(self) -> tuple[str, int]:
        if not self._state.active_call:
            raise ValueError("No active call to hang up")
        contact = self._state.active_call.get("contact", "?")
        self._state.active_call = None
        return f"hung up with {contact}", 1

    def _do_send(self, contact: str, text: str, *, channel: str) -> tuple[str, int]:
        self._require_contact(contact)
        now = self._state.current_time_min
        ev = MessageEvent(
            sender="user",
            recipient=contact,
            channel=channel,  # type: ignore[arg-type]
            text=text,
            sent_at_min=now,
        )
        self._state.messages.append(ev)
        # Schedule (or skip) a reply from the contact.
        schedule_reply(self._state, self._state.contacts[contact], channel, text, self._rng)
        return f"sent {channel} to {contact}: {text[:80]}", 1

    def _do_read_messages(self, contact: str | None, channel: str | None) -> tuple[str, int]:
        filtered = self._state.messages
        if contact:
            filtered = [m for m in filtered if contact in (m.sender, m.recipient)]
        if channel:
            filtered = [m for m in filtered if m.channel == channel]
        # Return compact dumps of the most recent 20.
        tail = filtered[-20:]
        summary_lines = [f"[{m.channel}] {m.sender}->{m.recipient}: {m.text[:120]}" for m in tail]
        return "\n".join(summary_lines) or "(no messages)", 1

    def _do_read_notifications(self) -> tuple[str, int]:
        # Notifications are surfaced via observation.notifications; this tool lets the agent
        # explicitly poll. We just report the message count from the watermark.
        unseen = [
            m
            for m in self._state.messages
            if m.sent_at_min > self._state.delivered_notifications_after_min
            and m.sender != "user"
        ]
        if not unseen:
            return "(no new notifications)", 1
        lines = [f"[{m.channel}] from {m.sender}: {m.text[:120]}" for m in unseen]
        return "\n".join(lines), 1

    def _do_calendar_view(self, date_str: str) -> tuple[str, int]:
        view = calendar_view(self._state, date=date_str)
        events = view.get("events", [])
        if not events:
            return "(no events)", 1
        lines = [
            f"{e['start']}  {e['title']}  ({e['duration_min']}m, with {', '.join(e['invitees']) or '-'})"
            for e in events
        ]
        return "\n".join(lines), 1

    def _do_calendar_add(self, a: CalendarAddAction) -> tuple[str, int]:
        result = calendar_add(
            self._state,
            title=a.title,
            start_time=a.start_time,
            duration_min=a.duration_min,
            invitees=list(a.invitees),
        )
        if result.get("view") == "error":
            raise ValueError(result.get("error", "calendar_add failed"))
        return f"added event {result['event_id']!r}: {a.title} @ {result['start']}", 1

    def _do_zomato_search(self, a: ZomatoSearchAction) -> tuple[str, int]:
        view = zomato_search(
            query=a.query,
            cuisine=a.cuisine,
            veg_only=a.veg_only,
            max_price_per_person=a.max_price_per_person,
        )
        hits = view.get("results", [])
        if not hits:
            return "(no restaurants matched)", 1
        compact = [(r["restaurant_id"], r["name"], r["location"], r["price_per_person"]) for r in hits]
        return f"results={compact}", 1

    def _do_zomato_open(self, restaurant_id: str) -> tuple[str, int]:
        detail = zomato_open(restaurant_id=restaurant_id)
        if detail.get("view") == "error":
            raise ValueError(detail.get("error", f"Unknown restaurant_id {restaurant_id!r}"))
        return f"opened {detail['name']}; menu={detail['menu']}", 1

    def _do_zomato_order(self, a: ZomatoOrderAction) -> tuple[str, int]:
        result = zomato_order(
            self._state,
            restaurant_id=a.restaurant_id,
            items=list(a.items),
            delivery_time=a.delivery_time,
        )
        if result.get("view") == "error":
            raise ValueError(result.get("error", f"Unknown restaurant_id {a.restaurant_id!r}"))
        return (
            f"order {result['order_id']!r} placed at {a.restaurant_id} ({a.delivery_time})",
            2,
        )

    # ------------------------------------------------------------ error paths

    def _handle_format_error(self, msg: str) -> PhonePilotObservation:
        self._state.step_count += 1
        self._state.format_error_streak += 1
        breakdown = apply_per_step(
            self._state, self._task, last_action=None, was_format_error=True
        )
        if self._state.format_error_streak >= MAX_FORMAT_ERROR_STREAK:
            self._state.terminated = True
        return self._build_observation(
            newly_delivered=[],
            last_outcome=None,
            format_error=msg,
            breakdown=breakdown,
        )

    def _surface_error(self, msg: str) -> PhonePilotObservation:
        """Semantic handler errors (e.g. unknown contact). Don't apply format penalty — the
        action was well-formed; the env just couldn't fulfill it. We still charge efficiency."""
        self._state.step_count += 1
        # Record a lightweight failed action for trajectory faithfulness.
        self._state.action_history.append(
            ActionRecord(tool="<error>", args={"message": msg}, outcome=msg, at_min=self._state.current_time_min)
        )
        rb = RewardBreakdown(efficiency=-0.02)
        self._state.reward_components["efficiency"] += rb.efficiency
        self._state.total_reward += rb.total
        return self._build_observation(
            newly_delivered=[], last_outcome=None, format_error=None, breakdown=rb, extra_error=msg
        )

    # ------------------------------------------------------ observation build

    def _build_observation(
        self,
        *,
        newly_delivered: list[MessageEvent],
        last_outcome: ActionRecord | None,
        format_error: str | None,
        breakdown: RewardBreakdown | None = None,
        extra_error: str | None = None,
    ) -> PhonePilotObservation:
        # recent_actions — last 5 records
        recent = [
            ActionOutcome(
                tool=a.tool,
                arg_summary=self._arg_summary(a.args),
                outcome=a.outcome,
                at_time=self._min_to_hhmm(a.at_min),
            )
            for a in self._state.action_history[-5:]
        ]

        # notifications — only messages the agent hasn't "seen" yet
        notifs = [
            Notification(
                kind="message",
                channel=m.channel,  # type: ignore[arg-type]
                contact=m.sender,
                preview=m.text[:120],
                timestamp=self._min_to_hhmm(m.sent_at_min),
            )
            for m in newly_delivered
            if m.sender != "user"
        ]
        # Update watermark.
        if newly_delivered:
            self._state.delivered_notifications_after_min = max(
                self._state.delivered_notifications_after_min,
                max(m.sent_at_min for m in newly_delivered),
            )

        # conversation summaries — last message per contact (any channel)
        summaries: dict[str, str] = {}
        for m in self._state.messages:
            if m.sender != "user":
                summaries[m.sender] = f"[{m.channel}] {m.text[:100]}"
            else:
                # Agent's own last line to each contact — also useful context.
                summaries.setdefault(m.recipient, f"(you) [{m.channel}] {m.text[:100]}")
                summaries[m.recipient + ":you"] = f"[{m.channel}] you: {m.text[:100]}"

        # active call display
        ac = self._state.active_call
        active_call_str: str | None = None
        if ac:
            active_call_str = (
                f"{'connected to' if ac.get('connected') else 'ringing'} "
                f"{ac.get('contact')} (via {ac.get('channel')})"
            )

        # open_app_view — last Zomato / Maps style result for convenience
        open_app_view: dict[str, Any] | None = None
        if last_outcome and last_outcome.tool == "zomato_open":
            detail = zomato_open(restaurant_id=last_outcome.args.get("restaurant_id", ""))
            if detail.get("view") != "error":
                open_app_view = detail

        reward_scalar = breakdown.total if breakdown is not None else None

        return PhonePilotObservation(
            done=self._state.terminated,
            reward=reward_scalar,
            user_goal=self._task.prompt,
            current_time=self._state.clock_hhmm(),
            time_budget_remaining_min=max(
                0, self._state.time_budget_min - self._state.current_time_min
            ),
            recent_actions=recent,
            active_call=active_call_str,
            open_app_view=open_app_view,
            notifications=notifs,
            conversation_summaries=summaries,
            error=format_error or extra_error,
            info={
                "task_id": self._task.id,
                "difficulty": self._task.difficulty,
                "sub_goals_fired": sorted(self._state.sub_goals_fired),
                "reward_components": dict(self._state.reward_components),
                "format_error_streak": self._state.format_error_streak,
            },
        )

    @staticmethod
    def _arg_summary(args: dict[str, Any]) -> str:
        parts: list[str] = []
        for k, v in args.items():
            s = str(v)
            if len(s) > 60:
                s = s[:57] + "..."
            parts.append(f"{k}={s}")
        return ", ".join(parts)

    def _min_to_hhmm(self, abs_min: int) -> str:
        total = (self._state.start_clock_min + abs_min) % (24 * 60)
        return f"{total // 60:02d}:{total % 60:02d}"


def build_env() -> PhonePilotEnvironment:
    """Factory used by :func:`openenv.core.create_app`."""
    return PhonePilotEnvironment()


__all__ = ["PhonePilotEnvironment", "build_env", "TASK_REGISTRY"]
