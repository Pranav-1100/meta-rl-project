"""Stubbed app backends (Calendar / Zomato / Maps / WebSearch).

All functions mutate :class:`PhonePilotState` where appropriate (e.g. adding a calendar
event or an order), and return a dict payload suitable for the observation's
``open_app_view`` or ``recent_actions[-1].outcome``.

Keeping the data tables in one place makes the task graders' string-matching checks
predictable and cheap to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .state import CalendarEvent, Order

if TYPE_CHECKING:
    from .state import PhonePilotState


# ---------------------------------------------------------------------------
# Zomato — canned restaurant catalog
# ---------------------------------------------------------------------------

_ZOMATO_CATALOG: dict[str, dict[str, Any]] = {
    "z_sushi_haven": {
        "name": "Sushi Haven",
        "cuisine": "Japanese",
        "location": "Indiranagar",
        "price_per_person": 850,
        "veg_options": True,
        "rating": 4.5,
        "menu": {
            "Veg Maki Platter": 450,
            "California Roll": 380,
            "Salmon Nigiri (6pc)": 550,
            "Miso Soup": 120,
            "Edamame": 180,
        },
    },
    "z_sakura_sushi": {
        "name": "Sakura Sushi Bar",
        "cuisine": "Japanese",
        "location": "Koramangala",
        "price_per_person": 1100,
        "veg_options": True,
        "rating": 4.3,
        "menu": {
            "Veg Tempura Roll": 520,
            "Tuna Sashimi": 780,
            "Dragon Roll": 680,
        },
    },
    "z_pizza_place": {
        "name": "Slice of Napoli",
        "cuisine": "Italian",
        "location": "Koramangala",
        "price_per_person": 650,
        "veg_options": True,
        "rating": 4.2,
        "menu": {"Margherita": 450, "Pepperoni": 520, "Garlic Bread": 180},
    },
    "z_biryani_house": {
        "name": "Biryani House",
        "cuisine": "Indian",
        "location": "Jayanagar",
        "price_per_person": 320,
        "veg_options": True,
        "rating": 4.0,
        "menu": {"Veg Biryani": 260, "Chicken Biryani": 320, "Raita": 40},
    },
}


def zomato_search(
    *,
    query: str,
    cuisine: str | None,
    veg_only: bool,
    max_price_per_person: int | None,
) -> dict[str, Any]:
    q = query.lower()
    results = []
    for rid, r in _ZOMATO_CATALOG.items():
        if cuisine and r["cuisine"].lower() != cuisine.lower():
            continue
        if veg_only and not r["veg_options"]:
            continue
        if max_price_per_person is not None and r["price_per_person"] > max_price_per_person:
            continue
        # naive fuzzy match
        if q and not any(tok in r["name"].lower() or tok in r["cuisine"].lower() for tok in q.split()):
            continue
        results.append(
            {
                "restaurant_id": rid,
                "name": r["name"],
                "cuisine": r["cuisine"],
                "location": r["location"],
                "price_per_person": r["price_per_person"],
                "veg_options": r["veg_options"],
                "rating": r["rating"],
            }
        )
    return {"app": "zomato", "view": "search_results", "query": query, "results": results}


def zomato_open(*, restaurant_id: str) -> dict[str, Any]:
    r = _ZOMATO_CATALOG.get(restaurant_id)
    if not r:
        return {"app": "zomato", "view": "error", "error": f"unknown restaurant {restaurant_id!r}"}
    return {
        "app": "zomato",
        "view": "restaurant",
        "restaurant_id": restaurant_id,
        "name": r["name"],
        "cuisine": r["cuisine"],
        "location": r["location"],
        "price_per_person": r["price_per_person"],
        "veg_options": r["veg_options"],
        "rating": r["rating"],
        "menu": r["menu"],
    }


def zomato_order(
    state: "PhonePilotState",
    *,
    restaurant_id: str,
    items: list[str],
    delivery_time: str,
) -> dict[str, Any]:
    r = _ZOMATO_CATALOG.get(restaurant_id)
    if not r:
        return {"app": "zomato", "view": "error", "error": f"unknown restaurant {restaurant_id!r}"}
    # Round up unknown items to 0 rather than fail — mirrors real app flexibility.
    total = sum(r["menu"].get(item, 0) for item in items)
    order_id = f"ord_{len(state.orders) + 1:03d}"
    order = Order(
        order_id=order_id,
        restaurant_id=restaurant_id,
        items=items,
        delivery_time=delivery_time,
        placed_at_min=state.current_time_min,
        price_per_person=r["price_per_person"],
    )
    state.orders.append(order)
    return {
        "app": "zomato",
        "view": "order_confirmation",
        "order_id": order_id,
        "restaurant_id": restaurant_id,
        "items": items,
        "delivery_time": delivery_time,
        "estimated_total": total,
        "price_per_person": r["price_per_person"],
    }


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def calendar_view(state: "PhonePilotState", *, date: str) -> dict[str, Any]:
    # date arg is advisory — our one-day sim doesn't need full date indexing.
    events = [
        {
            "event_id": e.event_id,
            "title": e.title,
            "start": _min_to_hhmm(e.start_min),
            "duration_min": e.duration_min,
            "invitees": e.invitees,
        }
        for e in state.calendar
    ]
    return {"app": "calendar", "view": "day", "date": date, "events": events}


def calendar_add(
    state: "PhonePilotState",
    *,
    title: str,
    start_time: str,
    duration_min: int,
    invitees: list[str],
) -> dict[str, Any]:
    start_min = _parse_hhmm(start_time)
    if start_min is None:
        return {"app": "calendar", "view": "error", "error": f"bad start_time {start_time!r}"}
    event_id = f"evt_{len(state.calendar) + 1:03d}"
    ev = CalendarEvent(
        event_id=event_id,
        title=title,
        start_min=start_min,
        duration_min=duration_min,
        invitees=invitees,
    )
    state.calendar.append(ev)
    return {
        "app": "calendar",
        "view": "event_created",
        "event_id": event_id,
        "title": title,
        "start": _min_to_hhmm(start_min),
        "duration_min": duration_min,
        "invitees": invitees,
    }


def _parse_hhmm(s: str) -> int | None:
    """Accept 'HH:MM' or '7pm' / '7:30pm' variants. Returns minutes-of-day, or None."""
    s = s.strip().lower().replace(" ", "")
    # handle am/pm
    suffix = None
    if s.endswith("pm"):
        suffix, s = "pm", s[:-2]
    elif s.endswith("am"):
        suffix, s = "am", s[:-2]
    if ":" in s:
        try:
            h, m = [int(x) for x in s.split(":", 1)]
        except ValueError:
            return None
    else:
        try:
            h, m = int(s), 0
        except ValueError:
            return None
    if suffix == "pm" and h < 12:
        h += 12
    elif suffix == "am" and h == 12:
        h = 0
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def _min_to_hhmm(total: int) -> str:
    total %= 24 * 60
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

# Approximate pairwise distance-km table (Bangalore-ish geometry).
_MAPS_DISTANCE_KM: dict[tuple[str, str], float] = {}
_CITY_NODES = ["Koramangala", "Indiranagar", "Whitefield", "Jayanagar", "HSR Layout"]
_DIST_MATRIX = [
    # Kor, Ind, Whi, Jay, HSR
    [0, 5, 15, 7, 4],
    [5, 0, 12, 11, 9],
    [15, 12, 0, 22, 18],
    [7, 11, 22, 0, 10],
    [4, 9, 18, 10, 0],
]
for i, a in enumerate(_CITY_NODES):
    for j, b in enumerate(_CITY_NODES):
        _MAPS_DISTANCE_KM[(a.lower(), b.lower())] = float(_DIST_MATRIX[i][j])


def maps_search(*, query: str) -> dict[str, Any]:
    q = query.lower()
    hits = [node for node in _CITY_NODES if q in node.lower() or node.lower() in q]
    return {
        "app": "maps",
        "view": "search_results",
        "query": query,
        "results": hits[:5] or _CITY_NODES[:3],
    }


def maps_travel_time(*, origin: str, destination: str) -> dict[str, Any]:
    key = (origin.lower().strip(), destination.lower().strip())
    km = _MAPS_DISTANCE_KM.get(key)
    if km is None:
        # best-effort: any substring match
        for (a, b), d in _MAPS_DISTANCE_KM.items():
            if origin.lower() in a and destination.lower() in b:
                km = d
                break
    if km is None:
        return {
            "app": "maps",
            "view": "error",
            "error": f"can't route from {origin!r} to {destination!r}",
        }
    # Simple heuristic: ~2.5 min/km in traffic, floor 5 min.
    minutes = max(5, int(round(km * 2.5)))
    return {
        "app": "maps",
        "view": "travel_time",
        "origin": origin,
        "destination": destination,
        "distance_km": km,
        "travel_time_min": minutes,
    }


# ---------------------------------------------------------------------------
# Web search (canned lookup)
# ---------------------------------------------------------------------------

_WEB_SEARCH_ANSWERS: dict[str, str] = {
    "sushi": "Top sushi spots in Bangalore: Sushi Haven (Indiranagar, 4.5★), Sakura Sushi Bar (Koramangala, 4.3★).",
    "pizza": "Top pizza spots: Slice of Napoli (Koramangala, 4.2★).",
    "biryani": "Top biryani: Biryani House (Jayanagar, 4.0★).",
}


def web_search(*, query: str) -> dict[str, Any]:
    q = query.lower()
    for kw, ans in _WEB_SEARCH_ANSWERS.items():
        if kw in q:
            return {"app": "web", "view": "answer", "query": query, "answer": ans}
    return {
        "app": "web",
        "view": "answer",
        "query": query,
        "answer": "(no strong match — try a more specific query)",
    }
