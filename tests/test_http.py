"""OpenEnv HTTP contract tests — what judges' tools will actually hit."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phonepilot_env.server import app


@pytest.fixture
def client() -> TestClient:
    # Fresh singleton per test would be nicer, but the server intentionally uses a
    # process-level singleton. Each test resets before stepping, which is sufficient.
    return TestClient(app)


def test_health_endpoint_reports_healthy(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_metadata_endpoint_has_name(client: TestClient):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body.get("name")  # non-empty string


def test_schema_endpoint_returns_all_three_schemas(client: TestClient):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    for key in ("action", "observation", "state"):
        assert key in body


def test_reset_returns_initial_observation(client: TestClient):
    r = client.post(
        "/reset",
        json={"seed": 42, "episode_id": "http_t1", "task_id": "easy_ria_late"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is False
    assert "observation" in body
    assert body["observation"]["user_goal"].startswith("Let Ria know")


def test_full_episode_over_http(client: TestClient):
    client.post(
        "/reset",
        json={"seed": 1, "episode_id": "http_t2", "task_id": "easy_ria_late"},
    )
    r1 = client.post(
        "/step",
        json={
            "action": {
                "body": {
                    "tool": "send_whatsapp",
                    "contact": "Ria",
                    "text": "I'll be 10 min late to the 4pm meeting",
                }
            }
        },
    )
    assert r1.status_code == 200
    assert r1.json()["reward"] is not None

    client.post("/step", json={"action": {"body": {"tool": "wait", "minutes": 15}}})

    r_end = client.post(
        "/step",
        json={
            "action": {
                "body": {
                    "tool": "end_task",
                    "success_claim": True,
                    "summary": "WhatsApped Ria to tell her I'd be 10 min late to our 4pm meeting",
                }
            }
        },
    )
    assert r_end.status_code == 200
    assert r_end.json()["done"] is True


def test_malformed_action_returns_422(client: TestClient):
    client.post(
        "/reset",
        json={"seed": 0, "episode_id": "http_err", "task_id": "easy_ria_late"},
    )
    r = client.post("/step", json={"action": {"body": {"tool": "summon_uber"}}})
    assert r.status_code == 422  # Pydantic validation error → FastAPI 422
