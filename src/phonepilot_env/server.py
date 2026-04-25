"""FastAPI server exposing the PhonePilot environment over OpenEnv's HTTP/WS protocol.

Run locally:
    uv run uvicorn phonepilot_env.server:app --reload --port 8000

Hugging Face Space / Docker entrypoint: same, just ``--host 0.0.0.0``. The container's
Dockerfile is wired for this exact module path.
"""

from __future__ import annotations

from openenv.core import create_app

from .actions import PhonePilotAction
from .env import PhonePilotEnvironment
from .observations import PhonePilotObservation


# OpenEnv's HTTP /step handler creates a fresh env instance on every call by default.
# Our env is multi-turn — state must persist across /step calls — so we return a singleton.
# ``close()`` is a no-op, and /reset re-seeds state in place, so this is safe.
_singleton: PhonePilotEnvironment | None = None


def _env_factory() -> PhonePilotEnvironment:
    global _singleton
    if _singleton is None:
        _singleton = PhonePilotEnvironment()
    return _singleton


app = create_app(
    _env_factory,
    PhonePilotAction,
    PhonePilotObservation,
    env_name="phonepilot",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``python -m phonepilot_env.server`` and ``uv run server``."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
