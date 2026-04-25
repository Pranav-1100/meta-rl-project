"""PhonePilot — a simulated smartphone-OS OpenEnv environment for personal-assistant RL."""

from .actions import PhonePilotAction
from .env import PhonePilotEnvironment, build_env
from .observations import PhonePilotObservation
from .state import PhonePilotState
from .tasks import TASK_REGISTRY, get_task

__all__ = [
    "PhonePilotAction",
    "PhonePilotObservation",
    "PhonePilotState",
    "PhonePilotEnvironment",
    "build_env",
    "TASK_REGISTRY",
    "get_task",
]

__version__ = "0.1.0"
