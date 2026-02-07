"""Domain models for terminaleyes.

This package contains all core data structures, enumerations, and value
objects used throughout the system. All models use Pydantic v2 for
validation and serialization.
"""

from terminaleyes.domain.models import (
    AgentAction,
    AgentContext,
    AgentGoal,
    CapturedFrame,
    KeyboardAction,
    KeyCombo,
    Keystroke,
    TaskStatus,
    TerminalState,
    TextInput,
)

__all__ = [
    "AgentAction",
    "AgentContext",
    "AgentGoal",
    "CapturedFrame",
    "KeyboardAction",
    "KeyCombo",
    "Keystroke",
    "TaskStatus",
    "TerminalState",
    "TextInput",
]
