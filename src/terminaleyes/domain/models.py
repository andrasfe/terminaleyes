"""Core domain models for the terminaleyes system.

These models represent the fundamental data structures flowing through
the system: captured frames from the webcam, interpreted terminal state
from the MLLM, agent decisions, and keyboard actions to execute.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, enum.Enum):
    """Status of the agent's current task/goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminalReadiness(str, enum.Enum):
    """Whether the terminal appears ready to accept input."""

    READY = "ready"  # Shell prompt is visible, waiting for input
    BUSY = "busy"  # A command is still running
    ERROR = "error"  # An error state is detected
    UNKNOWN = "unknown"  # Cannot determine state


# ---------------------------------------------------------------------------
# Vision / Capture Models
# ---------------------------------------------------------------------------


class CropRegion(BaseModel):
    """Defines a rectangular crop region within a captured frame.

    Coordinates are in pixels, origin at top-left.
    """

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0, description="Left edge x-coordinate in pixels")
    y: int = Field(ge=0, description="Top edge y-coordinate in pixels")
    width: int = Field(gt=0, description="Width of the crop region in pixels")
    height: int = Field(gt=0, description="Height of the crop region in pixels")


class CapturedFrame(BaseModel):
    """A single frame captured from the webcam.

    Contains the raw image data as a numpy array along with metadata
    about when and how it was captured.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: np.ndarray = Field(description="Raw image data as BGR numpy array (OpenCV format)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the frame was captured")
    frame_number: int = Field(ge=0, description="Sequential frame counter")
    source_device: str = Field(default="webcam", description="Identifier for the capture device")
    crop_applied: CropRegion | None = Field(
        default=None, description="Crop region applied to this frame, if any"
    )


# ---------------------------------------------------------------------------
# MLLM Interpretation Models
# ---------------------------------------------------------------------------


class TerminalContent(BaseModel):
    """Structured representation of visible terminal content.

    Extracted by the MLLM from a terminal screenshot.
    """

    model_config = ConfigDict(frozen=True)

    visible_text: str = Field(
        description="Full text visible on the terminal screen, as read by the MLLM"
    )
    last_command: str | None = Field(
        default=None, description="The most recent command visible on screen"
    )
    last_output: str | None = Field(
        default=None, description="Output of the most recent command"
    )
    prompt_text: str | None = Field(
        default=None, description="The current shell prompt text, if visible"
    )
    error_messages: list[str] = Field(
        default_factory=list, description="Any error messages visible on screen"
    )
    working_directory: str | None = Field(
        default=None, description="Current working directory if discernible from prompt"
    )


class TerminalState(BaseModel):
    """Complete interpreted state of the terminal at a point in time.

    Produced by the MLLM Interpreter module after analyzing a captured frame.
    """

    model_config = ConfigDict(frozen=True)

    content: TerminalContent = Field(description="Structured content from the terminal")
    readiness: TerminalReadiness = Field(
        description="Whether the terminal appears ready for input"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="MLLM's confidence in its interpretation (0-1)"
    )
    raw_interpretation: str = Field(
        description="The raw text response from the MLLM before structuring"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this interpretation was produced"
    )
    frame_number: int = Field(ge=0, description="Which captured frame this interprets")


# ---------------------------------------------------------------------------
# Keyboard Action Models (discriminated union)
# ---------------------------------------------------------------------------


class Keystroke(BaseModel):
    """A single key press (e.g., Enter, Tab, Escape, 'a')."""

    model_config = ConfigDict(frozen=True)

    action_type: Literal["keystroke"] = "keystroke"
    key: str = Field(description="The key to press (e.g., 'Enter', 'Tab', 'a', 'F1')")
    description: str = Field(default="", description="Human-readable description of why this key is pressed")


class KeyCombo(BaseModel):
    """A key combination (e.g., Ctrl+C, Alt+F4)."""

    model_config = ConfigDict(frozen=True)

    action_type: Literal["key_combo"] = "key_combo"
    modifiers: list[str] = Field(description="Modifier keys (e.g., ['ctrl'], ['ctrl', 'shift'])")
    key: str = Field(description="The main key in the combination")
    description: str = Field(default="", description="Human-readable description")


class TextInput(BaseModel):
    """A string of text to type character by character."""

    model_config = ConfigDict(frozen=True)

    action_type: Literal["text_input"] = "text_input"
    text: str = Field(description="The text string to type")
    description: str = Field(default="", description="Human-readable description")


# Discriminated union for keyboard actions
KeyboardAction = Annotated[
    Union[Keystroke, KeyCombo, TextInput],
    Field(discriminator="action_type"),
]


# ---------------------------------------------------------------------------
# Agent Models
# ---------------------------------------------------------------------------


class AgentGoal(BaseModel):
    """A goal or task the agent is working toward.

    Goals are high-level objectives that the agent breaks down into
    sequences of keyboard actions based on terminal observations.
    """

    goal_id: str = Field(description="Unique identifier for this goal")
    description: str = Field(description="Human-readable description of what to achieve")
    success_criteria: str = Field(
        description="How to determine the goal has been achieved"
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=0, ge=0, description="Priority level (0 = highest)")
    max_steps: int = Field(
        default=100, gt=0, description="Maximum number of actions before giving up"
    )


class AgentAction(BaseModel):
    """A record of an action taken by the agent.

    Combines the keyboard action with the context in which it was decided.
    """

    model_config = ConfigDict(frozen=True)

    step_number: int = Field(ge=0, description="Sequential step number within the current goal")
    action: Keystroke | KeyCombo | TextInput = Field(description="The keyboard action taken")
    reasoning: str = Field(description="The agent's reasoning for choosing this action")
    timestamp: datetime = Field(default_factory=datetime.now)
    terminal_state_before: TerminalState | None = Field(
        default=None, description="Terminal state that prompted this action"
    )


class AgentContext(BaseModel):
    """The accumulated context the agent maintains across steps.

    This is the agent's 'working memory' containing the history of
    observations and actions, plus the current goal state.
    """

    current_goal: AgentGoal = Field(description="The goal currently being pursued")
    action_history: list[AgentAction] = Field(
        default_factory=list, description="Ordered list of actions taken so far"
    )
    observation_history: list[TerminalState] = Field(
        default_factory=list, description="Ordered list of terminal observations"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Arbitrary key-value metadata for strategies"
    )

    @property
    def step_count(self) -> int:
        """Number of actions taken so far."""
        return len(self.action_history)

    @property
    def is_over_limit(self) -> bool:
        """Whether the agent has exceeded the maximum allowed steps."""
        return self.step_count >= self.current_goal.max_steps

    @property
    def last_observation(self) -> TerminalState | None:
        """The most recent terminal observation, if any."""
        return self.observation_history[-1] if self.observation_history else None
