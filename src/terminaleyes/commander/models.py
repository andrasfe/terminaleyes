"""Domain models for the visual command agent."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConditionSpec(BaseModel):
    """What to look for on screen."""

    model_config = ConfigDict(frozen=True)

    description: str
    element_type: str | None = None
    element_text: str | None = None
    visual_cues: list[str] = Field(default_factory=list)
    spatial_context: str | None = None


class ActionSpec(BaseModel):
    """What to do when condition is met."""

    model_config = ConfigDict(frozen=True)

    action_type: str  # mouse_click, keystroke, key_combo, text_input
    button: str | None = None
    key: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    text: str | None = None
    target: str = "element"  # "element" = click detected element, "current" = click in place


class CommandSpec(BaseModel):
    """Complete parsed command: what to watch for, what to do, how often."""

    model_config = ConfigDict(frozen=True)

    raw_instruction: str
    condition: ConditionSpec
    action: ActionSpec
    interval_seconds: float = 180.0
    max_attempts: int = 0  # 0 = unlimited
    one_shot: bool = True  # stop after first trigger


class ScreenLocation(BaseModel):
    """Estimated location of a detected element on screen."""

    model_config = ConfigDict(frozen=True)

    x_pct: float = Field(ge=0.0, le=1.0)
    y_pct: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ScreenCheckResult(BaseModel):
    """Result of checking whether the full monitor is visible in the webcam frame."""

    model_config = ConfigDict(frozen=True)

    full_screen_visible: bool = False
    edges_cut_off: list[str] = Field(default_factory=list)  # "top", "bottom", "left", "right"
    suggestion: str = ""
    reasoning: str = ""
    raw_response: str = ""


class CursorLocateResult(BaseModel):
    """Result of locating the cursor and target element on screen."""

    model_config = ConfigDict(frozen=True)

    cursor_found: bool = False
    cursor_location: ScreenLocation | None = None
    target_found: bool = False
    target_location: ScreenLocation | None = None
    cursor_on_target: bool = False
    reasoning: str = ""
    raw_response: str = ""


class ConditionResult(BaseModel):
    """Result of evaluating a condition against a frame."""

    model_config = ConfigDict(frozen=True)

    condition_met: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    location: ScreenLocation | None = None
    reasoning: str = ""
    raw_response: str = ""


class CommandSession(BaseModel):
    """Runtime state for a command execution session."""

    session_id: str
    command: CommandSpec
    started_at: datetime
    frames_checked: int = 0
    condition_met_count: int = 0
    actions_executed: int = 0
    last_check_at: datetime | None = None
    status: str = "running"  # running, triggered, completed, failed, cancelled
