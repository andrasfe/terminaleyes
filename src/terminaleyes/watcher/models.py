"""Domain models for screen watching sessions."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScreenObservation(BaseModel):
    """A single observation from the screen watcher."""

    timestamp: datetime
    frame_number: int
    content_type: str = Field(
        description="Type of content: web_browser, code_editor, terminal, etc."
    )
    application_context: str | None = Field(
        default=None,
        description="Identified application name, e.g. VS Code, Chrome",
    )
    visible_text: str = Field(description="All readable text from the screen")
    unreadable_notes: str = Field(
        default="",
        description="Description of what couldn't be read and why",
    )
    positioning_notes: str = Field(
        default="none",
        description="Screen edge clipping, glare, distance issues",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    raw_response: str = Field(default="")


class WatchSession(BaseModel):
    """Summary of a complete watch session."""

    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_minutes: float = 0.0
    capture_interval_minutes: float = 3.0
    total_captures: int = 0
    changes_detected: int = 0
    observations: list[ScreenObservation] = Field(default_factory=list)
    final_summary: str = ""
