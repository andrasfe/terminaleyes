"""Shared test fixtures for the terminaleyes test suite.

Provides common fixtures used across unit and integration tests:
sample frames, mock MLLM responses, mock keyboard outputs, etc.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from terminaleyes.domain.models import (
    AgentContext,
    AgentGoal,
    CapturedFrame,
    CropRegion,
    Keystroke,
    TaskStatus,
    TerminalContent,
    TerminalReadiness,
    TerminalState,
    TextInput,
)


# ---------------------------------------------------------------------------
# Frame / Image Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_image() -> np.ndarray:
    """A minimal 100x100 black image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame(sample_image: np.ndarray) -> CapturedFrame:
    """A CapturedFrame with a sample image."""
    return CapturedFrame(
        image=sample_image,
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        frame_number=0,
        source_device="test",
    )


@pytest.fixture
def sample_crop_region() -> CropRegion:
    """A sample crop region for testing."""
    return CropRegion(x=10, y=10, width=80, height=80)


# ---------------------------------------------------------------------------
# Terminal State Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_terminal_content() -> TerminalContent:
    """A sample TerminalContent representing a ready shell."""
    return TerminalContent(
        visible_text="user@host:~$ ls\nfile1.txt  file2.txt\nuser@host:~$ ",
        last_command="ls",
        last_output="file1.txt  file2.txt",
        prompt_text="user@host:~$ ",
        error_messages=[],
        working_directory="~",
    )


@pytest.fixture
def sample_terminal_state(sample_terminal_content: TerminalContent) -> TerminalState:
    """A sample TerminalState with a ready terminal."""
    return TerminalState(
        content=sample_terminal_content,
        readiness=TerminalReadiness.READY,
        confidence=0.95,
        raw_interpretation='{"visible_text": "..."}',
        timestamp=datetime(2025, 1, 1, 12, 0, 1),
        frame_number=0,
    )


@pytest.fixture
def error_terminal_state() -> TerminalState:
    """A TerminalState representing an error condition."""
    return TerminalState(
        content=TerminalContent(
            visible_text="user@host:~$ cat nonexistent\ncat: nonexistent: No such file or directory\nuser@host:~$ ",
            last_command="cat nonexistent",
            last_output="cat: nonexistent: No such file or directory",
            prompt_text="user@host:~$ ",
            error_messages=["cat: nonexistent: No such file or directory"],
            working_directory="~",
        ),
        readiness=TerminalReadiness.READY,
        confidence=0.90,
        raw_interpretation='{"visible_text": "..."}',
        timestamp=datetime(2025, 1, 1, 12, 0, 2),
        frame_number=1,
    )


# ---------------------------------------------------------------------------
# Agent Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_goal() -> AgentGoal:
    """A sample agent goal."""
    return AgentGoal(
        goal_id="test-goal-1",
        description="List the files in the home directory",
        success_criteria="The ls command output is visible on screen",
        status=TaskStatus.PENDING,
        max_steps=10,
    )


@pytest.fixture
def sample_context(sample_goal: AgentGoal) -> AgentContext:
    """A sample agent context with no history."""
    return AgentContext(current_goal=sample_goal)


# ---------------------------------------------------------------------------
# Mock Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_capture_source() -> AsyncMock:
    """A mock CaptureSource for testing components in isolation.

    TODO: Configure with appropriate return values for:
        - open() / close()
        - capture_frame() -> sample CapturedFrame
    """
    mock = AsyncMock()
    mock.is_open = True
    return mock


@pytest.fixture
def mock_mllm_provider() -> AsyncMock:
    """A mock MLLMProvider for testing without real API calls.

    TODO: Configure interpret() to return a sample TerminalState.
    """
    mock = AsyncMock()
    mock.model = "mock-model"
    return mock


@pytest.fixture
def mock_keyboard_output() -> AsyncMock:
    """A mock KeyboardOutput for testing without a real target.

    TODO: Configure send_keystroke/send_key_combo/send_text as AsyncMock.
    """
    mock = AsyncMock()
    return mock
