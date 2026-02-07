"""Tests for the AgentLoop orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from terminaleyes.agent.loop import AgentLoop
from terminaleyes.domain.models import AgentGoal, TaskStatus


class TestAgentLoop:
    """Test the agent loop orchestration.

    TODO: Add tests for:
        - Loop initializes with all required components
        - run() captures frames and sends to interpreter
        - run() passes observations to strategy.decide_action()
        - run() executes decided actions via keyboard output
        - run() calls strategy.evaluate_completion() after each step
        - run() stops when goal is completed
        - run() stops when goal fails
        - run() stops when max_steps is reached
        - run() handles consecutive errors up to the limit
        - stop() sets running flag to False
        - Action history is correctly accumulated in context
    """

    def test_agent_loop_init(
        self,
        mock_capture_source: AsyncMock,
        mock_mllm_provider: AsyncMock,
        mock_keyboard_output: AsyncMock,
    ) -> None:
        """AgentLoop should initialize with all components."""
        strategy = AsyncMock()
        loop = AgentLoop(
            capture=mock_capture_source,
            interpreter=mock_mllm_provider,
            keyboard=mock_keyboard_output,
            strategy=strategy,
        )
        assert loop.is_running is False
