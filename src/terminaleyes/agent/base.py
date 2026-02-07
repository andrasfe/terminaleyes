"""Abstract base class for agent strategies.

A strategy encapsulates the decision-making logic that determines
what keyboard action to take next based on the current terminal state
and action history. Different strategies can implement different
behaviors (e.g., command execution, file editing, exploration).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

from terminaleyes.domain.models import (
    AgentContext,
    KeyCombo,
    Keystroke,
    TaskStatus,
    TerminalState,
    TextInput,
)

logger = logging.getLogger(__name__)


class AgentStrategy(ABC):
    """Abstract strategy interface for agent decision-making.

    Each strategy defines how the agent should respond to terminal
    observations to achieve a particular kind of goal. The agent loop
    calls decide_action() after each observation.

    Example usage::

        class ShellCommandStrategy(AgentStrategy):
            async def decide_action(self, context, observation):
                if observation.readiness == TerminalReadiness.READY:
                    return TextInput(text="ls -la"), "Listing directory contents"
                return None, "Waiting for terminal"

    TODO: Implement concrete strategies for common use cases:
        - ShellCommandStrategy: Execute a sequence of shell commands
        - FileEditStrategy: Edit files using terminal editors
        - ExplorationStrategy: Explore system state and gather information
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy.

        TODO: Return a descriptive name string.
        """
        ...

    @abstractmethod
    async def decide_action(
        self,
        context: AgentContext,
        observation: TerminalState,
    ) -> tuple[Keystroke | KeyCombo | TextInput | None, str]:
        """Decide the next keyboard action based on current state.

        This is the core decision method. It receives the full agent
        context (goal, history) and the latest terminal observation,
        and returns an action to take.

        Args:
            context: The agent's accumulated context including goal,
                     action history, and observation history.
            observation: The latest terminal state observation.

        Returns:
            A tuple of (action, reasoning) where:
            - action is the KeyboardAction to execute, or None to wait/skip
            - reasoning is a human-readable explanation of the decision

        TODO: Implement decision logic in each concrete strategy.
              Consider:
              - Checking observation.readiness before acting
              - Examining observation.content for expected output
              - Checking context.is_over_limit for step limits
              - Using context.action_history to avoid loops
        """
        ...

    @abstractmethod
    async def evaluate_completion(
        self,
        context: AgentContext,
        observation: TerminalState,
    ) -> TaskStatus:
        """Evaluate whether the current goal has been achieved.

        Called after each action+observation cycle to determine if
        the goal is complete, still in progress, or has failed.

        Args:
            context: The agent's accumulated context.
            observation: The latest terminal state observation.

        Returns:
            TaskStatus indicating the goal's current status.

        TODO: Implement goal completion detection for each strategy.
              Consider:
              - Checking against context.current_goal.success_criteria
              - Detecting error states in the observation
              - Checking context.is_over_limit for timeout
        """
        ...
