"""Agent / Decision Engine module for terminaleyes.

Contains the central agentic loop that orchestrates vision capture,
MLLM interpretation, decision-making, and keyboard action output.
Supports pluggable strategies for different behaviors.

Public API:
    AgentStrategy -- Abstract strategy base class
    AgentLoop -- Central orchestrator
"""

from terminaleyes.agent.base import AgentStrategy
from terminaleyes.agent.loop import AgentLoop

__all__ = ["AgentLoop", "AgentStrategy"]
