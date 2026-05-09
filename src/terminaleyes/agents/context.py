"""Shared resources injected into every agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terminaleyes.agents.vault import Vault
    from terminaleyes.capture.webcam import WebcamCapture
    from terminaleyes.commander.evaluator import ConditionEvaluator
    from terminaleyes.keyboard.base import KeyboardOutput
    from terminaleyes.mouse.base import MouseOutput


@dataclass
class AgentContext:
    """Bag of shared infrastructure passed to every agent.

    Most fields are optional so agents that don't need them (e.g. the
    Vault used standalone in a CLI) can construct a minimal context.
    Construct with whatever you have; agents document which fields
    they require.
    """

    # I/O — Pi-side HID
    mouse: "MouseOutput | None" = None
    keyboard: "KeyboardOutput | None" = None

    # Vision input
    capture: "WebcamCapture | None" = None

    # LLM clients — OpenAI-compatible (LM Studio) for multimodal calls
    vision_client: Any = None        # an openai.AsyncClient or similar
    vision_model: str = ""

    # ShowUI grounding helper (callable: b64, prompt -> (x, y) | None)
    showui_query: Any = None

    # Misc helpers from the older commander stack — present so we
    # don't reimplement parsers/JSON extractors during the migration.
    evaluator: "ConditionEvaluator | None" = None

    # Storage
    vault: "Vault | None" = None

    # Free-form scratchpad for cross-agent state during a run.
    scratch: dict[str, Any] = field(default_factory=dict)
