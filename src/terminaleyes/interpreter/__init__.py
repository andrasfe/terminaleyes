"""MLLM Interpreter module for terminaleyes.

Provides a provider-agnostic interface for sending terminal screenshots
to multimodal LLMs and receiving structured interpretations of the
terminal state.

Public API:
    MLLMProvider -- Abstract base class
    AnthropicProvider -- Claude API implementation
    OpenAIProvider -- OpenAI / OpenRouter implementation
"""

from terminaleyes.interpreter.base import MLLMProvider, MLLMError

__all__ = ["MLLMProvider", "MLLMError", "AnthropicProvider", "OpenAIProvider"]


def __getattr__(name: str) -> type:
    """Lazy import for concrete implementations that require external deps."""
    if name == "AnthropicProvider":
        from terminaleyes.interpreter.anthropic import AnthropicProvider
        return AnthropicProvider
    if name == "OpenAIProvider":
        from terminaleyes.interpreter.openai import OpenAIProvider
        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
