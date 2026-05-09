"""Backwards-compat shim. SearchAgent is now an alias for ClickAgent."""

from terminaleyes.agents.click import ClickAgent, SearchAgent

__all__ = ["ClickAgent", "SearchAgent"]
