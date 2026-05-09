"""SearchAgent — find a target by description and click it.

Wraps the existing :class:`VisualServoHomer` for now; the homer's
internals (cursor finder + target finder + servo loop + click + post-
click oracle) will be unbundled into proper tier-1/2 agents in a
later refactor. SearchAgent gives the rest of the agent layer a clean
interface to "find and click X" without needing to know about the
homer's session-based API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class SearchOutcome(Outcome):
    pass


class SearchAgent(Agent):
    """Find-and-click a target by natural-language description.

    Args to :meth:`run`:
      - ``target``: free-form description (``"the 'Run' button"``,
        ``"the search bar"``, etc.)
      - ``button``: ``"left"`` (default) or ``"right"``
    """

    name = "search"

    async def run(
        self,
        *,
        target: str,
        button: str = "left",
    ) -> SearchOutcome:
        if self.ctx.capture is None:
            return SearchOutcome(
                success=False, reason="no capture in context",
            )
        if self.ctx.mouse is None:
            return SearchOutcome(
                success=False, reason="no mouse in context",
            )

        # Lazy import — the homer module is heavy.
        from terminaleyes.commander.visual_servo_homer import (
            VisualServoHomer,
        )
        # Same adapter shape used by LoginAgent for the visual click.
        from terminaleyes.agents.login import _SessionAdapter

        adapter = _SessionAdapter(self.ctx)
        homer = VisualServoHomer(session=adapter)
        outcome = await homer.run(target, button=button)
        return SearchOutcome(
            success=outcome.clicked,
            reason=outcome.reason,
            data={
                "steps": outcome.steps,
                "proof_path": outcome.proof_path,
            },
        )
