"""ClickAgent — find a target by description and click it.

The user-facing tier-3 click engine. Composes :class:`TargetAgent`
(target localisation), :class:`CursorAgent` (cursor finding via
HSV/oscillation/diff), and the existing visual servo loop in
:mod:`commander.visual_servo_homer`. The homer's per-step internals
(servo, ratio learning, click retry, post-click oracle) are unchanged
— this agent gives the rest of the layer a clean entrypoint.

Renamed from ``SearchAgent``. ``SearchAgent`` is kept as an alias for
any callers using the old name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class ClickOutcome(Outcome):
    pass


class ClickAgent(Agent):
    """Find a target by description and click it via the visual servo."""

    name = "click"

    async def run(
        self,
        *,
        target: str,
        button: str = "left",
    ) -> ClickOutcome:
        if self.ctx.capture is None:
            return ClickOutcome(
                success=False, reason="no capture in context",
            )
        if self.ctx.mouse is None:
            return ClickOutcome(
                success=False, reason="no mouse in context",
            )
        from terminaleyes.commander.visual_servo_homer import (
            VisualServoHomer,
        )
        from terminaleyes.agents.login import _SessionAdapter

        adapter = _SessionAdapter(self.ctx)
        homer = VisualServoHomer(session=adapter)
        outcome = await homer.run(target, button=button)
        return ClickOutcome(
            success=outcome.clicked,
            reason=outcome.reason,
            data={
                "steps": outcome.steps,
                "proof_path": outcome.proof_path,
            },
        )


# Back-compat alias — keep both names callable until external code
# migrates. Both classes share the same registry slot in the
# controller.
SearchAgent = ClickAgent
