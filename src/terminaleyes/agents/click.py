"""ClickAgent — find a target by description and click it.

The user-facing tier-3 click engine. Composes :class:`TargetAgent`
(target localisation), :class:`CursorAgent` (cursor finding), and the
existing visual servo loop in :mod:`commander.visual_servo_homer`.
The homer's per-step internals (servo, ratio learning, click retry,
post-click oracle) are unchanged — this agent gives the rest of the
layer a clean entrypoint.

Scroll-aware fallback: if the homer can't locate the target on the
first try, ClickAgent calls :class:`ScrollAgent` and retries. After
``scroll_attempts`` exhausted, it returns failure.

Renamed from ``SearchAgent``. ``SearchAgent`` is an alias.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class ClickOutcome(Outcome):
    pass


# Reasons the homer reports when the target wasn't located on screen.
# Used to decide whether scrolling-and-retrying could help.
_TARGET_NOT_FOUND_REASONS = (
    "target_lost",
)


class ClickAgent(Agent):
    """Find a target by description and click it via the visual servo.

    Scroll-aware: when the homer reports it couldn't locate the
    target, scroll the page (down by default) and try again, up to
    ``scroll_attempts`` times.
    """

    name = "click"

    async def run(
        self,
        *,
        target: str,
        button: str = "left",
        scroll_attempts: int = 3,
        scroll_direction: str = "down",
        scroll_amount: int = 4,
        scroll_hover_at: tuple[float, float] | None = (0.5, 0.5),
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
        from terminaleyes.agents.scroll import ScrollAgent

        adapter = _SessionAdapter(self.ctx)

        last_outcome = None
        for attempt in range(0, scroll_attempts + 1):
            if attempt > 0:
                # Scroll before retry. Hover the cursor over the
                # main pane so the wheel scrolls the right region.
                print(
                    f"ClickAgent: target not located; "
                    f"scroll {attempt}/{scroll_attempts} "
                    f"({scroll_direction} x{scroll_amount})"
                )
                await ScrollAgent(self.ctx).run(
                    direction=scroll_direction,
                    amount=scroll_amount,
                    hover_at=scroll_hover_at,
                )
                await asyncio.sleep(0.4)

            # Fresh homer per attempt — internal state is stale
            # after a scroll (target image position has changed).
            homer = VisualServoHomer(session=adapter)
            outcome = await homer.run(target, button=button)
            last_outcome = outcome
            if outcome.clicked:
                return ClickOutcome(
                    success=True,
                    reason=outcome.reason,
                    data={
                        "steps": outcome.steps,
                        "proof_path": outcome.proof_path,
                        "scroll_attempts_used": attempt,
                    },
                )
            # Only retry if the homer says the target wasn't located.
            # Other failures (validator held, etc.) shouldn't be
            # papered over with a scroll.
            reason_key = (outcome.reason or "").split(":", 1)[0]
            if reason_key not in _TARGET_NOT_FOUND_REASONS:
                break

        return ClickOutcome(
            success=False,
            reason=(last_outcome.reason
                    if last_outcome is not None
                    else "click failed"),
            data={
                "scroll_attempts_used": scroll_attempts,
                "proof_path": (last_outcome.proof_path
                               if last_outcome else None),
            },
        )


# Back-compat alias — keep both names callable until external code
# migrates. Both classes share the same registry slot in the
# controller.
SearchAgent = ClickAgent
