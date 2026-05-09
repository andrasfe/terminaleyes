"""ScrollAgent — vertical scroll via the BT/USB mouse wheel HID path.

Atomic primitive. Higher-level agents (e.g. a scroll-aware ClickAgent)
call this when a target isn't visible in the current frame.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class ScrollOutcome(Outcome):
    pass


class ScrollAgent(Agent):
    """Send mouse-wheel scroll events.

    ``direction`` ∈ {``"down"``, ``"up"``}. ``amount`` is the number
    of wheel ticks (positive ints; sign is set by ``direction``).
    Optionally ``hover_at=(x_pct, y_pct)`` first parks the cursor
    over a region (e.g. the page body) so the scroll lands on the
    right pane — useful when a sidebar and main panel scroll
    independently.
    """

    name = "scroll"

    async def run(
        self,
        *,
        direction: str = "down",
        amount: int = 4,
        hover_at: tuple[float, float] | None = None,
        between_ticks: float = 0.05,
        post_settle: float = 0.4,
    ) -> ScrollOutcome:
        if self.ctx.mouse is None:
            return ScrollOutcome(
                success=False, reason="no mouse in context",
            )
        if direction not in ("up", "down"):
            return ScrollOutcome(
                success=False,
                reason=f"unknown direction {direction!r}",
            )
        if amount <= 0:
            return ScrollOutcome(
                success=False, reason="amount must be > 0",
            )

        # Optional: move cursor over the requested region first.
        # Without a calibrated HID-to-screen ratio we can't aim
        # exactly, but a slam + biased move lands inside the right
        # pane on most layouts.
        if hover_at is not None:
            try:
                await self._approximate_hover(hover_at)
            except Exception as e:
                logger.debug("hover_at attempt failed: %s", e)

        # Wheel ticks. Pi BT mouse takes signed amount; use
        # negative for up, positive for down.
        signed = -amount if direction == "up" else amount
        try:
            for _ in range(amount):
                step = 1 if signed > 0 else -1
                await self.ctx.mouse.scroll(step)
                await asyncio.sleep(between_ticks)
        except Exception as e:
            logger.warning("ScrollAgent scroll failed: %s", e)
            return ScrollOutcome(
                success=False, reason=f"scroll failed: {e}",
            )
        await asyncio.sleep(post_settle)
        return ScrollOutcome(
            success=True,
            reason=f"scrolled {direction} by {amount}",
            data={"direction": direction, "amount": amount},
        )

    async def _approximate_hover(
        self, target: tuple[float, float],
    ) -> None:
        """Slam to corner then send roughly the right HID delta to land
        the cursor near ``(x_pct, y_pct)`` in image coordinates."""
        # Slam.
        for _ in range(120):
            try:
                await self.ctx.mouse.move(-20, -20)
            except Exception:
                pass
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.2)
        # Open-loop estimate: ~1.6 HID per image-percent on most
        # macOS / Ubuntu defaults. Good enough for scroll-region
        # targeting.
        scale_per_pct = 1.6 / 0.01
        dx = int(target[0] * 100 * scale_per_pct)
        dy = int(target[1] * 100 * scale_per_pct)
        # Send in chunks of ±20 to respect the HID step size.
        rem_x, rem_y = dx, dy
        while rem_x != 0 or rem_y != 0:
            sx = max(-20, min(20, rem_x))
            sy = max(-20, min(20, rem_y))
            if sx != 0 or sy != 0:
                await self.ctx.mouse.move(sx, sy)
            rem_x -= sx
            rem_y -= sy
            await asyncio.sleep(0.003)
