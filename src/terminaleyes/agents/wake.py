"""WakeAgent — bring a sleeping screen / monitor to a usable state.

Sends a sequence of stimuli that wake monitors and dismiss screensaver
or clock overlays without triggering destructive shortcuts. Used by
LoginAgent and FocusAgent. Standalone-callable.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class WakeOutcome(Outcome):
    pass


class WakeAgent(Agent):
    """Mouse jiggle + arrow key + click. Idempotent; safe to retry."""

    name = "wake"

    async def run(
        self,
        *,
        jiggle_count: int = 4,
        send_arrow: bool = True,
        click: bool = True,
        settle_seconds: float = 0.6,
    ) -> WakeOutcome:
        if self.ctx.mouse is None and self.ctx.keyboard is None:
            return WakeOutcome(
                success=False, reason="no mouse or keyboard in context",
            )

        # 1. Mouse jiggle — wakes monitors and registers activity.
        if self.ctx.mouse is not None:
            for _ in range(jiggle_count):
                try:
                    await self.ctx.mouse.move(20, 0)
                    await asyncio.sleep(0.04)
                    await self.ctx.mouse.move(-20, 0)
                    await asyncio.sleep(0.04)
                except Exception as e:
                    logger.warning("Wake jiggle failed: %s", e)
                    break

        # 2. Down arrow — dismisses GDM clock overlay; safe key.
        if send_arrow and self.ctx.keyboard is not None:
            try:
                await self.ctx.keyboard.send_keystroke("Down")
            except Exception as e:
                logger.warning("Wake keystroke failed: %s", e)
        await asyncio.sleep(0.4)

        # 3. Left click — covers lock screens that need a click before
        # showing the password prompt.
        if click and self.ctx.mouse is not None:
            try:
                await self.ctx.mouse.click("left")
            except Exception as e:
                logger.warning("Wake click failed: %s", e)
        await asyncio.sleep(settle_seconds)

        return WakeOutcome(success=True, reason="wake sequence completed")
