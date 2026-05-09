"""NavigateAgent — type a URL into the browser address bar.

The reliable path for "go to URL X": focus the URL bar via Ctrl+L,
select-all, type, Enter. No vision required, no homer overhead.
Cross-platform (Cmd+L on macOS, Ctrl+L on Linux/Windows).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.type_text import TypeAgent

logger = logging.getLogger(__name__)


@dataclass
class NavigateOutcome(Outcome):
    pass


class NavigateAgent(Agent):
    """Drive the browser address bar via keyboard shortcuts."""

    name = "navigate"

    async def run(
        self,
        *,
        url: str,
        platform: str = "linux",
        select_all_first: bool = True,
        post_settle: float = 1.5,
    ) -> NavigateOutcome:
        if self.ctx.keyboard is None:
            return NavigateOutcome(
                success=False, reason="no keyboard in context",
            )
        if not url:
            return NavigateOutcome(
                success=False, reason="empty url",
            )

        focus_mods = ["cmd"] if platform == "macos" else ["ctrl"]

        try:
            # 1. Focus URL bar.
            await self.ctx.keyboard.send_key_combo(focus_mods, "l")
            await asyncio.sleep(0.5)
            # 2. Select all (overwrite any pre-existing URL).
            if select_all_first:
                await self.ctx.keyboard.send_key_combo(focus_mods, "a")
                await asyncio.sleep(0.25)
            # 3. Type URL.
            await TypeAgent(self.ctx).run(
                text=url, secret=False, submit=False,
            )
            await asyncio.sleep(0.4)
            # 4. Enter.
            await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:
            logger.warning("NavigateAgent failed: %s", e)
            return NavigateOutcome(
                success=False, reason=f"send failed: {e}",
            )

        await asyncio.sleep(post_settle)
        return NavigateOutcome(
            success=True,
            reason=f"navigated to {url!r}",
            data={"url": url},
        )
