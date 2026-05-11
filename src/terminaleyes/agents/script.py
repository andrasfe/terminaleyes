"""ScriptAgent — type and execute a verbatim shell script line by line.

Tier-3 workflow primitive. Takes a multi-line ``script`` string and
types each non-empty, non-comment line followed by Enter. Designed
to drop into a plan after :class:`LaunchAgent` (terminal) so the
controller can run a script verbatim against the target machine.

Each line goes through :class:`TypeAgent` so it picks up the BT-HID
first-character warmup and the post-send Enter follows the same
keystroke pre-flight as any other key. Comments (``#``) and blank
lines are skipped on the wire to keep the prompt clean and avoid
the warmup's leading-character cost on no-op lines.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.type_text import TypeAgent

logger = logging.getLogger(__name__)


@dataclass
class ScriptOutcome(Outcome):
    pass


class ScriptAgent(Agent):
    """Type a multi-line shell script into the focused terminal."""

    name = "script"

    async def run(
        self,
        *,
        script: str,
        settle_per_line: float = 0.6,
        record_label: str = "script",
    ) -> ScriptOutcome:
        if self.ctx.keyboard is None:
            return ScriptOutcome(
                success=False, reason="no keyboard in context",
                data={"executed": 0, "total": 0},
            )
        body = (script or "").strip()
        if not body:
            return ScriptOutcome(
                success=False, reason="empty script",
                data={"executed": 0, "total": 0},
            )

        lines = body.splitlines()
        executed: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            res = await TypeAgent(self.ctx).run(
                text=line, submit=True, post_settle=0.3,
            )
            if not res.success:
                return ScriptOutcome(
                    success=False,
                    reason=f"type failed on line {len(executed)+1}: {res.reason}",
                    data={"executed": len(executed), "total": len(lines),
                          "lines": executed},
                )
            executed.append(line)
            await asyncio.sleep(settle_per_line)

        # Visual proof for the UI replay log.
        try:
            if self.ctx.capture is not None:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label=record_label)
        except Exception as e:
            logger.debug("post-script capture failed: %s", e)

        print(f"   ScriptAgent: executed {len(executed)} line(s)")
        return ScriptOutcome(
            success=True,
            reason=f"executed {len(executed)} line(s)",
            data={"executed": len(executed), "total": len(lines),
                  "lines": executed},
        )
