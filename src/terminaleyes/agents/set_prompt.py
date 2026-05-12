"""SetPromptAgent — change the focused shell's PS1 to a fixed label.

Tier-3 workflow primitive. Takes a ``label`` and types

    PS1='<label>$ '

into the focused terminal, followed by Enter. The PS1 change is
runtime-only (affects the current shell; not persisted to .bashrc).

Why this exists as its own agent
--------------------------------
The natural intent "change the bash prompt to mini1" requires the
LLM planner to emit a step that types a string containing both
single and double quotes (``PS1='mini1$ '``). The small OCR-model
planner kept truncating its JSON output on the quote characters,
producing 1-step plans that only launched the terminal. Pushing
the quoting into a dedicated agent removes the LLM's exposure to
the special characters — the planner only has to emit
``{"name": "set_prompt", "kwargs": {"label": "mini1"}}``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.type_text import TypeAgent

logger = logging.getLogger(__name__)


@dataclass
class SetPromptOutcome(Outcome):
    pass


class SetPromptAgent(Agent):
    """Type ``PS1='<label><suffix>'`` + Enter into the focused shell."""

    name = "set_prompt"

    async def run(
        self,
        *,
        label: str,
        suffix: str = "$ ",
        record_label: str = "set_prompt",
    ) -> SetPromptOutcome:
        if self.ctx.keyboard is None:
            return SetPromptOutcome(
                success=False, reason="no keyboard in context",
                data={"label": label},
            )
        clean = (label or "").strip()
        if not clean:
            return SetPromptOutcome(
                success=False, reason="empty label",
                data={"label": ""},
            )
        if "'" in clean:
            # We single-quote the value, so a literal apostrophe in
            # the label would terminate the quote and break the
            # shell. Reject loudly rather than silently mangling.
            return SetPromptOutcome(
                success=False,
                reason="label must not contain a single quote",
                data={"label": clean},
            )
        cmd = f"PS1='{clean}{suffix}'"
        res = await TypeAgent(self.ctx).run(
            text=cmd, submit=True, post_settle=0.4,
        )
        if not res.success:
            return SetPromptOutcome(
                success=False,
                reason=f"type failed: {res.reason}",
                data={"label": clean, "cmd": cmd},
            )
        try:
            if self.ctx.capture is not None:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label=record_label)
        except Exception as e:
            logger.debug("post-set_prompt capture failed: %s", e)
        print(f"   SetPromptAgent: prompt set to {clean!r}")
        return SetPromptOutcome(
            success=True,
            reason=f"set PS1 to {clean!r}",
            data={"label": clean, "cmd": cmd},
        )
