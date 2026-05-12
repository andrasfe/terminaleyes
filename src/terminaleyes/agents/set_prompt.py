"""SetPromptAgent — change the focused shell's PS1 to a fixed label.

Tier-3 workflow primitive. Takes a ``label`` and types

    PS1='<label>$ '

into the focused terminal, followed by Enter. By default the
change is runtime-only — it affects the current shell. Pass
``persist=True`` to also write a small ``~/.terminaleyes_prompt``
helper file and arrange for ``~/.bashrc`` to source it, so new
shells inherit the same prompt across reboots.

Persistence layout
------------------
``persist=True`` runs two extra typed commands after the live PS1
change:

  1. ``echo "export PS1='<label><suffix>'" > ~/.terminaleyes_prompt``
     — overwrites a small sourced file so changing the label later
     is just a re-write, never an accumulating list of prior labels.

  2. ``grep -q terminaleyes_prompt ~/.bashrc || echo "[ -f ~/.terminaleyes_prompt ] && . ~/.terminaleyes_prompt" >> ~/.bashrc``
     — idempotent: the ``grep`` short-circuits on the second run
     so ``~/.bashrc`` ends up with exactly one source line.

The split (separate file + one-line .bashrc include) means we
never have to parse or rewrite the user's .bashrc on subsequent
calls; we just rewrite our owned helper file.

Why this exists as its own agent
--------------------------------
The natural intent "change the bash prompt to mini1" requires the
LLM planner to emit a step that types a string containing both
single and double quotes (``PS1='mini1$ '``). The small OCR-model
planner kept truncating its JSON output on the quote characters,
producing 1-step plans that only launched the terminal. Pushing
the quoting into a dedicated agent removes the LLM's exposure to
the special characters — the planner only has to emit
``{"name": "set_prompt", "kwargs": {"label": "mini1", "persist": true}}``.
"""

from __future__ import annotations

import asyncio
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
        persist: bool = False,
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
        if "'" in clean or '"' in clean:
            # We embed the label inside both single- and (for the
            # persisted file write) double-quoted shell strings, so
            # either quote character in the label would terminate
            # the quote and break the shell. Reject loudly rather
            # than silently mangling.
            return SetPromptOutcome(
                success=False,
                reason="label must not contain a single or double quote",
                data={"label": clean},
            )

        # 1) Live: set PS1 for the current shell.
        live_cmd = f"PS1='{clean}{suffix}'"
        res = await TypeAgent(self.ctx).run(
            text=live_cmd, submit=True, post_settle=0.4,
        )
        if not res.success:
            return SetPromptOutcome(
                success=False,
                reason=f"type failed (live PS1): {res.reason}",
                data={"label": clean, "cmd": live_cmd},
            )

        persisted_cmds: list[str] = []
        if persist:
            # 2) Persist: overwrite ~/.terminaleyes_prompt with an
            #    export line; idempotent — re-running with a new
            #    label rewrites this file, doesn't accumulate.
            write_cmd = (
                f"echo \"export PS1='{clean}{suffix}'\""
                f" > ~/.terminaleyes_prompt"
            )
            await asyncio.sleep(0.4)
            r2 = await TypeAgent(self.ctx).run(
                text=write_cmd, submit=True, post_settle=0.4,
            )
            if not r2.success:
                return SetPromptOutcome(
                    success=False,
                    reason=(
                        f"live PS1 set but writing helper file "
                        f"failed: {r2.reason}"
                    ),
                    data={"label": clean, "cmd": live_cmd,
                          "persist_cmd": write_cmd},
                )
            persisted_cmds.append(write_cmd)

            # 3) Ensure ~/.bashrc sources the helper exactly once.
            #    grep -q short-circuits on subsequent runs.
            source_cmd = (
                "grep -q terminaleyes_prompt ~/.bashrc || "
                "echo \"[ -f ~/.terminaleyes_prompt ] && "
                ". ~/.terminaleyes_prompt\" >> ~/.bashrc"
            )
            await asyncio.sleep(0.4)
            r3 = await TypeAgent(self.ctx).run(
                text=source_cmd, submit=True, post_settle=0.4,
            )
            if not r3.success:
                return SetPromptOutcome(
                    success=False,
                    reason=(
                        f"helper file written but .bashrc include "
                        f"failed: {r3.reason}"
                    ),
                    data={"label": clean, "cmd": live_cmd,
                          "persisted_cmds": persisted_cmds},
                )
            persisted_cmds.append(source_cmd)

        try:
            if self.ctx.capture is not None:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label=record_label)
        except Exception as e:
            logger.debug("post-set_prompt capture failed: %s", e)

        if persist:
            print(
                f"   SetPromptAgent: prompt set to {clean!r} "
                f"(persisted via ~/.terminaleyes_prompt + .bashrc)"
            )
            reason = (
                f"set PS1 to {clean!r} and persisted to "
                f"~/.terminaleyes_prompt"
            )
        else:
            print(f"   SetPromptAgent: prompt set to {clean!r}")
            reason = f"set PS1 to {clean!r}"
        return SetPromptOutcome(
            success=True,
            reason=reason,
            data={
                "label": clean, "cmd": live_cmd,
                "persist": persist,
                "persisted_cmds": persisted_cmds,
            },
        )
