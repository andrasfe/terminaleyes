"""SaveAsAgent — drive the platform Save-As flow as a single step.

Tier-3 workflow primitive. Most apps' Save-As is the same sequence:

  1. Ctrl+Shift+S (or Cmd+Shift+S on macOS) — Save AS, NOT plain
     Save. Plain Ctrl+S silently re-saves an already-saved document
     and never opens the dialog, which leaves the path + Enter
     keystrokes to fall into the document body instead.
  2. Wait for the save dialog to mount.
  3. Type the destination path (overwriting any pre-selected
     default filename).
  4. Enter to commit.
  5. Optional second Enter to dismiss the "Keep current format / use
     native format" mismatch prompt that LibreOffice (and similar
     apps) raise when the chosen extension isn't the document's
     native one.

Folding this into one agent keeps the planner honest — the LLM
otherwise tends to emit only 3-step plans for "open X, type Y, save
as PATH", dropping the path-type and Enter that actually complete
the save.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class SaveAsOutcome(Outcome):
    pass


class SaveAsAgent(Agent):
    """Save the current document to a given path via Ctrl/Cmd+S."""

    name = "save_as"

    async def run(
        self,
        *,
        path: str,
        platform: str = "linux",
        dialog_settle: float = 1.4,
        commit_settle: float = 1.2,
        confirm_format_prompt: bool = True,
        record_label: str = "save_as",
    ) -> SaveAsOutcome:
        if self.ctx.keyboard is None:
            return SaveAsOutcome(
                success=False, reason="no keyboard in context",
                data={"path": path},
            )
        target = (path or "").strip()
        if not target:
            return SaveAsOutcome(
                success=False, reason="empty path",
                data={"path": ""},
            )

        primary_mod = "cmd" if platform == "macos" else "ctrl"
        save_mods = [primary_mod, "shift"]
        chord_name = f"{primary_mod}+shift+s"
        try:
            await self.ctx.keyboard.send_key_combo(save_mods, "s")
        except Exception as e:
            return SaveAsOutcome(
                success=False, reason=f"{chord_name} failed: {e}",
                data={"path": target},
            )
        await asyncio.sleep(dialog_settle)

        # The save dialog opens with the default filename pre-
        # selected in the name field. Typing replaces it on most
        # toolkits (GTK GtkFileChooser, LibreOffice's own dialog).
        # ``warmup=False`` would matter for browsers; here we keep
        # the default warmup so the first character isn't dropped
        # over BT-HID.
        try:
            await self.ctx.keyboard.send_text(target)
        except TypeError:
            await self.ctx.keyboard.send_text(target)
        except Exception as e:
            return SaveAsOutcome(
                success=False, reason=f"typing path failed: {e}",
                data={"path": target},
            )
        await asyncio.sleep(0.4)

        try:
            await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:
            return SaveAsOutcome(
                success=False, reason=f"Enter (commit) failed: {e}",
                data={"path": target},
            )
        await asyncio.sleep(commit_settle)

        if confirm_format_prompt:
            try:
                await self.ctx.keyboard.send_keystroke("Enter")
            except Exception as e:
                logger.debug("format-confirm Enter failed: %s", e)
            await asyncio.sleep(0.6)

        try:
            if self.ctx.capture is not None:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(frame.image, label=record_label)
        except Exception as e:
            logger.debug("post-save capture failed: %s", e)

        print(f"   SaveAsAgent: requested save -> {target}")
        return SaveAsOutcome(
            success=True,
            reason=f"saved to {target}",
            data={"path": target, "platform": platform},
        )
