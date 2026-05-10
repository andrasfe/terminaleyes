"""KeyComboAgent — send a keyboard shortcut.

Tier-1 atomic primitive. Wraps :meth:`KeyboardOutput.send_key_combo`
behind a typed agent so the controller / LLM-planner can emit it
directly in a plan.

Use this for actions with a known platform shortcut:

  * close window — Alt+F4 (Linux) / Cmd+W (macOS)
  * save — Ctrl+S / Cmd+S
  * copy / paste / cut — Ctrl+{C,V,X} / Cmd+{C,V,X}
  * new tab / close tab / next tab — Ctrl+{T,W,Tab}
  * undo / redo — Ctrl+Z / Ctrl+Shift+Z
  * quit app — Ctrl+Q / Cmd+Q
  * switch window — Alt+Tab / Cmd+Tab
  * maximise / minimise — Super+Up / Super+H

Visual clicks on close/save/copy buttons are wrong: they're slow,
unreliable on dark themes, and the homer wastes 5 retries before
declaring failure (as happened with "close the terminal window").
A single :class:`KeyComboAgent` step is one keystroke.

Inputs:

  * ``modifiers`` — list like ``["ctrl"]``, ``["alt"]``, ``["ctrl",
    "shift"]``, ``["super"]``, ``["cmd"]``. Empty list = bare key.
  * ``key`` — the non-modifier key, e.g. ``"F4"``, ``"s"``, ``"Tab"``.
    Empty string = a modifier-only tap (e.g. bare Super for GNOME
    Activities Overview).
  * ``platform`` — when set, ``ctrl`` is auto-translated to ``cmd``
    on macOS for the most common shortcuts (Ctrl+S, Ctrl+C, Ctrl+V,
    etc). Pass ``platform=None`` to disable auto-translation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


@dataclass
class KeyComboOutcome(Outcome):
    """``data['sent']`` carries the canonical chord string."""


# Shortcuts that mean the same thing on macOS but use Cmd instead
# of Ctrl. Anything not in this set is sent verbatim — the caller
# can already specify ``["cmd"]`` directly when they know better.
_CTRL_TO_CMD_SAFE_KEYS = frozenset({
    "a", "c", "v", "x", "z", "s", "f", "n", "o", "p", "t", "w",
    "q", "r", "l", "Tab",
})


class KeyComboAgent(Agent):
    """Send a single keyboard shortcut to the target."""

    name = "keys"

    async def run(
        self,
        *,
        modifiers: list[str] | None = None,
        key: str = "",
        platform: str | None = "linux",
        record_label: str = "keys",
    ) -> KeyComboOutcome:
        if self.ctx.keyboard is None:
            return KeyComboOutcome(
                success=False, reason="no keyboard in context",
                data={"sent": ""},
            )
        mods = [m.strip().lower() for m in (modifiers or []) if m.strip()]
        key = (key or "").strip()
        if not mods and not key:
            return KeyComboOutcome(
                success=False, reason="empty modifiers + key",
                data={"sent": ""},
            )

        # macOS auto-translation for the common Ctrl-* shortcuts.
        if (
            platform == "macos"
            and "ctrl" in mods
            and "cmd" not in mods
            and key in _CTRL_TO_CMD_SAFE_KEYS
        ):
            mods = ["cmd" if m == "ctrl" else m for m in mods]

        chord = self._format_chord(mods, key)
        try:
            await self.ctx.keyboard.send_key_combo(mods, key)
        except Exception as e:
            return KeyComboOutcome(
                success=False, reason=f"send_key_combo failed: {e}",
                data={"sent": chord},
            )
        # Visual proof for the UI replay log. We capture AFTER the
        # combo on a best-effort basis so the user sees the result.
        try:
            if self.ctx.capture is not None:
                frame = await self.ctx.capture.capture_frame()
                self.ctx.record_frame(
                    frame.image, label=f"{record_label}_{chord}",
                )
        except Exception as e:
            logger.debug("post-combo capture failed: %s", e)
        print(f"   KeyComboAgent: sent {chord}")
        return KeyComboOutcome(
            success=True, reason=f"sent {chord}",
            data={"sent": chord, "modifiers": mods, "key": key},
        )

    @staticmethod
    def _format_chord(mods: list[str], key: str) -> str:
        parts = [m.capitalize() for m in mods]
        if key:
            parts.append(key)
        return "+".join(parts) if parts else "(empty)"
