"""LaunchAgent — open a desktop app by name on the target machine.

Tier-3 workflow. Uses the same "Activities Overview + type + Enter"
pattern :class:`NavigateAgent` already uses for browsers, but
generalised to any app with a verify-then-act wrapper so failure is
reported cleanly:

  1. ``Esc`` (close any open menu/overview from a previous step).
  2. Tap the launcher hotkey:
        - Linux/GNOME → ``Super``  (opens Activities Overview)
        - macOS       → ``Cmd+Space`` (opens Spotlight)
  3. Type the app name.
  4. ``Enter``.
  5. Wait ``post_launch_settle_ms`` for the app to surface.
  6. Verify the foreground actually changed:
        a. OCR the top bar via :class:`OcrAgent` and substring-match
           against the expected name(s). Fast + deterministic.
        b. If OCR returns nothing usable (low confidence / sparse /
           empty), fall back to :class:`VerifyAgent` with a visual
           question. More tolerant of fonts / dark themes.

The ``app`` argument is run through a small alias map
(:data:`APP_ALIASES`) so callers can pass natural variants ("the
terminal", "shell", "file manager"). Unknown names pass through
verbatim — useful for any app the launcher's app search can find.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.ocr import OcrAgent
from terminaleyes.agents.verify import VerifyAgent

logger = logging.getLogger(__name__)


@dataclass
class LaunchOutcome(Outcome):
    """``data['app']`` carries the canonical app name on success."""


@dataclass(frozen=True)
class _AppAlias:
    type_as: str           # what to type into the launcher
    expect: tuple[str, ...]  # case-insensitive substrings to look for


# Small map of obvious variants → canonical (typed name + expected
# top-bar substrings). Unknown names pass through verbatim.
APP_ALIASES: dict[str, _AppAlias] = {
    "terminal":         _AppAlias("terminal", ("terminal",)),
    "the terminal":     _AppAlias("terminal", ("terminal",)),
    "a terminal":       _AppAlias("terminal", ("terminal",)),
    "shell":            _AppAlias("terminal", ("terminal",)),
    "gnome-terminal":   _AppAlias("gnome-terminal", ("terminal",)),
    "files":            _AppAlias("files", ("files", "nautilus")),
    "the files":        _AppAlias("files", ("files", "nautilus")),
    "file manager":     _AppAlias("files", ("files", "nautilus")),
    "nautilus":         _AppAlias("nautilus", ("files", "nautilus")),
    "calculator":       _AppAlias("calculator", ("calculator",)),
    "calc":             _AppAlias("calculator", ("calculator",)),
    "settings":         _AppAlias("settings", ("settings",)),
    "system settings":  _AppAlias("settings", ("settings",)),
    "text editor":      _AppAlias("text editor", ("text editor", "gedit")),
    "gedit":            _AppAlias("gedit", ("gedit", "text editor")),
    "firefox":          _AppAlias("firefox", ("firefox",)),
    "the firefox":      _AppAlias("firefox", ("firefox",)),
    "chrome":           _AppAlias("google-chrome", ("chrome", "chromium")),
    "google chrome":    _AppAlias("google-chrome", ("chrome", "chromium")),
    "chromium":         _AppAlias("chromium", ("chromium", "chrome")),
    "browser":          _AppAlias("firefox", ("firefox", "chrome")),
    "web browser":      _AppAlias("firefox", ("firefox", "chrome")),
}


class LaunchAgent(Agent):
    """Open a desktop app by name on the target."""

    name = "launch"

    async def run(
        self,
        *,
        app: str,
        platform: str = "linux",
        verify: bool = True,
        post_launch_settle_ms: int = 1800,
        max_attempts: int = 2,
        record_label: str = "launch",
    ) -> LaunchOutcome:
        if self.ctx.keyboard is None:
            return LaunchOutcome(
                success=False, reason="no keyboard in context",
            )
        if not app or not app.strip():
            return LaunchOutcome(
                success=False, reason="empty app name",
            )
        canonical = self._canonicalise(app)
        type_as = canonical.type_as
        expected = canonical.expect

        last_reason = ""
        for attempt in range(1, max_attempts + 1):
            print(
                f"LaunchAgent[{attempt}/{max_attempts}]: opening "
                f"{type_as!r} (alias of {app!r})"
            )
            try:
                await self._send_launch_keystrokes(type_as, platform)
            except Exception as e:
                last_reason = f"launcher keystrokes failed: {e}"
                logger.warning(last_reason)
                continue

            await asyncio.sleep(post_launch_settle_ms / 1000.0)

            if not verify:
                return LaunchOutcome(
                    success=True,
                    reason=f"sent launch keystrokes for {type_as!r} (unverified)",
                    data={"app": type_as, "verified": False},
                )

            verified, where = await self._verify_focus(
                expected=expected, app=app, attempt=attempt,
                record_label=record_label,
            )
            if verified:
                return LaunchOutcome(
                    success=True,
                    reason=(
                        f"opened {type_as!r} (foreground confirmed via {where})"
                    ),
                    data={
                        "app": type_as,
                        "verified": True,
                        "verified_via": where,
                    },
                )
            last_reason = (
                f"opened {type_as!r} but foreground did not match "
                f"(expected one of {list(expected)!r})"
            )
            print(f"   {last_reason}; retrying" if attempt < max_attempts
                  else f"   {last_reason}")

        return LaunchOutcome(
            success=False,
            reason=last_reason or f"failed to open {type_as!r}",
            data={"app": type_as, "verified": False},
        )

    # ───────────────────── canonicalisation ─────────────────────

    def _canonicalise(self, app: str) -> _AppAlias:
        key = app.strip().lower()
        if key in APP_ALIASES:
            return APP_ALIASES[key]
        # Unknown name: type it as-is, expect the same word in the
        # top bar (case-insensitive substring).
        # Strip leading articles ("the ", "a ", "an ") for the
        # expected match — top bar usually shows just the app name.
        base = re.sub(r"^(?:the|a|an)\s+", "", key)
        return _AppAlias(type_as=base, expect=(base,))

    # ───────────────────── keystrokes ─────────────────────

    async def _send_launch_keystrokes(
        self, type_as: str, platform: str,
    ) -> None:
        """Open the platform launcher, type the app name, press Enter."""
        kb = self.ctx.keyboard
        # Close any open menu / overview from a previous step.
        try:
            await kb.send_keystroke("Escape")
        except Exception:
            pass
        await asyncio.sleep(0.20)

        if platform == "macos":
            # Cmd+Space → Spotlight.
            await kb.send_key_combo(["cmd"], "space")
            await asyncio.sleep(0.55)
        else:
            # Bare Super tap → GNOME Activities Overview. The Pi
            # backend accepts an empty key as "modifier-only tap".
            await kb.send_key_combo(["super"], "")
            await asyncio.sleep(0.65)

        await kb.send_text(type_as, secret=False)
        await asyncio.sleep(0.45)
        await kb.send_keystroke("Enter")

    # ───────────────────── verification ─────────────────────

    async def _verify_focus(
        self,
        *,
        expected: tuple[str, ...],
        app: str,
        attempt: int,
        record_label: str,
    ) -> tuple[bool, str]:
        """Two-pass verification.

        1. **OCR the header strip** (top 20%, includes both the
           GNOME top bar AND the window title bar of any
           foregrounded app). Fast and deterministic. A positive
           hit on the expected name short-circuits.
        2. **VerifyAgent** when OCR didn't see the name. Many GNOME
           setups don't surface the focused app's name in the top
           bar at all, so an OCR negative is NOT a reliable
           conclusion — only a visual check is.
        """
        # 1) OCR the header — fast positive signal.
        ocr_outcome = await OcrAgent(self.ctx).run(
            region="header",
            record_label=f"{record_label}_verify_header_{attempt:02d}",
        )
        if ocr_outcome.success and ocr_outcome.data:
            text = (ocr_outcome.data.get("text") or "").lower()
            for needle in expected:
                if needle.lower() in text:
                    return True, f"header OCR ({needle!r} matched)"
            logger.debug(
                "LaunchAgent: header OCR %r — no match for %r; "
                "falling back to VerifyAgent",
                text[:80], expected,
            )
        else:
            logger.debug(
                "LaunchAgent: header OCR failed (%s) — falling "
                "back to VerifyAgent",
                ocr_outcome.reason,
            )

        # 2) Ask the multimodal model. This is the conclusive
        #    negative — many GNOME setups don't expose the focused
        #    app name in any OCR-able region.
        question = (
            f"Look at the screen. Is the FOREGROUND application a "
            f"{app!r}? Decide visually from the window contents and "
            f"any visible window title — NOT every desktop shows "
            f"the focused app's name in the top status bar, so don't "
            f"require that. Answer TRUE only if the foreground is "
            f"unmistakably a {app!r} (e.g. for a terminal: a shell "
            f"prompt, monospaced text, dark window with command-line "
            f"output)."
        )
        v = await VerifyAgent(self.ctx).run(
            question=question, visual_only=True,
            record_label=f"{record_label}_verify_visual_{attempt:02d}",
        )
        if v:
            return True, f"VerifyAgent ({v.reason})"
        return False, f"VerifyAgent ({v.reason})"
