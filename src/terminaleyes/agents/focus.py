"""Focus agent — bring the foreground app to a centred / maximised state.

Uses :class:`VerifyAgent` to decide visually whether the main
application window is "in focus and centred", and if not, takes a
sequence of corrective actions:

  1. Send the WM "maximise focused window" combo (GNOME ``Super+Up``;
     fall back to ``Alt+F10`` which most EWMH-compliant WMs honour).
  2. Re-verify after a brief settle.
  3. If still not centred, click in the image centre to give the
     window keyboard focus, then retry the maximise combo.
  4. If still not centred, send ``Super+h`` to unhide / Super to open
     the activities overview as a last resort.

Each attempt re-verifies. We never click without first having a
visual confirmation, and we abort cleanly after ``max_attempts``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.verify import VerifyAgent

logger = logging.getLogger(__name__)


@dataclass
class FocusOutcome(Outcome):
    pass


class FocusAgent(Agent):
    """Verify-then-fix the foreground window's centring."""

    name = "focus"

    QUESTION = (
        "Look at the screen. Decide whether the foreground application "
        "window is maximised/centred and READY for interaction.\n\n"
        "IMPORTANT — these are NOT 'desktop borders' and you must NOT "
        "treat them as evidence the window isn't maximised:\n"
        "  * the system dock / taskbar / launcher (e.g. the strip of "
        "app icons on the left side of an Ubuntu/GNOME desktop)\n"
        "  * the top bar / menu bar / status bar at the very top\n"
        "  * the system tray / notification area at the bottom-right\n"
        "  * a vertical app-switcher strip\n"
        "These persistent OS chrome elements are present even on a "
        "fully maximised window. Their presence is normal.\n\n"
        "Answer FALSE only if ANY of these are true:\n"
        "  * the screen is black, dark, dim, blurred, or appears off/asleep\n"
        "  * the screen shows a screensaver, lock screen, or login prompt\n"
        "  * no clear application UI (text, controls, menus, content) is "
        "visible at all\n"
        "  * the foreground window is genuinely small (e.g. a floating "
        "popup that occupies less than half the area excluding the "
        "OS dock/taskbar) or sits in just one quadrant\n\n"
        "Answer TRUE when ALL of these are true:\n"
        "  * clear application UI is visible (text/controls/content)\n"
        "  * the foreground window fills most of the area NOT occupied "
        "by the OS dock/taskbar/menu bar\n"
        "  * the user could comfortably interact with it as the primary "
        "window right now"
    )

    AWAKE_QUESTION = (
        "Is the screen currently showing clear, readable application "
        "content — i.e. NOT dark, blurred, off, asleep, on a screensaver, "
        "or on a lock/login screen? Answer true only if normal app UI "
        "is visible."
    )

    async def run(
        self,
        *,
        max_attempts: int = 3,
        platform: str = "linux",
        settle_seconds: float = 0.7,
        wake_first: bool = True,
    ) -> FocusOutcome:
        if self.ctx.keyboard is None:
            return FocusOutcome(
                success=False, reason="no keyboard in context",
            )

        verifier = VerifyAgent(self.ctx)

        if wake_first:
            await self._wake()

        # Pre-check: is the screen even showing content? Refuses to
        # answer the "is it centred" question until we have something
        # to look at — prevents hallucinated "yes" verdicts on dark
        # frames.
        awake = await verifier.run(
            question=self.AWAKE_QUESTION, visual_only=True,
            record_label="focus_awake_check",
        )
        print(
            f"FocusAgent: awake check — awake={bool(awake)} "
            f"reason={awake.reason!r}"
        )
        if not awake:
            # Try one more wake nudge cycle, then re-check.
            await self._wake()
            awake = await verifier.run(
                question=self.AWAKE_QUESTION, visual_only=True,
                record_label="focus_awake_recheck",
            )
            print(
                f"FocusAgent: awake re-check — awake={bool(awake)} "
                f"reason={awake.reason!r}"
            )
        if not awake:
            return FocusOutcome(
                success=False,
                reason=(
                    f"screen is not awake / showing no content "
                    f"({awake.reason}); won't act"
                ),
                data={"attempts": 0, "awake": False},
            )

        # 0. Initial check.
        v0 = await verifier.run(
            question=self.QUESTION, visual_only=True,
            record_label="focus_initial_check",
        )
        print(
            f"FocusAgent: initial check — focused={bool(v0)} "
            f"reason={v0.reason!r}"
        )
        if v0:
            return FocusOutcome(
                success=True,
                reason=f"already focused: {v0.reason}",
                data={"attempts": 0},
            )

        for attempt in range(1, max_attempts + 1):
            await self._apply_action(attempt, platform)
            await asyncio.sleep(settle_seconds)
            v = await verifier.run(
                question=self.QUESTION, visual_only=True,
                record_label=f"focus_recheck_{attempt:02d}",
            )
            print(
                f"FocusAgent: attempt {attempt}/{max_attempts} — "
                f"focused={bool(v)} reason={v.reason!r}"
            )
            if v:
                return FocusOutcome(
                    success=True,
                    reason=f"focused after attempt {attempt}: {v.reason}",
                    data={"attempts": attempt},
                )

        return FocusOutcome(
            success=False,
            reason=(
                f"still not centred/focused after {max_attempts} "
                f"attempts"
            ),
            data={"attempts": max_attempts},
        )

    async def _wake(self) -> None:
        """Wake the monitor / dismiss screensaver before checking."""
        kb = self.ctx.keyboard
        mouse = self.ctx.mouse
        try:
            if mouse is not None:
                for _ in range(3):
                    await mouse.move(20, 0)
                    await asyncio.sleep(0.04)
                    await mouse.move(-20, 0)
                    await asyncio.sleep(0.04)
            if kb is not None:
                # A safe-no-op key on most desktops.
                await kb.send_keystroke("Down")
        except Exception as e:
            logger.warning("Wake step failed: %s", e)
        await asyncio.sleep(0.8)

    async def _apply_action(self, attempt: int, platform: str) -> None:
        """Send one corrective action.

        Strategy escalates each attempt:

          1. ``Escape`` (gentle dismiss of any open menu/dropdown) +
             maximise.
          2. Click in the screen centre to transfer WM focus to the
             visible app + maximise.
          3. **Last resort, destructive**: ``Alt+F4`` to close the
             stuck foreground window (e.g. an App Center pop-up that
             holds keyboard focus while a real app is visible behind),
             then maximise whatever's now in front.

        Increase ``max_attempts`` to a 4th attempt to also try the
        EWMH ``Alt+F10`` maximise hint as a final non-destructive
        try after the close.
        """
        kb = self.ctx.keyboard
        mouse = self.ctx.mouse
        if attempt == 1:
            # Gentle dismissal of an open menu / dropdown first.
            try:
                await kb.send_keystroke("Escape")
                await asyncio.sleep(0.15)
            except Exception:
                pass
            # Then maximise focused window.
            try:
                if platform == "macos":
                    print(
                        "FocusAgent: macOS has no portable maximise "
                        "combo; sending Option+Cmd+F (fullscreen "
                        "toggle) — be careful."
                    )
                    await kb.send_key_combo(["alt", "cmd"], "f")
                else:
                    await kb.send_key_combo(["super"], "Up")
            except Exception as e:
                logger.warning("Maximise combo failed: %s", e)
        elif attempt == 2:
            # Click image centre to transfer WM focus to the visible
            # app, then retry maximise.
            try:
                if mouse is not None:
                    for dx, dy in [(80, 80)] * 6:
                        await mouse.move(dx, dy)
                        await asyncio.sleep(0.02)
                    await mouse.click("left")
                    await asyncio.sleep(0.2)
            except Exception as e:
                logger.warning("Centre-click step failed: %s", e)
            try:
                if platform == "macos":
                    await kb.send_key_combo(["alt", "cmd"], "f")
                else:
                    await kb.send_key_combo(["super"], "Up")
            except Exception as e:
                logger.warning("Maximise combo (retry) failed: %s", e)
        elif attempt == 3:
            # Last-resort DESTRUCTIVE: close the stuck foreground
            # window. Useful when a small popup (App Center,
            # software-updater toast, modal dialog) holds WM focus
            # while a real app is visible behind it. After Alt+F4
            # closes the popup, the previously-behind app becomes
            # the foreground naturally — then we maximise it.
            print(
                "FocusAgent: attempt 3 — closing focused window with "
                "Alt+F4 (destructive last resort)"
            )
            try:
                await kb.send_key_combo(["alt"], "F4")
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("Alt+F4 close failed: %s", e)
            try:
                if platform == "macos":
                    await kb.send_key_combo(["alt", "cmd"], "f")
                else:
                    await kb.send_key_combo(["super"], "Up")
            except Exception as e:
                logger.warning("Post-close maximise failed: %s", e)
        else:
            # 4th+ attempt (only fires when max_attempts is raised
            # above the default 3): EWMH maximise hint.
            try:
                if platform == "macos":
                    await kb.send_key_combo(["alt", "cmd"], "f")
                else:
                    await kb.send_key_combo(["alt"], "F10")
            except Exception as e:
                logger.warning("Fallback combo failed: %s", e)
