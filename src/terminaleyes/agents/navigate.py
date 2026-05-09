"""NavigateAgent — type a URL into a browser address bar.

Verify-then-act flow:

  1. **Pre-flight** — VerifyAgent decides whether the foreground app
     is a web browser (URL bar visible, tab strip, page content). If
     not, try to bring a browser to focus:
       a. ClickAgent on a "Firefox browser icon" / "Chrome browser
          icon" in the taskbar / dock (visual).
       b. Fall back to ``Super+1`` ... ``Super+9`` GNOME activation,
          re-verifying after each.
  2. **Type** — Ctrl+L → Ctrl+A → text → Enter.
  3. **Post-flight** — OCR the URL bar; if the typed URL doesn't
     appear, fail explicitly. No more silent successes when keystrokes
     went into the wrong app.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.type_text import TypeAgent
from terminaleyes.agents.verify import VerifyAgent

logger = logging.getLogger(__name__)


@dataclass
class NavigateOutcome(Outcome):
    pass


BROWSER_QUESTION = (
    "Look at the screen. Is the FOREGROUND application a web "
    "browser (visible URL/address bar at the top, browser tab strip, "
    "and a web page rendered in the body)? Answer TRUE only if the "
    "foreground app is unmistakably a browser. If the foreground is "
    "a settings dialog, file manager, terminal, software updater, "
    "media player, etc., answer FALSE — even if a browser window is "
    "partially visible behind it."
)

BROWSER_ICON_DESCRIPTIONS = [
    "the Firefox browser icon in the taskbar",
    "the Google Chrome browser icon in the taskbar",
    "the Firefox icon in the dock on the left",
    "the Chrome icon in the dock on the left",
    "the orange Firefox icon",
    "the colorful Chrome icon",
]


class NavigateAgent(Agent):
    """Drive a URL into a browser address bar, verifying both ends."""

    name = "navigate"

    async def run(
        self,
        *,
        url: str,
        platform: str = "linux",
        select_all_first: bool = True,
        post_settle: float = 1.8,
        ensure_browser: bool = True,
        verify_after: bool = True,
        max_focus_attempts: int = 4,
    ) -> NavigateOutcome:
        if self.ctx.keyboard is None:
            return NavigateOutcome(
                success=False, reason="no keyboard in context",
            )
        if not url:
            return NavigateOutcome(
                success=False, reason="empty url",
            )

        # 1. Pre-flight: ensure a browser is the foreground app.
        if ensure_browser:
            ok, reason = await self._ensure_browser_focused(
                max_attempts=max_focus_attempts, platform=platform,
            )
            if not ok:
                return NavigateOutcome(
                    success=False,
                    reason=f"could not focus a browser: {reason}",
                )

        # 2. Type the URL.
        focus_mods = ["cmd"] if platform == "macos" else ["ctrl"]
        try:
            await self.ctx.keyboard.send_key_combo(focus_mods, "l")
            await asyncio.sleep(0.5)
            if select_all_first:
                await self.ctx.keyboard.send_key_combo(focus_mods, "a")
                await asyncio.sleep(0.25)
            await TypeAgent(self.ctx).run(
                text=url, secret=False, submit=False,
            )
            await asyncio.sleep(0.4)
            await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:
            logger.warning("NavigateAgent typing failed: %s", e)
            return NavigateOutcome(
                success=False, reason=f"send failed: {e}",
            )

        await asyncio.sleep(post_settle)

        # 3. Post-flight: confirm the URL actually appeared.
        if verify_after and self.ctx.capture is not None:
            verified, vreason = await self._verify_url_in_address_bar(
                url,
            )
            if not verified:
                return NavigateOutcome(
                    success=False,
                    reason=(
                        f"typed URL but address bar does NOT confirm "
                        f"navigation ({vreason})"
                    ),
                    data={"url": url, "verified": False},
                )
            return NavigateOutcome(
                success=True,
                reason=f"navigated to {url!r} ({vreason})",
                data={"url": url, "verified": True},
            )

        return NavigateOutcome(
            success=True,
            reason=f"sent navigation keystrokes for {url!r} (unverified)",
            data={"url": url, "verified": False},
        )

    # ─────────────────── browser-focus pre-flight ───────────────────

    async def _ensure_browser_focused(
        self, *, max_attempts: int, platform: str,
    ) -> tuple[bool, str]:
        """Verify-then-correct loop until the foreground is a browser."""
        verifier = VerifyAgent(self.ctx)

        v0 = await verifier.run(
            question=BROWSER_QUESTION, visual_only=True,
        )
        print(
            f"NavigateAgent: browser check — is_browser={bool(v0)} "
            f"reason={v0.reason!r}"
        )
        if v0:
            return True, v0.reason

        for attempt in range(1, max_attempts + 1):
            print(
                f"NavigateAgent: focus attempt {attempt}/{max_attempts} "
                "— activating a browser"
            )
            method = await self._activate_browser(attempt, platform)
            await asyncio.sleep(0.9)
            v = await verifier.run(
                question=BROWSER_QUESTION, visual_only=True,
            )
            print(
                f"NavigateAgent: re-check — is_browser={bool(v)} "
                f"({method}) reason={v.reason!r}"
            )
            if v:
                return True, f"activated via {method}; {v.reason}"
        return False, (
            f"{max_attempts} activation attempts did not bring a "
            "browser to the foreground"
        )

    async def _activate_browser(
        self, attempt: int, platform: str,
    ) -> str:
        """Try one corrective action to bring a browser to focus.

        Strategy escalates each attempt:
          1. Visually click a known browser icon via ClickAgent.
          2. Try the next-most-likely browser description.
          3. Fall back to Super+N (or Cmd+Tab on macOS).
          4. Repeat the visual click with a different description.
        """
        # 1. Visual click on browser icon — most reliable when the
        # taskbar/dock is visible.
        if attempt <= len(BROWSER_ICON_DESCRIPTIONS):
            desc = BROWSER_ICON_DESCRIPTIONS[attempt - 1]
            try:
                from terminaleyes.agents.click import ClickAgent
                outcome = await ClickAgent(self.ctx).run(
                    target=desc, button="left",
                )
                if outcome:
                    return f"clicked {desc!r}"
            except Exception as e:
                logger.debug(
                    "Browser-icon click attempt failed: %s", e,
                )

        # 2. GNOME / Ubuntu: Super+<N> activates favourites in order.
        # Try a small sweep — a browser is usually at position 1, 2,
        # or 3.
        if platform != "macos" and self.ctx.keyboard is not None:
            slot = (attempt - len(BROWSER_ICON_DESCRIPTIONS)) % 4 + 1
            try:
                await self.ctx.keyboard.send_key_combo(
                    ["super"], str(slot),
                )
                return f"Super+{slot}"
            except Exception as e:
                logger.debug("Super+%d failed: %s", slot, e)

        # 3. macOS fallback: Cmd+Tab once.
        if platform == "macos" and self.ctx.keyboard is not None:
            try:
                await self.ctx.keyboard.send_key_combo(["cmd"], "Tab")
                return "Cmd+Tab"
            except Exception as e:
                logger.debug("Cmd+Tab failed: %s", e)

        return "no-op"

    # ─────────────────── URL bar post-flight ───────────────────

    async def _verify_url_in_address_bar(
        self, url: str,
    ) -> tuple[bool, str]:
        """OCR the top strip and check the typed URL appears there.

        Tolerant of common letter-substitution garbling (tesseract
        often reads ``r`` as ``t``); we normalise to alphanumerics
        and substring-match.
        """
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:
            return False, f"post-capture failed: {e}"
        h, w = frame.image.shape[:2]
        urlbar = frame.image[: int(h * 0.10), :]

        # Try OCR (with both polarities).
        try:
            from terminaleyes.commander.ocr_finder import (
                _preprocess_for_ocr, have_ocr,
            )
            import pytesseract  # type: ignore
            if not have_ocr():
                return True, "OCR unavailable; trusting keystrokes"
            normal = pytesseract.image_to_string(urlbar)
            inv = pytesseract.image_to_string(
                _preprocess_for_ocr(urlbar, scale=4, invert=True),
            )
        except Exception as e:
            return True, f"OCR failed ({e}); trusting keystrokes"

        text = (normal + " " + inv).lower()
        text_norm = re.sub(r"[^a-z0-9]", "", text)
        # The "domain core" is more stable than the full URL — trim
        # protocol / path and pick the most distinctive portion.
        core = url.lower()
        core = re.sub(r"^https?://", "", core)
        core = re.sub(r"[^a-z0-9]", "", core)
        # If the URL is long, look for the first 8 chars (subdomain
        # or hostname start) AND the last 8 chars (path tail).
        candidates = [core]
        if len(core) > 12:
            candidates.append(core[:8])
            candidates.append(core[-8:])

        # Allow tesseract's classic r→t substitution.
        for c in candidates:
            if c in text_norm:
                return True, f"address bar contains {c!r}"
            if c.replace("r", "t") in text_norm:
                return True, f"address bar contains {c!r} (r/t sub)"
            if c.replace("t", "r") in text_norm:
                return True, f"address bar contains {c!r} (t/r sub)"
        return False, (
            f"address bar text {text.strip()[:120]!r} does not "
            f"contain {core!r}"
        )
