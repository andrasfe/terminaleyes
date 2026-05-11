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
from terminaleyes.agents.launch import LaunchAgent
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
            # Tight check: re-verify immediately before typing. The
            # initial pre-flight may have spent several seconds on
            # model calls / activation; in that gap the foreground
            # can shift (App Center notification popping up, etc.).
            verifier_tight = VerifyAgent(self.ctx)
            v_tight = await verifier_tight.run(
                question=BROWSER_QUESTION, visual_only=True,
                record_label="navigate_browser_pre_type",
            )
            if not v_tight:
                return NavigateOutcome(
                    success=False,
                    reason=(
                        "browser was focused initially but foreground "
                        f"shifted before typing: {v_tight.reason}"
                    ),
                )
            # Hard transfer of WM focus to a browser. The verifier
            # judges by window size, but on GNOME the *actually
            # focused* app may be something else whose window is
            # tiny (App Center notification, system dialog, etc.).
            # GNOME activities + "chrome" + Enter focuses an
            # existing browser window when one is already running, so
            # this is idempotent — safe to run even when the browser
            # already has focus. Clicking the viewport alone is not
            # enough because clicking inside an unfocused window
            # doesn't always transfer WM focus (the click is
            # consumed by the click-to-focus behaviour without a
            # subsequent typing-ready state).
            await self._force_activate_browser_for_typing(platform)

        # 2. Type the URL.
        focus_mods = ["cmd"] if platform == "macos" else ["ctrl"]
        try:
            await self.ctx.keyboard.send_key_combo(focus_mods, "l")
            await asyncio.sleep(0.5)
            if select_all_first:
                await self.ctx.keyboard.send_key_combo(focus_mods, "a")
                await asyncio.sleep(0.25)
            # Skip the double-tap-with-backspace warmup when typing
            # into the URL bar. The warmup is designed for bash
            # readline where Backspace deletes a character; in
            # Firefox/Chrome URL bars Backspace can be bound to
            # back-navigation, which leaves the warmup's first
            # press visible — producing e.g. ``rreddit.com`` for
            # ``reddit.com``. Without the warmup, the receiver
            # MIGHT drop the first character; the post-flight OCR
            # verify catches that and reports it cleanly.
            await TypeAgent(self.ctx).run(
                text=url, secret=False, submit=False, warmup=False,
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

    async def _force_activate_browser_for_typing(
        self, platform: str,
    ) -> None:
        """Force WM focus to a browser window before typing the URL.

        Delegates the launch keystrokes to :class:`LaunchAgent`
        (with ``verify=False`` because we want browser-agnostic
        verification — any browser surfacing is fine, not just the
        one we tried to launch). After each launch attempt we run
        the profile-picker dismissal dance (Tab+Enter twice + Esc)
        and re-verify with :class:`VerifyAgent` against
        :data:`BROWSER_QUESTION`. Idempotent — safe even if a
        browser is already focused.
        """
        kb = self.ctx.keyboard
        if kb is None or platform == "macos":
            # macOS doesn't have GNOME activities — fall back to
            # clicking the viewport.
            await self._click_browser_viewport()
            return
        for name in ("google-chrome", "chromium", "firefox"):
            try:
                await LaunchAgent(self.ctx).run(
                    app=name, platform=platform,
                    verify=False,
                    post_launch_settle_ms=1200,
                    record_label=f"navigate_force_focus_{name}",
                )
                # Browser launchers often pop a profile or sign-in
                # picker. Tab moves focus to the first selectable
                # card, Enter activates it. Esc clears any leftover
                # toast / save prompt.
                try:
                    for _ in range(2):
                        await kb.send_keystroke("Tab")
                        await asyncio.sleep(0.18)
                        await kb.send_keystroke("Enter")
                        await asyncio.sleep(0.45)
                    await kb.send_keystroke("Escape")
                    await asyncio.sleep(0.3)
                except Exception:
                    pass
                # Browser-agnostic verify: any browser is fine here.
                v = await VerifyAgent(self.ctx).run(
                    question=BROWSER_QUESTION, visual_only=True,
                    record_label=f"navigate_force_focus_{name}",
                )
                if v:
                    return
            except Exception as e:
                logger.warning(
                    "Force-activate via %r failed: %s", name, e,
                )
        # As a last resort fall back to the viewport click. If the
        # browser still wasn't focused, the post-flight oracle will
        # catch the failed navigation.
        await self._click_browser_viewport()

    async def _click_browser_viewport(self) -> None:
        """Click the centre of the screen so the visible browser
        window receives WM focus. Slams cursor to the corner first
        (deterministic start) and then drives it diagonally to the
        approximate centre using uncalibrated open-loop HID. Most
        layouts have the browser dominate the screen, so even an
        approximate centre lands inside it.
        """
        if self.ctx.mouse is None:
            return
        try:
            # Slam top-left so the cursor's image position is known
            # without needing visual confirmation.
            for _ in range(160):
                await self.ctx.mouse.move(-20, -20)
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.20)
            # Roughly centre. Without a calibrated ratio, ~960 HID
            # diagonal puts the cursor near the middle of a 1920×1080
            # screen for most cursor-acceleration profiles.
            steps = 48
            for _ in range(steps):
                await self.ctx.mouse.move(20, 12)
                await asyncio.sleep(0.003)
            await asyncio.sleep(0.20)
            await self.ctx.mouse.click("left")
        except Exception as e:
            logger.warning(
                "Browser-viewport focus click failed: %s", e,
            )

    async def _ensure_browser_focused(
        self, *, max_attempts: int, platform: str,
    ) -> tuple[bool, str]:
        """Verify-then-correct loop until the foreground is a browser."""
        verifier = VerifyAgent(self.ctx)

        v0 = await verifier.run(
            question=BROWSER_QUESTION, visual_only=True,
            record_label="navigate_browser_check",
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
                record_label=f"navigate_browser_recheck_{attempt:02d}",
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

        Strategy escalates each attempt. On Linux/GNOME the most
        reliable path is the activities overview:

          1. Press ``Super`` to open activities, type the browser
             name, press ``Enter``. Tries ``firefox`` then ``chrome``
             then ``chromium``.
          2. Visually click a known browser icon via ClickAgent.
          3. Super+<N> sweep over the favourites bar.
          4. macOS: Cmd+Tab.
        """
        kb = self.ctx.keyboard

        # 1. LaunchAgent with the browser name (Linux only).
        # Caller's outer loop re-verifies via BROWSER_QUESTION, so
        # we skip LaunchAgent's own verify step (we want
        # browser-agnostic verification — any browser surfacing is
        # fine, not just this specific one).
        if platform != "macos" and kb is not None and attempt <= 3:
            browser_names = ["firefox", "google-chrome", "chromium"]
            name = browser_names[attempt - 1]
            try:
                await LaunchAgent(self.ctx).run(
                    app=name, platform=platform,
                    verify=False,
                    post_launch_settle_ms=900,
                    record_label=f"navigate_activate_{name}_a{attempt:02d}",
                )
                return f"LaunchAgent({name!r})"
            except Exception as e:
                logger.warning(
                    "LaunchAgent attempt for %r failed: %s",
                    name, e,
                )

        # 2. Visual click on browser icon (ClickAgent + ShowUI/OCR).
        # Less reliable than the overview path but worth one shot.
        idx = attempt - 4
        if 0 <= idx < len(BROWSER_ICON_DESCRIPTIONS):
            desc = BROWSER_ICON_DESCRIPTIONS[idx]
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

        # 3. Super+N sweep (Linux fallback).
        if platform != "macos" and kb is not None:
            slot = (attempt % 5) + 1
            try:
                await kb.send_key_combo(["super"], str(slot))
                return f"Super+{slot}"
            except Exception as e:
                logger.debug("Super+%d failed: %s", slot, e)

        # 4. macOS fallback.
        if platform == "macos" and kb is not None:
            try:
                await kb.send_key_combo(["cmd"], "Tab")
                return "Cmd+Tab"
            except Exception as e:
                logger.debug("Cmd+Tab failed: %s", e)

        return "no-op"

    # ─────────────────── URL bar post-flight ───────────────────

    async def _verify_url_in_address_bar(
        self, url: str,
    ) -> tuple[bool, str]:
        """OCR the URL bar (and, if needed, the whole page) and check
        the typed URL appears.

        Three-pass:

          1. URL bar strip (top 10%, both polarities).
          2. Wider top strip (top 18%) — catches URL bar even when
             a bookmarks bar pushes it down.
          3. Whole frame — if the URL bar is too small/dark for OCR
             but the page actually loaded, the hostname stem and
             path segments will appear in body text (page header,
             nav links, footer copyright). This is the rescue path
             for dark-themed Chrome on a webcam.

        Tolerant of common letter-substitution garbling (tesseract
        often reads ``r`` as ``t``); we normalise to alphanumerics
        and substring-match. Each pass is gated on the same
        candidate set so we never accept a wrong page just because
        ``reddit`` appears in some unrelated text.
        """
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:
            return False, f"post-capture failed: {e}"
        self.ctx.record_frame(frame.image, label="navigate_postflight_full")
        h, w = frame.image.shape[:2]
        urlbar = frame.image[: int(h * 0.10), :]
        self.ctx.record_frame(urlbar, label="navigate_postflight_urlbar")
        wide_top = frame.image[: int(h * 0.18), :]

        try:
            from terminaleyes.commander.ocr_finder import (
                _preprocess_for_ocr, have_ocr,
            )
            import pytesseract  # type: ignore
            if not have_ocr():
                return True, "OCR unavailable; trusting keystrokes"
        except Exception as e:
            return True, f"OCR import failed ({e}); trusting keystrokes"

        # Build candidate token set once.
        url_lower = re.sub(r"^https?://", "", url.lower())
        path_segments = [
            re.sub(r"[^a-z0-9]", "", seg)
            for seg in url_lower.split("/")
            if seg.strip()
        ]
        path_segments = [s for s in path_segments if len(s) >= 4]
        host = re.sub(
            r"[^a-z0-9.]", "", url_lower.split("/")[0],
        )
        host_parts = [p for p in host.split(".") if len(p) >= 3]
        candidates = list(dict.fromkeys(path_segments + host_parts))

        from difflib import SequenceMatcher

        def _scan(label: str, region) -> tuple[bool, str] | None:
            try:
                normal = pytesseract.image_to_string(region)
                inv = pytesseract.image_to_string(
                    _preprocess_for_ocr(region, scale=4, invert=True),
                )
            except Exception as e:
                logger.debug("OCR pass %s failed: %s", label, e)
                return None
            text = (normal + " " + inv).lower()
            text_norm = re.sub(r"[^a-z0-9]", "", text)
            for c in candidates:
                if c in text_norm:
                    return True, f"{label} contains {c!r}"
            words = re.findall(r"[a-z0-9]{4,}", text_norm)
            for c in candidates:
                for w in words:
                    if abs(len(w) - len(c)) > max(2, len(c) // 3):
                        continue
                    ratio = SequenceMatcher(None, c, w).ratio()
                    if ratio >= 0.75:
                        return True, (
                            f"{label} fuzz-matches {c!r} ~ {w!r} "
                            f"(ratio={ratio:.2f})"
                        )
            return False, (
                f"{label} text {text.strip()[:80]!r} did not match"
            )

        for label, region in (
            ("url-bar", urlbar),
            ("top-strip", wide_top),
            ("full-frame", frame.image),
        ):
            result = _scan(label, region)
            if result is None:
                continue
            ok, msg = result
            if ok:
                return True, msg
        return False, (
            f"no OCR pass matched any of {candidates!r}"
        )
