"""DismissModalsAgent — close any modal/popup blocking the interface.

Loop:

  1. Capture a frame.
  2. Ask the multimodal model whether a modal/dialog/popup/profile-
     picker/notification is on screen, and if so, whether it has a
     visible close (X) button — and where.
  3. If a modal exists, try (in order):
       a. Click the close button at the model-supplied coordinates.
       b. Press ``Escape`` (universal "dismiss menu/dialog" hint).
       c. Press ``Alt+F4`` as a last resort (closes the focused
          window outright).
  4. Re-verify; repeat up to ``max_attempts`` times.

Useful as a precondition before NavigateAgent / ClickAgent on real
desktops where Chrome's profile picker, Firefox's "refresh" prompt,
GNOME notifications, and the Ubuntu App Center can all stack up and
swallow keyboard focus.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.utils.imaging import (
    enhance_for_screen,
    numpy_to_base64_png,
    resize_for_mllm,
)

logger = logging.getLogger(__name__)


_MODAL_PROMPT = (
    "You are a JSON API. Look at the screen carefully. Is there a "
    "modal dialog, popup window, profile picker, software-updater "
    "toast, browser-recovery prompt, notification overlay, or any "
    "small window that is BLOCKING normal interaction with the main "
    "app?\n\n"
    "Examples of things to flag:\n"
    "  * 'Who's using Chrome?' profile picker\n"
    "  * 'Refresh Firefox' prompt\n"
    "  * 'Software updates available' Ubuntu nag\n"
    "  * GNOME notification banners with action buttons\n"
    "  * Save / discard / replace dialogs in any app\n"
    "  * Login prompts for sites overlaying the page\n\n"
    "Do NOT flag the OS top bar, the dock, the system tray, or the "
    "browser's own URL bar / tabs / page content as a modal. Those "
    "are persistent OS chrome.\n\n"
    "If a modal IS present, locate its CLOSE button (typically an "
    "'X' icon in the top-right corner of the modal, or a 'Cancel' "
    "or 'Dismiss' button). Provide its centre as image-fraction "
    "coordinates (0.0–1.0).\n\n"
    "Respond with ONLY a JSON object — no preamble, no markdown.\n\n"
    'Schema: {"has_modal": true|false, '
    '"description": "<short>", '
    '"close_x_pct": <float 0..1 or null>, '
    '"close_y_pct": <float 0..1 or null>}'
)


@dataclass
class DismissOutcome(Outcome):
    pass


class DismissModalsAgent(Agent):
    """Detect + dismiss any modal/popup blocking interaction."""

    name = "dismiss"

    async def run(
        self,
        *,
        max_attempts: int = 4,
        aggressive: bool = False,
    ) -> DismissOutcome:
        """Detect+dismiss modals.

        ``aggressive=True`` adds an unconditional pre-blast of
        ``Escape``×3 + ``Alt+F4`` + ``Escape``×2 BEFORE the
        detect-loop. Useful when chained as a precondition to
        NavigateAgent on a target where the visual verifier
        regularly fails to flag stuck windows like the Chrome
        profile picker, GNOME App Center, etc. Best-effort —
        may close the wrong window if the user has critical work
        in the foreground; the controller wraps this step as
        best_effort=True for that reason.
        """
        if self.ctx.vision_client is None:
            return DismissOutcome(
                success=False, reason="no vision client in context",
            )
        if self.ctx.capture is None:
            return DismissOutcome(
                success=False, reason="no capture in context",
            )

        if aggressive:
            await self._aggressive_blast()

        for attempt in range(1, max_attempts + 1):
            verdict = await self._detect_modal(attempt)
            if not verdict.get("has_modal"):
                if attempt == 1:
                    return DismissOutcome(
                        success=True, reason="no modal on screen",
                    )
                return DismissOutcome(
                    success=True,
                    reason=f"dismissed after {attempt - 1} step(s)",
                    data={"attempts": attempt - 1},
                )
            desc = verdict.get("description", "<unknown>")
            print(
                f"DismissModals[{attempt}/{max_attempts}]: "
                f"detected → {desc!r}"
            )
            cx = verdict.get("close_x_pct")
            cy = verdict.get("close_y_pct")
            # Strategy escalates: click → Escape → Alt+F4.
            method = await self._apply_dismiss(attempt, cx, cy)
            print(f"  ↺ tried {method}")
            await asyncio.sleep(0.7)

        # Final verification pass.
        verdict = await self._detect_modal(max_attempts + 1)
        if not verdict.get("has_modal"):
            return DismissOutcome(
                success=True,
                reason=f"dismissed after {max_attempts} step(s)",
                data={"attempts": max_attempts},
            )
        return DismissOutcome(
            success=False,
            reason=(
                f"modal persists after {max_attempts} attempts: "
                f"{verdict.get('description', '')}"
            ),
            data={"attempts": max_attempts},
        )

    # Apps whose top-bar name should NEVER be Alt+F4'd (catastrophic
    # for the user). Anything else gets the close hammer.
    _BROWSER_APP_NAMES = (
        "firefox", "chrome", "chromium", "brave", "edge",
        "safari", "opera", "vivaldi",
    )
    _PRESERVE_APP_NAMES = _BROWSER_APP_NAMES + ("nautilus",)

    async def _aggressive_blast(self) -> None:
        """Close non-browser focused windows until a browser surfaces.

        Strategy:
          1. Send ``Esc`` (close any open menu/dropdown).
          2. OCR the GNOME top bar to read the focused app name.
          3. If the name matches a browser → done.
          4. If unrecognised / non-browser → ``Alt+F4`` and re-OCR.
          5. Repeat up to N times.

        Surgical because we never blindly Alt+F4 a focused browser:
        the top-bar OCR distinguishes 'App Center' / 'Software' /
        'Settings' from 'Firefox' / 'Chrome' before pressing close.
        """
        kb = self.ctx.keyboard
        if kb is None:
            return
        print("DismissModals: aggressive blast (Esc + targeted Alt+F4 loop)")
        try:
            await kb.send_keystroke("Escape")
        except Exception:
            pass
        await asyncio.sleep(0.20)

        for cycle in range(1, 6):
            name = await self._focused_app_name()
            print(f"  top-bar OCR cycle {cycle}: {name!r}")
            if name and any(
                b in name for b in self._BROWSER_APP_NAMES
            ):
                print(f"  ✓ browser ({name}) is foreground; stopping blast")
                return
            if name and any(
                p in name for p in self._PRESERVE_APP_NAMES
            ):
                # Recognised non-browser-but-still-don't-close (e.g.
                # Files when the user might have it open intentionally).
                print(f"  ↺ preserving {name}; sending Esc instead")
                try:
                    await kb.send_keystroke("Escape")
                except Exception:
                    pass
                await asyncio.sleep(0.25)
                continue
            # Unrecognised foreground (App Center, Software, Settings,
            # toast, etc.). Close it.
            try:
                await kb.send_key_combo(["alt"], "F4")
            except Exception as e:
                logger.warning("blast Alt+F4 failed: %s", e)
            await asyncio.sleep(0.7)

        # Final dismiss-any-leftover-popup keystroke.
        try:
            await kb.send_keystroke("Escape")
        except Exception:
            pass
        await asyncio.sleep(0.20)

    async def _focused_app_name(self) -> str | None:
        """OCR the GNOME top bar's focused-app indicator."""
        if self.ctx.capture is None:
            return None
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception:
            return None
        h, w = frame.image.shape[:2]
        # The GNOME app indicator sits in the very top strip, just
        # right of the activities corner. Crop top ~4% × left 25%.
        y0 = 0
        y1 = max(20, int(h * 0.045))
        x0 = max(0, int(w * 0.03))
        x1 = int(w * 0.30)
        crop = frame.image[y0:y1, x0:x1]
        try:
            from terminaleyes.commander.ocr_finder import (
                _preprocess_for_ocr, have_ocr,
            )
            if not have_ocr():
                return None
            import pytesseract  # type: ignore
            # Top bar is white text on dark bg → invert.
            for invert in (True, False):
                pre = _preprocess_for_ocr(crop, scale=5, invert=invert)
                text = pytesseract.image_to_string(
                    pre, config="--psm 7"
                ).strip().lower()
                if text:
                    # Trim noise.
                    return " ".join(text.split())
        except Exception as e:
            logger.debug("top-bar OCR failed: %s", e)
        return None

    # ───────────────────── strategy ─────────────────────

    async def _apply_dismiss(
        self,
        attempt: int,
        cx_pct: float | None,
        cy_pct: float | None,
    ) -> str:
        """Run one dismissal attempt."""
        kb = self.ctx.keyboard
        mouse = self.ctx.mouse

        # Attempt 1: click the close button if the model gave us a
        # location; otherwise Escape.
        if attempt == 1:
            if (
                cx_pct is not None and cy_pct is not None
                and 0.0 <= cx_pct <= 1.0 and 0.0 <= cy_pct <= 1.0
                and mouse is not None
            ):
                await self._click_at(cx_pct, cy_pct)
                return f"click on close button at ({cx_pct:.2f},{cy_pct:.2f})"
            if kb is not None:
                await kb.send_keystroke("Escape")
                return "Escape"
            return "no-op (no kb/mouse)"

        # Attempt 2: Escape (gentler than Alt+F4).
        if attempt == 2 and kb is not None:
            await kb.send_keystroke("Escape")
            return "Escape"

        # Attempt 3+: Alt+F4 — destructive but effective for stuck
        # focused windows.
        if kb is not None:
            try:
                await kb.send_key_combo(["alt"], "F4")
                return "Alt+F4"
            except Exception as e:
                logger.warning("Alt+F4 failed: %s", e)
        return "no-op"

    async def _click_at(self, x_pct: float, y_pct: float) -> None:
        """Slam to corner, then move open-loop to (x_pct, y_pct)."""
        mouse = self.ctx.mouse
        if mouse is None:
            return
        # Slam top-left so we have a deterministic origin.
        for _ in range(160):
            try:
                await mouse.move(-20, -20)
            except Exception:
                pass
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.20)
        # Open-loop estimate. With macOS-style accel ≈ 1.6 HID per
        # pixel, ~3072 HID for full image width on 1920px screen.
        # Per-image-percent: 30 HID/% (rounded conservative).
        per_pct = 30
        target_x_hid = int(x_pct * 100 * per_pct)
        target_y_hid = int(y_pct * 100 * per_pct)
        rem_x, rem_y = target_x_hid, target_y_hid
        while rem_x != 0 or rem_y != 0:
            sx = max(-20, min(20, rem_x))
            sy = max(-20, min(20, rem_y))
            if sx != 0 or sy != 0:
                try:
                    await mouse.move(sx, sy)
                except Exception:
                    pass
            rem_x -= sx
            rem_y -= sy
            await asyncio.sleep(0.003)
        await asyncio.sleep(0.20)
        try:
            await mouse.click("left")
        except Exception as e:
            logger.warning("Click at modal close failed: %s", e)

    # ───────────────────── detection ─────────────────────

    async def _detect_modal(self, attempt: int) -> dict[str, Any]:
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:
            return {
                "has_modal": False,
                "description": f"(capture failed: {e})",
            }
        self.ctx.record_frame(
            frame.image, label=f"dismiss_check_{attempt:02d}",
        )
        b64 = numpy_to_base64_png(
            resize_for_mllm(
                enhance_for_screen(frame.image),
                max_dimension=1280, min_dimension=768,
            )
        )
        messages = [
            {"role": "system", "content": _MODAL_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": "Reply JSON only."},
                ],
            },
        ]
        for try_idx in range(2):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.ctx.vision_model,
                    max_tokens=600,
                    temperature=0.0,
                    messages=messages,
                )
                if try_idx == 0:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await self.ctx.vision_client.chat.completions.create(
                    **kwargs,
                )
                break
            except Exception as e:
                if try_idx == 0:
                    continue
                logger.warning("Modal-detect call failed: %s", e)
                return {
                    "has_modal": False,
                    "description": f"(model call failed: {e})",
                }

        raw = self._best_text_from_response(resp) or ""
        parsed = self._extract_json(raw) or {}
        return {
            "has_modal": bool(parsed.get("has_modal", False)),
            "description": str(parsed.get("description", ""))[:200],
            "close_x_pct": parsed.get("close_x_pct"),
            "close_y_pct": parsed.get("close_y_pct"),
        }

    # ───────────────────── helpers ─────────────────────

    def _best_text_from_response(self, resp) -> str:
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._best_text_from_response(resp) or ""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def _extract_json(self, raw: str) -> dict | None:
        if self.ctx.evaluator is not None:
            return self.ctx.evaluator._extract_json(raw)
        import json
        import re
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
