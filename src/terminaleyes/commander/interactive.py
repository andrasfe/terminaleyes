"""Interactive visual commander — REPL for controlling a screen via webcam.

The user types commands or questions at a prompt. Each input triggers:
1. A fresh webcam capture
2. An MLLM vision call with the screenshot + user input
3. Either a text answer (for questions) or an action execution (for commands)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

import cv2
import numpy as np

from terminaleyes.capture.base import CaptureSource
from terminaleyes.commander.evaluator import ConditionEvaluator
from terminaleyes.commander.executor import ActionExecutor
from terminaleyes.commander.models import ActionSpec, ScreenLocation
from terminaleyes.utils.imaging import enhance_for_screen, numpy_to_base64_png, resize_for_mllm

logger = logging.getLogger(__name__)

INTERACTIVE_SYSTEM_PROMPT = """You see a computer screen through a webcam photo. You control the mouse and keyboard.

If the user asks a QUESTION, reply with plain text describing what you see.

If the user wants an ACTION, reply with ONLY a JSON object. Examples:

Click something: {"action_type": "mouse_click", "button": "left", "target_description": "the Review button", "location_x_pct": 0.85, "location_y_pct": 0.92, "reasoning": "Review button at bottom right"}

Type text: {"action_type": "text_input", "text": "hello world", "reasoning": "typing requested text"}

Press a key: {"action_type": "keystroke", "key": "Enter", "reasoning": "pressing enter"}

Key combo: {"action_type": "key_combo", "modifiers": ["ctrl"], "key": "c", "reasoning": "sending ctrl+c"}

Scroll: {"action_type": "mouse_scroll", "amount": -3, "reasoning": "scrolling down"}

Rules:
- action_type must be exactly one of: mouse_click, keystroke, key_combo, text_input, mouse_scroll
- For mouse_click, always include location_x_pct (0.0=left, 1.0=right) and location_y_pct (0.0=top, 1.0=bottom)
- Colors in the webcam photo are approximate — match elements by text and shape, not exact color
- The mouse cursor may cover parts of the screen"""


class InteractiveSession:
    """Interactive REPL for visual screen control."""

    def __init__(
        self,
        capture: CaptureSource,
        evaluator: ConditionEvaluator,
        executor: ActionExecutor,
        model: str,
        base_url: str | None = None,
        api_key: str = "not-needed",
        max_tokens: int = 2048,
        vision_model: str | None = None,
        vision_base_url: str | None = None,
        skip_screen_check: bool = False,
        force_calibration: bool = True,
        single_message: str | None = None,
    ) -> None:
        self._capture = capture
        self._evaluator = evaluator
        self._executor = executor
        self._model = model
        self._vision_model = vision_model or model
        self._base_url = base_url
        self._vision_base_url = vision_base_url
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = None
        self._vision_client = None
        self._skip_screen_check = skip_screen_check
        self._force_calibration = force_calibration
        self._single_message = single_message

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI

        kwargs: dict = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = AsyncOpenAI(**kwargs)

    async def _ensure_vision_client(self) -> None:
        if self._vision_client is not None:
            return
        from openai import AsyncOpenAI

        kwargs: dict = {"api_key": self._api_key}
        if self._vision_base_url:
            kwargs["base_url"] = self._vision_base_url
        elif self._base_url:
            kwargs["base_url"] = self._base_url
        self._vision_client = AsyncOpenAI(**kwargs)

    async def start(self) -> None:
        """Run setup checks then enter the interactive REPL."""
        async with self._capture:
            if not self._skip_screen_check:
                print("  Checking camera view...")
                frame = await self._capture.capture_frame()
                result = await self._evaluator.check_full_screen(frame.image)
                if result.full_screen_visible:
                    print("  Full screen visible — camera position OK\n")
                else:
                    edges = ", ".join(result.edges_cut_off) if result.edges_cut_off else "unknown"
                    print(f"  WARNING: Edges cut off: {edges}")
                    if result.suggestion:
                        print(f"  Suggestion: {result.suggestion}")
                    print(f"  Continuing anyway — adjust camera if needed.\n")

            # Calibration — force by default, skip with --skip-calibration
            if self._force_calibration:
                from terminaleyes.commander.calibration import MouseCalibrator, CalibrationResult
                import os
                # Delete old calibration to force re-run
                cal_path = CalibrationResult.load()
                if cal_path is not None:
                    from terminaleyes.commander.calibration import CALIBRATION_FILE
                    os.remove(CALIBRATION_FILE)
                calibrator = MouseCalibrator(mouse=self._executor._mouse)
                await calibrator.calibrate_or_load()

            # Single message mode or REPL
            if self._single_message:
                try:
                    handled = await self._try_fast_action(self._single_message)
                    if not handled:
                        await self._handle_input(self._single_message)
                except Exception as e:
                    logger.error("Error: %s", e)
                    print(f"  Error: {e}")
            else:
                print("Ready. Type commands or questions.")
                print("  help     — show available commands")
                print("  quit     — exit")
                print()
                await self._repl()

    async def _repl(self) -> None:
        """Main read-eval-print loop."""
        while True:
            try:
                user_input = await self._read_input("> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            user_input = user_input.strip()
            if not user_input:
                continue

            # Local commands
            if user_input.lower() in ("quit", "exit"):
                print("Exiting.")
                return

            if user_input.lower() == "help":
                self._print_help()
                continue

            if user_input.lower() == "calibrate":
                from terminaleyes.commander.calibration import MouseCalibrator
                calibrator = MouseCalibrator(mouse=self._executor._mouse)
                self._executor._calibration = await calibrator.force_calibrate()
                continue

            if user_input.lower() == "screenshot":
                frame = await self._capture.capture_frame()
                ts = datetime.now().strftime("%H%M%S")
                path = f"screenshot_{ts}.png"
                cv2.imwrite(path, frame.image)
                print(f"  Saved to {path}")
                continue

            # Try fast local parsing first — skip gemma for simple actions
            handled = await self._try_fast_action(user_input)
            if not handled:
                # Fall through to gemma for questions / complex commands
                try:
                    await self._handle_input(user_input)
                except Exception as e:
                    logger.error("Error handling input: %s", e)
                    print(f"  Error: {e}")

            print()

    async def _try_fast_action(self, user_input: str) -> bool:
        """Parse simple action commands locally — no gemma needed.

        Returns True if handled, False to fall through to gemma.
        Patterns: click X, type X, press X, scroll up/down
        """
        lower = user_input.lower().strip()

        # Click commands → go straight to ShowUI
        if lower.startswith("click on "):
            target = user_input[9:].strip()
            await self._homing_click("left", target)
            return True

        if lower.startswith("click "):
            target = user_input[6:].strip()
            await self._homing_click("left", target)
            return True

        if lower.startswith("right click ") or lower.startswith("right-click "):
            target = user_input.split(" ", 2)[-1].strip()
            await self._homing_click("right", target)
            return True

        # Type commands
        if lower.startswith("type "):
            text = user_input[5:].strip().strip('"').strip("'")
            print(f"  Typing: {text}")
            await self._executor._keyboard.send_text(text)
            print("  Done.")
            return True

        # Press key commands
        if lower.startswith("press "):
            key_str = user_input[6:].strip()
            # Handle combos like "Ctrl+C"
            if "+" in key_str:
                parts = [p.strip() for p in key_str.split("+")]
                modifiers = [p.lower() for p in parts[:-1]]
                key = parts[-1]
                print(f"  Pressing {key_str}")
                await self._executor._keyboard.send_key_combo(modifiers, key)
            else:
                print(f"  Pressing {key_str}")
                await self._executor._keyboard.send_keystroke(key_str)
            print("  Done.")
            return True

        # Scroll commands
        if lower in ("scroll down", "scroll up"):
            amount = -3 if "down" in lower else 3
            print(f"  Scrolling {'down' if amount < 0 else 'up'}")
            await self._executor._mouse.scroll(amount)
            print("  Done.")
            return True

        return False  # not a simple action — use gemma

    async def _handle_input(self, user_input: str) -> None:
        """Capture frame, send to MLLM with user input, execute or respond."""
        await self._ensure_client()

        # Capture fresh frame
        frame = await self._capture.capture_frame()
        resized = resize_for_mllm(enhance_for_screen(frame.image), max_dimension=1280, min_dimension=768)
        b64_image = numpy_to_base64_png(resized)

        messages = [
            {"role": "system", "content": INTERACTIVE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": user_input,
                    },
                ],
            },
        ]

        print("  Thinking...")
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
        )

        raw_text = self._evaluator._best_text_from_response(response)
        if not raw_text:
            print("  (empty response from model)")
            return

        logger.debug("Interactive response: %s", raw_text[:300])

        # Try to parse as action JSON
        data = self._evaluator._extract_json(raw_text)

        if data is not None and "action_type" in data:
            await self._execute_action(data, user_input)
        else:
            # It's a text answer — print it
            # Strip markdown code blocks if the model wrapped it
            text = raw_text.strip()
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match and "action_type" not in text:
                text = match.group(1).strip()
            # Just print the plain text answer
            print()
            print(text)

    @staticmethod
    def _normalize_action_type(data: dict) -> str:
        """Normalize action_type — handle models that output garbage like 'mouse_click|keystroke'."""
        raw = data.get("action_type", "")

        # If the model copied the pipe-separated options, pick the best one
        if "|" in raw:
            # Infer from other fields
            if data.get("location_x_pct") is not None or data.get("button"):
                return "mouse_click"
            if data.get("text"):
                return "text_input"
            if data.get("modifiers"):
                return "key_combo"
            if data.get("key"):
                return "keystroke"
            if data.get("amount") is not None:
                return "mouse_scroll"
            # Default to first option
            return raw.split("|")[0].strip()

        # Handle common misspellings/variants
        raw_lower = raw.lower().strip()
        for valid in ("mouse_click", "keystroke", "key_combo", "text_input", "mouse_scroll"):
            if valid in raw_lower:
                return valid

        return raw

    async def _execute_action(self, data: dict, user_input: str = "") -> None:
        """Execute a parsed action from the MLLM response."""
        action_type = self._normalize_action_type(data)
        reasoning = data.get("reasoning", "")

        if reasoning:
            print(f"  {reasoning}")

        if action_type == "mouse_click":
            button = "left"
            raw_button = (data.get("button") or "").lower()
            if raw_button in ("right", "middle"):
                button = raw_button
            target_desc = user_input if user_input else data.get("target_description", "the target element")
            # Pass gemma's coordinates as fallback in case ShowUI can't find the target
            gemma_x = data.get("location_x_pct")
            gemma_y = data.get("location_y_pct")
            await self._homing_click(button, target_desc, gemma_fallback=(gemma_x, gemma_y))

        elif action_type == "keystroke":
            key = data.get("key", "Enter")
            print(f"  Pressing {key}...")
            action = ActionSpec(action_type="keystroke", key=key)
            await self._executor.execute(action)
            print("  Done.")

        elif action_type == "key_combo":
            key = data.get("key", "")
            modifiers = data.get("modifiers") or []
            combo = "+".join(modifiers + [key])
            print(f"  Pressing {combo}...")
            action = ActionSpec(
                action_type="key_combo", modifiers=modifiers, key=key
            )
            await self._executor.execute(action)
            print("  Done.")

        elif action_type == "text_input":
            text = data.get("text", "")
            print(f"  Typing: {text[:60]}{'...' if len(text) > 60 else ''}")
            action = ActionSpec(action_type="text_input", text=text)
            await self._executor.execute(action)
            print("  Done.")

        elif action_type == "mouse_scroll":
            amount = int(data.get("amount", -3))
            direction = "up" if amount > 0 else "down"
            print(f"  Scrolling {direction} ({amount})...")
            from terminaleyes.mouse.base import MouseOutput
            await self._executor._mouse.scroll(amount)
            print("  Done.")

        else:
            print(f"  Unknown action: {action_type}")

    async def _homing_click(
        self, button: str, target_desc: str,
        gemma_fallback: tuple[float | None, float | None] = (None, None),
    ) -> None:
        """Visual homing: ShowUI for target, cached calibration for movement.

        1. ShowUI finds the target element (fast, ~0.1s)
        2. Load calibration (interactive on first run, cached after)
        3. Slam to corner (known 0,0), move calibrated distance to target
        4. Click and save proof screenshot
        """
        from terminaleyes.commander.calibration import CalibrationResult, MouseCalibrator

        print(f"  Homing to: {target_desc}")

        # Step 1: Find target with ShowUI — query 3 times and check consistency
        b64 = await self._capture_b64()
        showui_prompt = target_desc
        if not showui_prompt.lower().startswith("click"):
            showui_prompt = f"Click on {showui_prompt}"

        # Detect screen boundaries in image space (screen != full photo)
        screen_bounds = await self._detect_screen_bounds(b64)
        sx0, sy0, sx1, sy1 = screen_bounds
        print(f"  Screen bounds in image: ({sx0:.0%},{sy0:.0%})→({sx1:.0%},{sy1:.0%})")

        # Try ShowUI first (fast), fall back to gemma (accurate)
        pos = await self._showui_locate(b64, showui_prompt)
        if pos is not None:
            img_x, img_y = pos
            # Map from image space to screen space
            tx = (img_x - sx0) / (sx1 - sx0) if sx1 > sx0 else img_x
            ty = (img_y - sy0) / (sy1 - sy0) if sy1 > sy0 else img_y
            tx = max(0.0, min(1.0, tx))
            ty = max(0.0, min(1.0, ty))
            print(f"  ShowUI: image ({img_x:.1%},{img_y:.1%}) → screen ({tx:.1%},{ty:.1%})")
        else:
            gx, gy = gemma_fallback
            if gx is not None and gy is not None:
                # Gemma coords are also in image space
                tx = (float(gx) - sx0) / (sx1 - sx0)
                ty = (float(gy) - sy0) / (sy1 - sy0)
                tx = max(0.0, min(1.0, tx))
                ty = max(0.0, min(1.0, ty))
                print(f"  Gemma: screen ({tx:.1%}, {ty:.1%})")
            else:
                print(f"  Asking gemma...")
                gemma_pos = await self._ask_gemma_location(b64, target_desc)
                if gemma_pos is not None:
                    tx = (gemma_pos[0] - sx0) / (sx1 - sx0)
                    ty = (gemma_pos[1] - sy0) / (sy1 - sy0)
                    tx = max(0.0, min(1.0, tx))
                    ty = max(0.0, min(1.0, ty))
                    print(f"  Gemma: screen ({tx:.1%}, {ty:.1%})")
                else:
                    print(f"  Cannot find '{target_desc}'.")
                    return

        # Step 2: Load or run calibration (cached to disk after first run)
        cal = CalibrationResult.load()
        if cal is None:
            print(f"  No calibration found — running interactive calibration...")
            calibrator = MouseCalibrator(mouse=self._executor._mouse)
            cal = await calibrator.calibrate_or_load()

        # Step 3: Slam to corner, move to target
        print(f"  Slamming to corner...")
        for _ in range(200):
            await self._executor._mouse.move(-20, -20)
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.3)

        dx_hid, dy_hid = cal.hid_units_for_pct(tx, ty)
        print(f"  Moving to target: ({dx_hid}, {dy_hid}) HID  [cal: {cal.hid_units_per_full_x:.0f}x{cal.hid_units_per_full_y:.0f}]")
        await self._send_hid_moves(dx_hid, dy_hid)
        await asyncio.sleep(0.3)

        # Step 4: Verify and correct loop.
        # After each move, check if target is still visible.
        # If yes → cursor missed → compute correction from where target still is.
        # If target vanished → cursor covering it → click.
        import cv2 as _cv2
        cursor_x, cursor_y = tx, ty  # where we think cursor is

        for attempt in range(5):
            await asyncio.sleep(0.3)
            b64_after = await self._capture_b64()
            target_after = await self._showui_locate(b64_after, showui_prompt)

            frame = await self._capture.capture_frame()
            proof_path = "/tmp/cursor_on_target.png"
            _cv2.imwrite(proof_path, frame.image)

            if target_after is not None:
                # Map from image space to screen space
                img_ax, img_ay = target_after
                target_after = (
                    max(0.0, min(1.0, (img_ax - sx0) / (sx1 - sx0) if sx1 > sx0 else img_ax)),
                    max(0.0, min(1.0, (img_ay - sy0) / (sy1 - sy0) if sy1 > sy0 else img_ay)),
                )

            if target_after is None:
                # Target vanished — cursor likely covering it
                print(f"  [{attempt+1}] Target vanished — clicking {button}.")
                await self._executor._mouse.click(button)
                print(f"  Screenshot: {proof_path}")
                return

            ax, ay = target_after

            # Target is still visible → cursor didn't land on it.
            # The difference between where we aimed and where the target
            # still appears tells us how far off the cursor is.
            dx = ax - cursor_x
            dy = ay - cursor_y

            if abs(dx) < 0.02 and abs(dy) < 0.02:
                # Target barely moved — cursor is very close but not covering it.
                # Nudge toward the target center and click.
                print(f"  [{attempt+1}] Very close — nudging and clicking {button}.")
                dx_hid, dy_hid = cal.hid_units_for_pct(dx, dy)
                await self._send_hid_moves(dx_hid, dy_hid)
                await asyncio.sleep(0.2)
                await self._executor._mouse.click(button)
                frame = await self._capture.capture_frame()
                _cv2.imwrite(proof_path, frame.image)
                print(f"  Screenshot: {proof_path}")
                return

            # Cursor missed — correct toward where target still is
            print(f"  [{attempt+1}] Target still at ({ax:.0%},{ay:.0%}), cursor at ~({cursor_x:.0%},{cursor_y:.0%}). Correcting ({dx:+.1%},{dy:+.1%})")
            dx_hid, dy_hid = cal.hid_units_for_pct(dx, dy)
            await self._send_hid_moves(dx_hid, dy_hid)
            cursor_x += dx
            cursor_y += dy

        print(f"  Could not reach target after corrections — NOT clicking.")
        print(f"  Screenshot: {proof_path}")

    @staticmethod
    def _find_cursor_by_diff(
        frame_a: np.ndarray, frame_b: np.ndarray
    ) -> tuple[float, float] | None:
        """Detect cursor position by differencing two grayscale frames.

        Looks for a small, compact changed region (cursor-sized).
        Ignores large changes (UI updates, hover effects) and tiny noise.
        """
        import cv2 as _cv2

        diff = _cv2.absdiff(frame_a, frame_b)
        _, thresh = _cv2.threshold(diff, 25, 255, _cv2.THRESH_BINARY)

        # Dilate to connect nearby changed pixels (cursor is small)
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (5, 5))
        thresh = _cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = _cv2.findContours(thresh, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        h, w = frame_a.shape[:2]
        img_area = h * w

        # Filter: cursor is small (< 1% of image) but not noise (> 10px)
        # If multiple candidates, pick the most compact (lowest area/perimeter ratio)
        candidates = []
        for c in contours:
            area = _cv2.contourArea(c)
            if area < 10 or area > img_area * 0.05:  # skip noise and large UI changes
                continue
            peri = _cv2.arcLength(c, True)
            compactness = (4 * 3.14159 * area) / (peri * peri + 1e-6) if peri > 0 else 0
            M = _cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidates.append((compactness, area, cx, cy))

        if not candidates:
            return None

        # Pick the most compact small change (most cursor-like)
        candidates.sort(key=lambda c: (-c[0], c[1]))  # most compact first
        _, _, cx, cy = candidates[0]

        return (cx / w, cy / h)

    async def _capture_gray(self) -> np.ndarray:
        """Capture a grayscale frame."""
        import cv2 as _cv2
        frame = await self._capture.capture_frame()
        return _cv2.cvtColor(frame.image, _cv2.COLOR_BGR2GRAY)

    async def _detect_screen_bounds(self, b64_image: str) -> tuple[float, float, float, float]:
        """Detect screen edges in image space using ShowUI.

        Returns (x0, y0, x1, y1) — the screen's top-left and bottom-right
        as fractions of the webcam image. Cached after first call.
        """
        if hasattr(self, '_screen_bounds_cache'):
            return self._screen_bounds_cache

        tl = await self._showui_query(b64_image, "Click on the top left corner of the monitor screen")
        br = await self._showui_query(b64_image, "Click on the bottom right corner of the monitor screen")

        x0 = tl[0] if tl else 0.05
        y0 = tl[1] if tl else 0.05
        x1 = br[0] if br else 0.95
        y1 = br[1] if br else 0.95

        # Sanity check
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0.05, 0.05, 0.95, 0.95

        self._screen_bounds_cache = (x0, y0, x1, y1)
        return self._screen_bounds_cache

    async def _ask_gemma_location(self, b64_image: str, target_desc: str) -> tuple[float, float] | None:
        """Ask gemma for element coordinates. Slower but more accurate than ShowUI."""
        await self._ensure_client()

        prompt = f"""Find this element on the screen: {target_desc}

Reply with ONLY a JSON object:
{{"x": 0.0 to 1.0, "y": 0.0 to 1.0}}

Where x=0 is the left edge, x=1 is the right edge, y=0 is the top, y=1 is the bottom.
If the element is not visible, reply: {{"not_found": true}}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=512,
                messages=messages,
            )
            raw = self._evaluator._best_text_from_response(response)
            if not raw:
                return None

            data = self._evaluator._extract_json(raw)
            if data is None or data.get("not_found"):
                return None

            x = data.get("x") or data.get("location_x_pct")
            y = data.get("y") or data.get("location_y_pct")
            if x is not None and y is not None:
                return (float(x), float(y))
            return None

        except Exception as e:
            logger.error("Gemma location query failed: %s", e)
            return None

    async def _capture_b64(self) -> str:
        """Capture, enhance, resize, encode — one-liner helper."""
        frame = await self._capture.capture_frame()
        enhanced = enhance_for_screen(frame.image)
        resized = resize_for_mllm(enhanced, max_dimension=1280, min_dimension=768)
        return numpy_to_base64_png(resized)

    async def _showui_locate(self, b64_image: str, prompt: str) -> tuple[float, float] | None:
        """Ask ShowUI for coordinates. Returns (x_pct, y_pct) or None.

        Tries the prompt as-is first. If no coordinates found, tries
        alternative phrasings. Handles both ShowUI output formats:
        - Integer: "(838, 712)" in [0,1000] space
        - Float: "[0.5, 0.17]" in [0,1] space
        """
        # Try the original prompt and progressively shorter/simpler variants
        prompts = [prompt]
        core = prompt.replace("Click on ", "").replace("click on ", "").replace("Click ", "").replace("click ", "")
        if core != prompt:
            # Strip "the" and "button" for shorter prompts (ShowUI prefers short)
            bare = core.replace("the ", "").replace(" button", "").replace(" icon", "").strip()
            prompts.append(f"Click on {bare}")
            prompts.append(f"Click on the button that says {bare}")
            prompts.append(f"Click on the tab that says {bare}")
            prompts.append(f"Click on {core}")

        for p in prompts:
            result = await self._showui_query(b64_image, p)
            if result is not None:
                logger.debug("ShowUI found target with prompt: %s → %s", p, result)
                return result
            logger.debug("ShowUI miss: %s", p)

        logger.warning("ShowUI could not find target with any prompt: %s", prompts)
        return None

    async def _showui_query(self, b64_image: str, prompt: str) -> tuple[float, float] | None:
        """Single ShowUI query. Returns (x_pct, y_pct) or None."""
        await self._ensure_vision_client()

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            response = await self._vision_client.chat.completions.create(
                model=self._vision_model,
                max_tokens=50,
                messages=messages,
                temperature=0.1,
            )
            content = response.choices[0].message.content or ""
            return self._parse_showui_coords(content)

        except Exception as e:
            logger.error("ShowUI query failed: %s", e)
            return None

    @staticmethod
    def _parse_showui_coords(content: str) -> tuple[float, float] | None:
        """Parse ShowUI coordinates from response text.

        Handles two formats:
        - Integer in [0,1000]: "element(838, 712)" → (0.838, 0.712)
        - Float in [0,1]: "[0.5, 0.17]" → (0.5, 0.17)
        """
        import re

        # Try integer format: (838, 712)
        m = re.search(r"\((\d+),\s*(\d+)\)", content)
        if m:
            x = int(m.group(1)) / 1000.0
            y = int(m.group(2)) / 1000.0
            return (min(1.0, x), min(1.0, y))

        # Try float format: [0.5, 0.17]
        m = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", content)
        if m:
            x = float(m.group(1))
            y = float(m.group(2))
            # If values > 1, they might be in [0,1000] space
            if x > 1.0 or y > 1.0:
                x /= 1000.0
                y /= 1000.0
            return (min(1.0, x), min(1.0, y))

        return None

    async def _send_hid_moves(self, dx_hid: int, dy_hid: int) -> None:
        """Send HID moves 1 unit at a time."""
        from terminaleyes.commander.calibration import MOVE_DELAY, MOVE_STEP_SIZE
        rem_x, rem_y = dx_hid, dy_hid
        while rem_x != 0 or rem_y != 0:
            sx = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_x))
            sy = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_y))
            if sx != 0 or sy != 0:
                await self._executor._mouse.move(sx, sy)
            rem_x -= sx
            rem_y -= sy
            await asyncio.sleep(MOVE_DELAY)

    @staticmethod
    def _print_help() -> None:
        print("""
  Interactive Visual Commander — Commands:

  Questions (answered by the vision model):
    what do you see?          — describe the screen
    is there a dialog box?    — check for specific elements
    what text is at the top?  — read specific areas

  Actions (executed on the target machine):
    click on the Run button   — find and click an element
    click at the top right    — click a screen region
    type hello world          — type text
    press Enter               — send a keystroke
    press Ctrl+C              — send a key combo
    scroll down               — scroll the mouse wheel

  Local commands:
    screenshot                — save current webcam frame to file
    calibrate                 — re-run mouse calibration
    help                      — show this help
    quit / exit               — exit the session
""")

    @staticmethod
    async def _read_input(prompt: str) -> str:
        """Read a line from stdin without blocking the async event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))
