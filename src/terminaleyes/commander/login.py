"""Remote-login flow.

Wakes the screen, optionally clicks the password field, then types
the password and submits with Enter. The password is sourced from a
file path or environment variable — never a positional CLI arg
(arguments are visible in ``ps`` to other users on the dev machine).

Default flow (Ubuntu/GNOME GDM lock screen):

    1. Wake — mouse jiggle + Down arrow + a left-click. This dismisses
       the clock overlay and brings up the password input.
    2. Click password field (optional, ``--click-input``). Most lock
       screens auto-focus the password field, so this is off by default.
    3. Type password. The text is sent as raw HID over BT/USB, with
       ``secret=True`` so it is not echoed in any local log.
    4. Press Enter to submit.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_password(
    file_path: str | None = None,
    env_var: str | None = None,
) -> str:
    """Return the password from the chosen source.

    Priority: ``file_path`` > ``env_var`` > interactive ``getpass`` prompt.
    Never logs or prints the value.
    """
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Password file not found: {path}")
        pw = path.read_text()
        # Strip a single trailing newline (most editors append one).
        if pw.endswith("\r\n"):
            pw = pw[:-2]
        elif pw.endswith("\n"):
            pw = pw[:-1]
        return pw
    if env_var:
        val = os.environ.get(env_var)
        if val is None:
            raise ValueError(
                f"Environment variable {env_var!r} is not set"
            )
        return val
    return getpass.getpass("Remote password: ")


class LoginFlow:
    """Wake-screen + type-password sequence."""

    def __init__(self, *, mouse, keyboard, session=None) -> None:
        self._mouse = mouse
        self._keyboard = keyboard
        # Optional: an InteractiveSession so we can use the visual
        # homer to click the password field. Only required when
        # ``click_input=True`` is passed to ``login``.
        self._session = session

    async def login(
        self,
        password: str,
        *,
        wake: bool = True,
        click_input: bool = False,
        submit: bool = True,
        verify: bool = True,
        verify_attempts: int = 6,
        verify_interval: float = 1.0,
    ) -> bool:
        """Perform the login sequence end-to-end.

        Returns ``True`` if the password was sent, ``False`` if the
        flow aborted (e.g. visual verification said the screen does
        not look like a password prompt after all attempts).
        """
        if wake:
            await self._wake()
        if verify:
            looks_ok = await self._poll_for_login_screen(
                attempts=verify_attempts,
                interval=verify_interval,
            )
            if not looks_ok:
                logger.warning(
                    "Visual verification: %d attempts did not show a "
                    "login/password prompt. Aborting without typing.",
                    verify_attempts,
                )
                print(
                    f"Visual check failed after {verify_attempts} polls "
                    "— password NOT typed."
                )
                return False
        if click_input:
            await self._click_password_field()
            await asyncio.sleep(0.4)
        # Type password — secret=True suppresses any local logging.
        await self._keyboard.send_text(password, secret=True)
        await asyncio.sleep(0.3)
        if submit:
            await self._keyboard.send_keystroke("Enter")
        logger.info(
            "Login submitted (password length=%d, wake=%s, "
            "click_input=%s, submit=%s, verify=%s)",
            len(password), wake, click_input, submit, verify,
        )
        return True

    async def _wake(self) -> None:
        """Try to bring a sleeping/locked screen to the password prompt."""
        # 1. Mouse jiggle — wakes monitors and registers activity.
        for _ in range(4):
            try:
                await self._mouse.move(20, 0)
                await asyncio.sleep(0.04)
                await self._mouse.move(-20, 0)
                await asyncio.sleep(0.04)
            except Exception as e:
                logger.warning("Wake jiggle failed: %s", e)
                break
        # 2. Press an arrow key — GDM dismisses the clock overlay on
        # any keystroke. Down is safe (won't open menus or scroll a
        # page if focus is elsewhere).
        try:
            await self._keyboard.send_keystroke("Down")
        except Exception as e:
            logger.warning("Wake keystroke failed: %s", e)
        await asyncio.sleep(0.4)
        # 3. Left click — covers the case where the lock screen wants
        # an explicit click before showing the password prompt.
        try:
            await self._mouse.click("left")
        except Exception as e:
            logger.warning("Wake click failed: %s", e)
        await asyncio.sleep(0.6)

    async def _click_password_field(self) -> None:
        """Visual click on the password/login input via the homer.

        Uses non-text-based grounding prompts so the flow works even
        when the screen has no literal label like "password" — visual
        cues (centred input, dots, avatar) drive ShowUI.
        """
        if self._session is None:
            logger.warning(
                "click_input requested but no session available; "
                "skipping (relying on auto-focus)"
            )
            return
        from terminaleyes.commander.visual_servo_homer import (
            VisualServoHomer,
        )
        homer = VisualServoHomer(session=self._session)
        # Try visual descriptions first (work on label-less screens),
        # then fall through to text-aware ones.
        for desc in (
            "the centred text input field on the lock screen",
            "the input box where the password is typed",
            "the highlighted text input in the middle of the screen",
            "the 'password' input field",
        ):
            outcome = await homer.run(desc, button="left")
            if outcome.clicked:
                return
        logger.warning(
            "Could not visually click an input field; relying on "
            "auto-focus."
        )

    async def _poll_for_login_screen(
        self, attempts: int, interval: float,
    ) -> bool:
        """Poll up to ``attempts`` times, nudging the screen between
        checks, until the vision model says the current frame looks
        like a login/password prompt.

        Returns ``True`` on the first positive verdict; ``False`` if
        all attempts fail.
        """
        for i in range(1, attempts + 1):
            print(
                f"Login-screen verification attempt {i}/{attempts}..."
            )
            looks_ok = await self._looks_like_login_screen()
            if looks_ok:
                return True
            if i == attempts:
                break
            # Nudge between attempts: small mouse move + an arrow key
            # to push the lock screen / monitor wake along. Alternate
            # between mouse and key so we cover both wake mechanisms.
            try:
                if i % 2 == 1:
                    await self._mouse.move(20, 0)
                    await asyncio.sleep(0.05)
                    await self._mouse.move(-20, 0)
                else:
                    await self._keyboard.send_keystroke("Down")
            except Exception as e:
                logger.warning("Nudge between polls failed: %s", e)
            await asyncio.sleep(interval)
        return False

    async def _looks_like_login_screen(self) -> bool:
        """Return True if the current frame looks like a login/password
        prompt — by visual structure alone, not by the literal text
        'password' appearing anywhere.

        Visual cues:
          * a single prominent text input, usually centred
          * input may show hidden-character dots/circles
          * a user avatar, a clock, or a "sign in / unlock" button
          * dark/blurred background typical of lock screens
        """
        if self._session is None:
            logger.info(
                "Verification skipped — no session wired. Trusting "
                "the wake sequence."
            )
            return True

        # Capture a frame and ask the multimodal model.
        try:
            frame = await self._session._capture.capture_frame()
        except Exception as e:
            logger.warning("Verification capture failed: %s", e)
            return True  # don't block on infrastructure failure

        from terminaleyes.utils.imaging import (
            enhance_for_screen,
            numpy_to_base64_png,
            resize_for_mllm,
        )
        b64 = numpy_to_base64_png(
            resize_for_mllm(
                enhance_for_screen(frame.image),
                max_dimension=1280, min_dimension=768,
            )
        )
        await self._session._ensure_client()
        prompt = (
            "You are a JSON API. Look at the screen. Decide whether "
            "it looks like a LOGIN / PASSWORD entry screen, judging "
            "ONLY by visual structure — NOT by whether the literal "
            "word 'password' appears.\n\n"
            "Visual cues that count:\n"
            "  * a single prominent text input, often centred\n"
            "  * the input may show hidden-character dots or circles\n"
            "  * a user avatar / username\n"
            "  * a 'Sign in', 'Unlock', or 'Log in' button\n"
            "  * a system lock screen with a large clock\n"
            "  * dark/blurred background\n\n"
            "If the screen is a normal application, terminal, file "
            "manager, browser, etc., return false even if the word "
            "'password' happens to be visible somewhere.\n\n"
            "Respond with ONLY a JSON object — no preamble, no "
            "markdown.\n\n"
            'Schema: {"is_login_screen": true|false, '
            '"reason": "<one short sentence>"}'
        )
        messages = [
            {"role": "system", "content": prompt},
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
                    {
                        "type": "text",
                        "text": "Login screen? Reply JSON only.",
                    },
                ],
            },
        ]
        try:
            resp = await self._session._client.chat.completions.create(
                model=self._session._model,
                max_tokens=200,
                temperature=0.0,
                messages=messages,
            )
            raw = (
                self._session._evaluator._best_text_from_response(resp)
                or ""
            )
            data = self._session._evaluator._extract_json(raw) or {}
            verdict = bool(data.get("is_login_screen", False))
            reason = str(data.get("reason", ""))[:200]
            logger.info(
                "Login-screen check: is_login_screen=%s reason=%r",
                verdict, reason,
            )
            print(
                f"Visual check: is_login_screen={verdict} "
                f"({reason or 'no reason'})"
            )
            return verdict
        except Exception as e:
            logger.warning(
                "Login-screen check failed: %s — proceeding anyway", e,
            )
            return True
