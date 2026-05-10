"""LoginAgent — wake screen, verify it's a login prompt, type password.

Compose-only agent. Builds on:

  - :class:`WakeAgent` — mouse jiggle / arrow / click
  - :class:`VerifyAgent` — visual yes/no oracle (polled until login
    screen is showing)
  - :class:`TypeAgent` — secret-mode text entry + optional Enter

Password sources, in priority order: explicit ``password=`` arg >
:class:`Vault` lookup by ``vault_name`` > file path > env var >
interactive ``getpass``. The agent never echoes or logs the value.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.type_text import TypeAgent
from terminaleyes.agents.verify import VerifyAgent
from terminaleyes.agents.wake import WakeAgent

logger = logging.getLogger(__name__)


LOGIN_QUESTION = (
    "Look at the screen. Decide whether it shows a LOGIN or PASSWORD "
    "entry screen, judging by visual structure ONLY — NOT by whether "
    "the literal word 'password' appears.\n\n"
    "Visual cues that count:\n"
    "  * a single prominent text input, often centred\n"
    "  * the input may show hidden-character dots or circles\n"
    "  * a user avatar / username\n"
    "  * a 'Sign in', 'Unlock', or 'Log in' button\n"
    "  * a system lock screen with a large clock\n"
    "  * dark or blurred background\n\n"
    "If the screen is a normal application, terminal, file manager, "
    "browser, etc., return false even if the word 'password' happens "
    "to be visible somewhere."
)


@dataclass
class LoginOutcome(Outcome):
    pass


def resolve_password(
    *,
    password: str | None = None,
    vault: object | None = None,
    vault_name: str | None = None,
    file_path: str | None = None,
    env_var: str | None = None,
) -> str:
    """Return the password from the chosen source.

    Priority: ``password`` (explicit) > ``vault_name`` (via ``vault``)
    > ``file_path`` > ``env_var`` > interactive ``getpass`` prompt.

    Never logs or prints the value.
    """
    if password is not None:
        return password
    if vault_name:
        if vault is None:
            from terminaleyes.agents.vault import Vault, get_passphrase
            passphrase = get_passphrase(prompt="Vault passphrase: ")
            vault = Vault(passphrase)
        try:
            return vault.get(vault_name)
        except KeyError as e:
            raise ValueError(
                f"Vault has no entry named {vault_name!r}. "
                "Use `terminaleyes vault add` first."
            ) from e
    if file_path:
        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Password file not found: {path}")
        pw = path.read_text()
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


class LoginAgent(Agent):
    """Wake → poll-verify login screen → type password → submit."""

    name = "login"

    async def run(
        self,
        *,
        password: str | None = None,
        vault_name: str | None = None,
        password_file: str | None = None,
        password_env: str | None = None,
        wake: bool = True,
        verify: bool = True,
        verify_attempts: int = 6,
        verify_interval: float = 1.0,
        click_input: bool = False,
        submit: bool = True,
    ) -> LoginOutcome:
        # 0. Resolve password BEFORE any I/O so a bad source fails fast.
        try:
            pw = resolve_password(
                password=password,
                vault=self.ctx.vault,
                vault_name=vault_name,
                file_path=password_file,
                env_var=password_env,
            )
        except Exception as e:
            return LoginOutcome(
                success=False, reason=f"password resolution failed: {e}",
            )
        if not pw:
            return LoginOutcome(
                success=False, reason="empty password — refusing to send",
            )

        # 1. Wake.
        if wake:
            wake_outcome = await WakeAgent(self.ctx).run()
            if not wake_outcome:
                logger.warning("Wake step reported failure: %s", wake_outcome.reason)

        # 2. Polled visual verification.
        if verify:
            ok, reason = await self._poll_for_login(
                attempts=verify_attempts, interval=verify_interval,
            )
            if not ok:
                return LoginOutcome(
                    success=False,
                    reason=(
                        f"visual verification: not a login screen "
                        f"after {verify_attempts} polls ({reason})"
                    ),
                    data={"verified": False},
                )

        # 3. Optional visual click on the input field. Most lock
        # screens auto-focus, so off by default.
        if click_input:
            await self._click_password_field()
            await asyncio.sleep(0.4)

        # 4. Type and submit.
        type_outcome = await TypeAgent(self.ctx).run(
            text=pw, secret=True, submit=submit,
        )
        # Drop the password reference promptly.
        pw = None
        if not type_outcome:
            return LoginOutcome(
                success=False,
                reason=f"type step failed: {type_outcome.reason}",
            )
        return LoginOutcome(
            success=True,
            reason="login submitted",
            data={"submit": submit, "verified": verify},
        )

    # ───────────────────── helpers ─────────────────────

    async def _poll_for_login(
        self, *, attempts: int, interval: float,
    ) -> tuple[bool, str]:
        verifier = VerifyAgent(self.ctx)
        last_reason = ""
        for i in range(1, attempts + 1):
            print(f"LoginAgent: verification attempt {i}/{attempts}...")
            v = await verifier.run(
                question=LOGIN_QUESTION, visual_only=True,
            )
            last_reason = v.reason
            if v:
                return True, v.reason
            if i == attempts:
                break
            # Nudge between polls. Alternate mouse + arrow.
            try:
                if i % 2 == 1 and self.ctx.mouse is not None:
                    await self.ctx.mouse.move(20, 0)
                    await asyncio.sleep(0.05)
                    await self.ctx.mouse.move(-20, 0)
                elif self.ctx.keyboard is not None:
                    await self.ctx.keyboard.send_keystroke("Down")
            except Exception as e:
                logger.warning("Nudge between polls failed: %s", e)
            await asyncio.sleep(interval)
        return False, last_reason

    async def _click_password_field(self) -> None:
        """Visually click the password input via the visual servo homer.

        Uses non-text-based descriptions so the flow works even when
        the screen has no literal "password" label.
        """
        if self.ctx.capture is None:
            return
        # Lazy import to avoid importing the homer at module load time.
        try:
            from terminaleyes.commander.visual_servo_homer import (
                VisualServoHomer,
            )
        except Exception as e:
            logger.warning("Could not import visual servo homer: %s", e)
            return
        # The homer needs a session-like object; build a minimal
        # adapter from the AgentContext.
        adapter = _SessionAdapter(self.ctx)
        homer = VisualServoHomer(session=adapter)
        for desc in (
            "the centred text input field on the lock screen",
            "the input box where the password is typed",
            "the highlighted text input in the middle of the screen",
        ):
            outcome = await homer.run(desc, button="left")
            if outcome.clicked:
                return
        logger.warning(
            "Could not visually click an input field; relying on "
            "auto-focus."
        )


class _SessionAdapter:
    """Just enough of the old InteractiveSession surface for the homer.

    The visual servo homer was built against ``InteractiveSession`` and
    pokes at private attributes. This adapter exposes the AgentContext
    pieces under the names the homer expects, so we can run it from
    the agent layer without changing the homer yet (that refactor is
    SearchAgent's job).
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._capture = ctx.capture
        self._client = ctx.vision_client
        self._model = ctx.vision_model
        self._evaluator = ctx.evaluator
        # Expose output_dir so the homer drops its per-step dump
        # alongside the rest of the session's artefacts.
        self.output_dir = getattr(ctx, "output_dir", None)
        # The homer reaches through `_executor._mouse` and calls
        # `_send_hid_moves` and `_showui_query`.
        self._executor = _ExecutorAdapter(ctx)

    async def _ensure_client(self) -> None:
        return None

    async def _send_hid_moves(self, dx: int, dy: int) -> None:
        # Match interactive.py's chunked, throttled send.
        from terminaleyes.commander.calibration import (
            MOVE_DELAY, MOVE_STEP_SIZE,
        )
        rem_x, rem_y = dx, dy
        while rem_x != 0 or rem_y != 0:
            sx = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_x))
            sy = max(-MOVE_STEP_SIZE, min(MOVE_STEP_SIZE, rem_y))
            if sx != 0 or sy != 0:
                await self._ctx.mouse.move(sx, sy)
            rem_x -= sx
            rem_y -= sy
            await asyncio.sleep(MOVE_DELAY)

    async def _showui_query(self, b64: str, prompt: str):
        # Use the existing InteractiveSession's helper if available.
        # Otherwise, return None — homer falls back to OCR/scene-map.
        if self._ctx.showui_query is None:
            return None
        return await self._ctx.showui_query(b64, prompt)


class _ExecutorAdapter:
    def __init__(self, ctx) -> None:
        self._mouse = ctx.mouse
        self._keyboard = ctx.keyboard
