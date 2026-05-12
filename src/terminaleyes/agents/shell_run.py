"""ShellRunAgent — run a shell command and capture its stdout.

Tier-3 workflow primitive. Wraps a single shell command in unique
ASCII markers so the actual stdout can be sliced out of the OCR'd
screen regardless of whatever else is on it (journalctl tails,
status lines, previous output, etc.).

Why this exists
---------------
The naive verification pattern was "ScriptAgent runs `ls foo`, then
ask the verifier whether `foo` is there." Two failure modes hit it
constantly:

  1. The foreground terminal isn't a shell — it's running journalctl
     or another long-lived foreground process — and silently eats
     the keystrokes. No command ever runs, but the verifier sees
     plausible text on screen and hallucinates success.

  2. Output from prior commands or background log spam dominates the
     OCR result. The verifier's natural-language answer drifts off
     the actual stdout we care about.

ShellRunAgent neutralises both:

  * **Pre-flight** — sends Ctrl+C (twice, settle-spaced) to break out
    of any foreground process, then `clear` so the marker-bracketed
    output starts from a clean screen.
  * **Markers** — wraps the user's command as
    ``printf 'TEBEGIN<id>\\n'; <command>; printf 'TEEND<id>\\n'``
    with a fresh random id per run, so we can locate the exact
    output region in the OCR text. Using ``printf`` keeps the marker
    on its own line; ``echo`` is fine too but printf survives shells
    where ``echo`` is aliased.
  * **Poll** — captures frames + OCRs until both markers are visible
    or the timeout expires; returns the substring between them as
    ``outcome.data["stdout"]`` (verbatim, line-preserving).

This is the right agent to use whenever the next planner step needs
to *react* to a command's output. The plain :class:`ScriptAgent`
remains the right tool for fire-and-forget multi-line scripts where
verification happens visually.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.ocr import OcrAgent

logger = logging.getLogger(__name__)


_BEGIN_PREFIX = "TEBEGIN"
_END_PREFIX = "TEEND"


@dataclass
class ShellRunOutcome(Outcome):
    """``data['stdout']`` carries the text between the BEGIN/END
    markers (verbatim, with line breaks). ``data['marker_id']``
    is the random run-id for debug correlation."""


def _new_marker_id() -> str:
    # 8 uppercase letters (no digits). OCR models confuse digits
    # with letters (0/O, 8/B, 1/I/l, 5/S) far more often than they
    # mangle pure letters, so a hex id was unreliable in practice.
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # drop O/I for clarity
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _extract_between_markers(
    text: str, begin_token: str, end_token: str,
) -> str | None:
    """Return the substring strictly between the two markers (excl.
    the markers themselves and the surrounding newlines), or ``None``
    if either is missing.

    Uses the LAST occurrence of each marker. The shell echoes the
    typed command (which contains BOTH markers as printf arguments)
    on its own line above the actual output — the *real* BEGIN
    marker appears later, alone on its line, followed by the
    command's output, followed by the real END marker alone on its
    line. ``rfind`` skips past the echo line in both directions."""
    bpos = text.rfind(begin_token)
    if bpos < 0:
        return None
    bend = text.find("\n", bpos)
    if bend < 0:
        bend = bpos + len(begin_token)
    epos = text.rfind(end_token)
    if epos <= bend:
        return None
    return text[bend:epos].strip("\r\n")


class ShellRunAgent(Agent):
    """Run a shell command, return its stdout from the screen."""

    name = "shell_run"

    async def run(
        self,
        *,
        command: str,
        timeout: float = 12.0,
        poll_interval: float = 1.2,
        clear_first: bool = True,
        record_label: str = "shell_run",
    ) -> ShellRunOutcome:
        if self.ctx.keyboard is None:
            return ShellRunOutcome(
                success=False, reason="no keyboard in context",
                data={"stdout": "", "marker_id": "", "command": command},
            )
        cmd = (command or "").strip()
        if not cmd:
            return ShellRunOutcome(
                success=False, reason="empty command",
                data={"stdout": "", "marker_id": "", "command": ""},
            )

        marker_id = _new_marker_id()
        begin_token = f"{_BEGIN_PREFIX}{marker_id}"
        end_token = f"{_END_PREFIX}{marker_id}"

        if clear_first:
            await self._reset_shell()

        wrapped = (
            f"printf '{begin_token}\\n'; {cmd}; printf '{end_token}\\n'"
        )
        try:
            await self.ctx.keyboard.send_text(wrapped)
        except TypeError:
            await self.ctx.keyboard.send_text(wrapped)
        except Exception as e:
            return ShellRunOutcome(
                success=False, reason=f"typing command failed: {e}",
                data={"stdout": "", "marker_id": marker_id, "command": cmd},
            )
        await asyncio.sleep(0.3)
        try:
            await self.ctx.keyboard.send_keystroke("Enter")
        except Exception as e:
            return ShellRunOutcome(
                success=False, reason=f"Enter (submit) failed: {e}",
                data={"stdout": "", "marker_id": marker_id, "command": cmd},
            )

        deadline = asyncio.get_event_loop().time() + timeout
        last_text = ""
        attempts = 0
        # OCR cascade per poll: try a tight centre crop first (where
        # the focused terminal window usually lives — a freshly
        # opened gnome-terminal occupies roughly the middle 70 % of
        # the screen), then fall back to a wider centre, then the
        # full frame. The reason for the cascade: nanonets-ocr-s
        # caps at ~1500 output tokens. A large background app like
        # LibreOffice burns all of that on its toolbar before
        # reaching the terminal text. Cropping to the area we expect
        # the terminal to occupy keeps the markers inside the OCR
        # budget.
        ocr_crops: list[tuple[float, float, float, float] | None] = [
            (0.10, 0.10, 0.90, 0.80),
            (0.05, 0.05, 0.95, 0.95),
            None,  # full frame
        ]
        while asyncio.get_event_loop().time() < deadline:
            attempts += 1
            await asyncio.sleep(poll_interval)
            crop = ocr_crops[(attempts - 1) % len(ocr_crops)]
            ocr_kwargs: dict = {
                "record_label": f"{record_label}_ocr_{attempts}",
            }
            if crop is None:
                ocr_kwargs["region"] = "full"
            else:
                ocr_kwargs["crop"] = crop
            ocr_out = await OcrAgent(self.ctx).run(**ocr_kwargs)
            text = ""
            if ocr_out.success and ocr_out.data:
                text = str(ocr_out.data.get("text") or "")
            last_text = text
            stdout = _extract_between_markers(
                text, begin_token, end_token,
            )
            if stdout is None:
                # OCR may have mangled the marker letters. Fall back
                # to fuzzy: look for the marker prefix + first 5
                # chars of the id, allowing one substitution.
                stdout = _fuzzy_extract(
                    text, _BEGIN_PREFIX, _END_PREFIX, marker_id,
                )
            if stdout is not None:
                # Strip the leading command-echo line (the shell
                # often re-echoes the wrapped command between the
                # marker and the real output).
                cleaned = _strip_echo_line(stdout, wrapped, cmd)
                print(
                    f"   ShellRunAgent: captured {len(cleaned)} char(s) "
                    f"of stdout (marker={marker_id}, attempts={attempts})"
                )
                return ShellRunOutcome(
                    success=True,
                    reason=f"captured stdout ({len(cleaned)} chars)",
                    data={
                        "stdout": cleaned,
                        "marker_id": marker_id,
                        "command": cmd,
                        "attempts": attempts,
                    },
                )

        # Log a head of the OCR'd text so we can see WHY the markers
        # weren't found — usually OCR mangled the id, less often the
        # terminal didn't receive the command.
        if last_text:
            preview = last_text[:600].replace("\n", " ⏎ ")
            logger.warning(
                "ShellRunAgent timed out; OCR last seen (first 600 "
                "chars): %s", preview,
            )
        return ShellRunOutcome(
            success=False,
            reason=(
                f"timed out after {timeout:.1f}s waiting for markers "
                f"({begin_token}…{end_token}); shell may be busy or "
                f"foreground process is eating keystrokes"
            ),
            data={
                "stdout": "",
                "marker_id": marker_id,
                "command": cmd,
                "attempts": attempts,
                "last_ocr": last_text[:600],
            },
        )

    async def _reset_shell(self) -> None:
        """Break out of any foreground process, maximise the focused
        window so the terminal dominates the OCR frame (otherwise a
        big background app like LibreOffice burns through the OCR
        token budget before reaching the terminal text), and clear
        the screen so the next command's output starts fresh."""
        kb = self.ctx.keyboard
        if kb is None:
            return
        try:
            await kb.send_key_combo(["ctrl"], "c")
            await asyncio.sleep(0.4)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.3)
            # Second Ctrl+C in case the first interrupted a less/pager
            # rather than reaching a shell prompt.
            await kb.send_key_combo(["ctrl"], "c")
            await asyncio.sleep(0.3)
            # Maximise the focused window. Super+Up = GNOME maximise.
            # If the window is already maximised this is a no-op.
            await kb.send_key_combo(["super"], "Up")
            await asyncio.sleep(0.6)
            await kb.send_text("clear")
            await asyncio.sleep(0.2)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.6)
        except Exception as e:
            logger.debug("shell reset best-effort failed: %s", e)


def _fuzzy_extract(
    text: str, begin_prefix: str, end_prefix: str, marker_id: str,
) -> str | None:
    """OCR-tolerant fallback for :func:`_extract_between_markers`.

    OCR commonly drops or substitutes one character in the random
    marker id (e.g. ``RCTRAUMB`` → ``RTRAUMB``), which defeats
    position-by-position matching. We instead match by *prefix
    plus character-overlap*: a line whose stripped form starts
    with the BEGIN/END prefix and whose tail shares enough chars
    with the marker id is treated as a hit. Position is ignored
    so a single OCR insertion/deletion doesn't break the match.

    The screen is cleared before each run, so the only ``TEBEGIN``/
    ``TEEND`` strings on the frame come from THIS run."""
    lines = text.splitlines()
    id_chars = set(marker_id)
    threshold = max(len(marker_id) - 3, 4)  # need ≥4 of N chars

    def matches(prefix: str, line: str) -> bool:
        s = line.strip()
        if not s.startswith(prefix):
            return False
        tail = s[len(prefix):]
        # Stop the tail at the first non-letter character so a
        # following command (in the echo line) doesn't pollute
        # the overlap count.
        cut = 0
        for ch in tail:
            if not ch.isalpha():
                break
            cut += 1
        if cut == 0:
            return False
        tail_letters = tail[:cut]
        overlap = sum(1 for ch in tail_letters if ch in id_chars)
        return overlap >= threshold

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, ln in enumerate(lines):
        if matches(begin_prefix, ln):
            # Take the LAST begin (skips command-echo line above).
            begin_idx = i
        if matches(end_prefix, ln) and (
            begin_idx is None or i > begin_idx
        ):
            end_idx = i
    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        return None
    return "\n".join(lines[begin_idx + 1: end_idx]).strip("\r\n")


def _strip_echo_line(captured: str, wrapped: str, cmd: str) -> str:
    """Drop the leading command-echo line if the shell printed our
    wrapped command above its output. We compare on a substring of
    the actual command so OCR distortions in the marker portion
    don't defeat the match.

    Falls back to returning ``captured`` unchanged if no echo line
    looks present."""
    if not captured:
        return captured
    lines = captured.splitlines()
    # Look for the first line that contains a recognisable chunk of
    # the command (10+ chars of it) — that's the echo. Everything
    # after it is the real stdout.
    needle = cmd if len(cmd) <= 40 else cmd[:40]
    needle_short = needle[:10] if len(needle) >= 10 else needle
    for i, ln in enumerate(lines):
        if needle_short and needle_short in ln:
            rest = "\n".join(lines[i + 1:]).rstrip()
            return rest if rest else captured.strip()
    return captured.strip()
