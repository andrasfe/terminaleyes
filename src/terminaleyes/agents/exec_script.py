"""ExecScriptAgent — write a script to a tmp file on the target,
chmod +x, and execute it. Captures stdout between markers.

Tier-3 workflow primitive. Different from :class:`ScriptAgent` in
that the script body is *written to a file* via a quoted heredoc
(``<< 'EOF'``), rather than typed line-by-line into the live shell.
Three benefits over line-by-line typing:

  * Heredoc lines aren't interpreted until ``EOF`` closes, so shell
    metacharacters in the body (``$``, ``$()``, ``` ` ``` , globs,
    ``&&``) are preserved verbatim. With line-by-line typing, each
    line is evaluated at submit time.
  * The script is reusable — it lives at the configured path and
    can be re-invoked later.
  * Failure of one body line aborts the script execution, not the
    heredoc collection — the file ends up complete, you just see
    the error in the captured output.

Execution wraps the run in unique ASCII markers (same trick as
:class:`ShellRunAgent`) so the actual stdout can be sliced out of
whatever else is on the OCR'd frame.

Inputs
------
``script``      — verbatim body. Don't include a shebang unless you
                  want one in the file; we'll prepend ``#!/bin/bash``
                  only if the body doesn't start with ``#!``.
``filename``    — name of the file under ``/tmp``. Defaults to
                  ``te-script-<random>.sh``.
``capture_output``  — when True (default) waits for both markers in
                  OCR and returns the slice as ``data['stdout']``.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.agents.ocr import OcrAgent
from terminaleyes.agents.shell_run import (
    _BEGIN_PREFIX, _END_PREFIX, _extract_between_markers, _fuzzy_extract,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecScriptOutcome(Outcome):
    """``data['stdout']`` — captured output between markers (verbatim).
    ``data['path']`` — file path written on the target.
    ``data['marker_id']`` — id used for the BEGIN/END markers."""


def _new_marker_id() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(secrets.choice(alphabet) for _ in range(8))


def _normalise_script(body: str) -> str:
    body = body.strip("\r\n")
    if not body.lstrip().startswith("#!"):
        body = "#!/bin/bash\nset -e\n" + body
    return body + "\n"


class ExecScriptAgent(Agent):
    """Write a script to /tmp, chmod +x, execute, return its stdout."""

    name = "exec_script"

    async def run(
        self,
        *,
        script: str,
        filename: str | None = None,
        capture_output: bool = True,
        timeout: float = 20.0,
        poll_interval: float = 1.2,
        clear_first: bool = True,
        record_label: str = "exec_script",
    ) -> ExecScriptOutcome:
        if self.ctx.keyboard is None:
            return ExecScriptOutcome(
                success=False, reason="no keyboard in context",
                data={"stdout": "", "path": "", "marker_id": ""},
            )
        body = _normalise_script(script or "")
        if not body.strip():
            return ExecScriptOutcome(
                success=False, reason="empty script",
                data={"stdout": "", "path": "", "marker_id": ""},
            )

        marker_id = _new_marker_id()
        begin = f"{_BEGIN_PREFIX}{marker_id}"
        end = f"{_END_PREFIX}{marker_id}"
        path = (
            "/tmp/" + filename if filename
            else f"/tmp/te-script-{marker_id}.sh"
        )
        eof = f"TE_EOF_{marker_id}"

        kb = self.ctx.keyboard

        if clear_first:
            await self._reset_shell()

        # ── 1) heredoc-write the script body
        # Quoted EOF (single-quoted on the open) disables every
        # interpolation: $, backticks, history expansion, glob.
        # Lines go in verbatim.
        try:
            await kb.send_text(f"cat > {path} << '{eof}'")
            await asyncio.sleep(0.3)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.4)

            for line in body.splitlines():
                await kb.send_text(line)
                await asyncio.sleep(0.2)
                await kb.send_keystroke("Enter")
                await asyncio.sleep(0.2)

            await kb.send_text(eof)
            await asyncio.sleep(0.3)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.5)
        except Exception as e:
            return ExecScriptOutcome(
                success=False,
                reason=f"heredoc write failed: {e}",
                data={
                    "stdout": "", "path": path, "marker_id": marker_id,
                },
            )

        # ── 2) chmod +x and execute, bracketed by markers
        exec_cmd = (
            f"printf '{begin}\\n'; chmod +x {path}; {path}; "
            f"printf '{end}\\n'"
        )
        try:
            await kb.send_text(exec_cmd)
            await asyncio.sleep(0.3)
            await kb.send_keystroke("Enter")
        except Exception as e:
            return ExecScriptOutcome(
                success=False,
                reason=f"exec submit failed: {e}",
                data={
                    "stdout": "", "path": path, "marker_id": marker_id,
                },
            )

        if not capture_output:
            await asyncio.sleep(1.0)
            try:
                if self.ctx.capture is not None:
                    frame = await self.ctx.capture.capture_frame()
                    self.ctx.record_frame(frame.image, label=record_label)
            except Exception as e:
                logger.debug("post-exec capture failed: %s", e)
            return ExecScriptOutcome(
                success=True,
                reason=f"executed {path} (output not captured)",
                data={
                    "stdout": "", "path": path, "marker_id": marker_id,
                    "captured": False,
                },
            )

        # ── 3) poll OCR for the marker pair
        deadline = asyncio.get_event_loop().time() + timeout
        last_text = ""
        attempts = 0
        ocr_crops: list[tuple[float, float, float, float] | None] = [
            (0.10, 0.10, 0.90, 0.80),
            (0.05, 0.05, 0.95, 0.95),
            None,
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
            stdout = _extract_between_markers(text, begin, end)
            if stdout is None:
                stdout = _fuzzy_extract(
                    text, _BEGIN_PREFIX, _END_PREFIX, marker_id,
                )
            if stdout is not None:
                print(
                    f"   ExecScriptAgent: ran {path}; captured "
                    f"{len(stdout)} char(s) of stdout "
                    f"(marker={marker_id}, attempts={attempts})"
                )
                return ExecScriptOutcome(
                    success=True,
                    reason=(
                        f"executed {path}; captured "
                        f"{len(stdout)} chars"
                    ),
                    data={
                        "stdout": stdout, "path": path,
                        "marker_id": marker_id,
                        "attempts": attempts, "captured": True,
                    },
                )

        return ExecScriptOutcome(
            success=False,
            reason=(
                f"executed {path} but timed out after {timeout:.1f}s "
                f"waiting for markers ({begin}…{end})"
            ),
            data={
                "stdout": "", "path": path, "marker_id": marker_id,
                "attempts": attempts, "captured": False,
                "last_ocr": last_text[:600],
            },
        )

    async def _reset_shell(self) -> None:
        kb = self.ctx.keyboard
        if kb is None:
            return
        try:
            await kb.send_key_combo(["ctrl"], "c")
            await asyncio.sleep(0.4)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.3)
            await kb.send_key_combo(["ctrl"], "c")
            await asyncio.sleep(0.3)
            await kb.send_key_combo(["super"], "Up")
            await asyncio.sleep(0.6)
            await kb.send_text("clear")
            await asyncio.sleep(0.2)
            await kb.send_keystroke("Enter")
            await asyncio.sleep(0.6)
        except Exception as e:
            logger.debug("shell reset best-effort failed: %s", e)
