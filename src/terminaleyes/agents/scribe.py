"""ScribeAgent — append a one-paragraph journal entry per controller run.

Tier-1, post-completion, best-effort. Fired by :class:`ControllerAgent`
after the final-state verifier runs, with the verdict + the OCR'd
final-state text already in hand. The scribe asks the planner LLM
to write a short summary of:

  * what was attempted
  * whether it appears to have succeeded (with a 1-clause reason)
  * what's notable about the screen state right now

…and appends the result to the journal file (default
``~/.local/share/terminaleyes/journal.md``, override with
``TERMINALEYES_JOURNAL``). The journal is consumed by
:func:`controller.load_journal_tail` and injected into the LLM
planner's prompt so future planning has episodic memory of recent
outcomes.

The agent is best-effort throughout — every failure path returns a
clean :class:`ScribeOutcome` and never raises. Journaling failure
must not block the controller's outcome.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from terminaleyes.agents.base import Agent, Outcome

logger = logging.getLogger(__name__)


DEFAULT_JOURNAL_PATH = (
    Path.home() / ".local" / "share" / "terminaleyes" / "journal.md"
)
JOURNAL_MAX_ENTRIES = 500


def journal_path() -> Path:
    env = os.environ.get("TERMINALEYES_JOURNAL")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_JOURNAL_PATH


def append_entry(entry: str) -> Path:
    """Append a markdown entry block to the journal, FIFO-rotating
    once we exceed :data:`JOURNAL_MAX_ENTRIES`. Returns the path
    written. Caller is responsible for formatting the entry."""
    p = journal_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    if p.exists():
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            logger.debug("journal read failed: %s", e)
    blocks = [b for b in text.split("\n---\n") if b.strip()] if text else []
    blocks.append(entry.strip())
    if len(blocks) > JOURNAL_MAX_ENTRIES:
        blocks = blocks[-JOURNAL_MAX_ENTRIES:]
    out = "\n---\n".join(blocks) + "\n"
    try:
        p.write_text(out, encoding="utf-8")
    except OSError as e:
        logger.warning("journal write failed: %s", e)
    return p


def read_tail(n: int = 20) -> list[str]:
    """Return the last ``n`` journal entries, oldest-first. Empty
    list if the file is missing or unreadable."""
    p = journal_path()
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("journal read failed: %s", e)
        return []
    blocks = [b.strip() for b in text.split("\n---\n") if b.strip()]
    return blocks[-n:] if n > 0 else blocks


@dataclass
class ScribeOutcome(Outcome):
    """``data['entry']`` carries the formatted journal entry on success."""


class ScribeAgent(Agent):
    """Write a one-paragraph journal entry summarising a finished run."""

    name = "scribe"

    async def run(
        self,
        *,
        intent: str,
        run_id: str,
        success: bool,
        verdict_reason: str,
        ocr_text: str,
        max_tokens: int = 220,
    ) -> ScribeOutcome:
        if self.ctx.vision_client is None:
            return ScribeOutcome(
                success=False, reason="no vision client; scribe skipped",
                data={"entry": ""},
            )
        if not intent:
            return ScribeOutcome(
                success=False, reason="empty intent", data={"entry": ""},
            )

        # Cap OCR text fed to the LLM so a noisy 2000-char screen
        # doesn't blow tokens. The model only needs enough to
        # describe the salient state.
        ocr_for_prompt = ocr_text.strip()
        if len(ocr_for_prompt) > 1200:
            ocr_for_prompt = ocr_for_prompt[:1200] + "\n…(truncated)…"

        sys_prompt = (
            "You are a journal scribe. Given the user's intent, the "
            "system's final verdict, and the OCR-extracted text from "
            "the final screen, write a SHORT entry (3-4 lines) for the "
            "operator's journal.\n\n"
            "Required structure (plain text, no markdown headers):\n"
            "  - Line 1: 'Tried: <one sentence restating the intent>'\n"
            "  - Line 2: 'Result: <succeeded|failed> — <one-clause reason "
            "from the verdict>'\n"
            "  - Line 3: 'Screen: <one-clause description of what's "
            "visibly on screen now — the foreground app, key text, "
            "any error/output text>'\n"
            "  - Line 4 (optional): 'Note: <anything worth remembering "
            "for next time, e.g. an unusual app state or a working "
            "command>'\n\n"
            "Rules:\n"
            "  * Plain text only. No markdown, no JSON, no preamble.\n"
            "  * Be concrete — quote screen text where useful.\n"
            "  * Do NOT invent details that aren't in the OCR text.\n"
            "  * Keep each line under ~120 characters."
        )
        user = (
            f"Intent:\n{intent}\n\n"
            f"Final verdict: {'succeeded' if success else 'failed'}\n"
            f"Reason: {verdict_reason or '(none)'}\n\n"
            f"OCR-extracted screen text:\n-----\n{ocr_for_prompt}\n-----\n\n"
            "Write the journal entry."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        # 2-pass attempt: vision_model at temp=0.4, then ocr_model
        # at temp=0.4 if available. Nemotron-Nano returns empty at
        # low temperatures for this prompt shape; the fallback gives
        # a different model class a shot before we give up.
        attempts = [
            (self.ctx.vision_model, 0.4),
            (self.ctx.ocr_model or self.ctx.vision_model, 0.4),
        ]
        body = ""
        last_err = ""
        for model_name, temp in attempts:
            if not model_name:
                continue
            try:
                resp = await self.ctx.vision_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                body = (resp.choices[0].message.content or "").strip()
                if body:
                    break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.debug(
                    "scribe attempt %s failed: %s", model_name, e,
                )
        if not body:
            # Synthesize a deterministic minimal entry rather than
            # leaving the journal empty. Episodic memory is more
            # useful than nothing — and the planner prompt only
            # cares about intent + outcome anyway.
            body = (
                f"Tried: {intent}\n"
                f"Result: {'succeeded' if success else 'failed'} — "
                f"{verdict_reason or '(no verdict)'}\n"
                f"Screen: "
                f"{(ocr_for_prompt.splitlines() or [''])[0][:120]}"
            )
            note = "scribe synthesized deterministic entry "
            note += f"(model returned empty: {last_err})" if last_err else \
                    "(model returned empty)"
            logger.info(note)

        ts = datetime.now().isoformat(timespec="seconds")
        verdict_word = "✓" if success else "✗"
        entry = (
            f"## {ts} · run {run_id} · {verdict_word}\n"
            f"{body.strip()}"
        )
        try:
            path = append_entry(entry)
            print(f"   Scribe: appended entry to {path}")
        except Exception as e:
            return ScribeOutcome(
                success=False, reason=f"scribe append failed: {e}",
                data={"entry": entry},
            )
        return ScribeOutcome(
            success=True,
            reason=f"journal entry appended ({len(body)} chars)",
            data={"entry": entry, "ts": time.time()},
        )
