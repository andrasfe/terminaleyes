"""Prompt / response formatting for the learned planner.

The contract between the dataset and the model lives here so a row
from ``scripts/build_ml_dataset.py`` and a runtime decision from
:class:`MlPlannerAgent` produce *identical* prompt strings — no
training/inference skew.

Prompt shape::

    <SYSTEM>
    You are the terminaleyes controller. Given the current screen
    image, the user's intent, and the steps already taken in this
    run, emit the next agent call as a single JSON object.

    Available agents:
      - keys: send a keyboard shortcut. kwargs: modifiers, key.
      - type: type text. kwargs: text, submit.
      - launch: open a desktop app. kwargs: app, platform.
      - ...

    Reply EXACTLY with: {"agent": "<name>", "kwargs": { ... }}
    </SYSTEM>

    <USER>
    intent: <free-form text>
    history:
      1. launch {"app":"terminal","platform":"linux"} → ok
      2. wake {} → ok
    next step:
    </USER>
    <IMAGE: frame_before>

Response shape (one line of JSON, terminated by EOS)::

    {"agent": "keys", "kwargs": {"modifiers": ["super"], "key": "l"}}

We keep this format model-agnostic — same string goes into UI-TARS,
Qwen2.5-VL, or ShowUI by wrapping it in each model's chat template
at the loader boundary, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable


# Description block injected into every prompt. Kept short so the
# image budget dominates the token count. We extract the same
# REGISTRY descriptions the controller uses so this stays in sync
# with the agent layer; if a new agent is added, training prompts
# pick it up automatically.
def _agent_descriptions() -> str:
    # Lazy import: this module is also used by the dataset builder
    # which we want to keep dependency-light. Importing controller
    # would pull in the full agent stack including cv2.
    try:
        from terminaleyes.agents.controller import REGISTRY
    except Exception:
        return "  (controller registry unavailable)"
    lines: list[str] = []
    for name, (_, desc) in REGISTRY.items():
        short = desc.split(". ", 1)[0]
        lines.append(f"  - {name}: {short}.")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = (
    "You are the terminaleyes controller. Given the current screen "
    "image, the user's intent, and the steps already taken in this "
    "run, emit the next agent call as a single JSON object.\n\n"
    "Available agents:\n"
    "{agent_descriptions}\n\n"
    'Reply EXACTLY with: {{"agent": "<name>", "kwargs": {{ ... }}}}'
)


@dataclass
class FormattedSample:
    """One training row after format conversion.

    ``prompt`` is everything the model sees as input (system +
    user text). ``response`` is the supervised target (JSON line).
    ``image_path`` is the on-disk frame the model also sees.
    """

    prompt: str
    response: str
    image_path: str
    intent: str
    agent: str  # for stratified split / per-class accuracy


def format_history(history: Iterable[dict]) -> str:
    """One ``N. <agent> <kwargs> → ok|fail`` line per prior step."""
    lines: list[str] = []
    for i, h in enumerate(history, 1):
        agent = h.get("agent", "?")
        kwargs = h.get("kwargs") or {}
        ok = h.get("success", True)
        verdict = "ok" if ok else "fail"
        lines.append(
            f"  {i}. {agent} "
            f"{json.dumps(kwargs, ensure_ascii=False)} → {verdict}"
        )
    return "\n".join(lines) if lines else "  (none)"


def format_prompt(*, intent: str, history: Iterable[dict]) -> str:
    """Build the prompt string (system + user blocks)."""
    sys_block = SYSTEM_PROMPT_TEMPLATE.format(
        agent_descriptions=_agent_descriptions(),
    )
    user_block = (
        f"intent: {intent}\n"
        f"history:\n{format_history(history)}\n"
        "next step:"
    )
    return f"<SYSTEM>\n{sys_block}\n</SYSTEM>\n\n<USER>\n{user_block}\n</USER>"


def format_response(*, agent: str, kwargs: dict | None) -> str:
    """Build the supervised target string."""
    return json.dumps(
        {"agent": agent, "kwargs": kwargs or {}},
        ensure_ascii=False,
        separators=(", ", ": "),
    )


def format_sample(row: dict) -> FormattedSample | None:
    """Convert a dataset row (from ``build_ml_dataset.py``) into a
    :class:`FormattedSample` suitable for training.

    Returns ``None`` when the row is missing a usable input frame
    or a labelled action.
    """
    frame = row.get("frame_before")
    action = row.get("action") or {}
    agent = action.get("agent")
    if not frame or not agent:
        return None
    prompt = format_prompt(
        intent=str(row.get("intent", "")),
        history=row.get("history") or [],
    )
    response = format_response(
        agent=str(agent),
        kwargs=action.get("kwargs") or {},
    )
    return FormattedSample(
        prompt=prompt,
        response=response,
        image_path=str(frame),
        intent=str(row.get("intent", "")),
        agent=str(agent),
    )


def parse_response(text: str) -> dict | None:
    """Inverse of :func:`format_response`: pull ``{"agent":..., "kwargs":...}``
    out of arbitrary model output.

    The model may emit extra whitespace, a Markdown fence, or a
    trailing reasoning line. We accept the first JSON object that
    parses and contains an ``agent`` key. Returns ``None`` on
    unrecoverable garbage.
    """
    if not text:
        return None
    raw = text.strip()
    # Strip Markdown fences if present.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    # Try direct parse first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "agent" in obj:
            kw = obj.get("kwargs")
            obj["kwargs"] = kw if isinstance(kw, dict) else {}
            return obj
    except Exception:
        pass
    # Fall back to brace-balanced scan for the first top-level
    # JSON object — handles "{...} extra text" / "extra text {...}".
    start = raw.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(raw)):
            ch = raw[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start: end + 1])
                        if isinstance(obj, dict) and "agent" in obj:
                            kw = obj.get("kwargs")
                            obj["kwargs"] = (
                                kw if isinstance(kw, dict) else {}
                            )
                            return obj
                    except Exception:
                        pass
                    break
        start = raw.find("{", start + 1)
    return None
