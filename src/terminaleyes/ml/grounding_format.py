"""Prompt / response formatting for grounding fine-tune.

Mirrors :mod:`terminaleyes.ml.format` but for the much narrower
grounding task — input is `(frame, target query)`, output is a
single point in normalised image coordinates.

Output format follows ShowUI's convention so the head can be
plugged in (or compared against) the existing llama.cpp ShowUI-2B
endpoint without re-mapping coordinates:

    <point>x,y</point>

where ``x`` and ``y`` are in ``[0, 1]`` rounded to 4 decimal
places. We keep the angle-bracket fence rather than raw JSON
because the trained model output is tiny (~13 tokens) and the
fence makes parsing trivially regex-able.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are a UI grounding model. Given a screenshot and a short "
    "natural-language description of a UI element, reply with the "
    "centre of that element as a single point in normalised image "
    "coordinates, formatted exactly as <point>x,y</point> where "
    "x and y are decimals in [0, 1]."
)


@dataclass
class GroundingSample:
    prompt: str
    response: str
    image_path: str
    target_xy: tuple[float, float]  # normalised


def format_prompt(query: str) -> str:
    """Build the user-side text. The image is attached separately
    by the loader; we don't embed any token for it here so the same
    string works across mlx-vlm and HF processors."""
    q = (query or "").strip()
    return f"target: {q}"


def format_response(x: float, y: float) -> str:
    x = max(0.0, min(1.0, float(x)))
    y = max(0.0, min(1.0, float(y)))
    return f"<point>{round(x, 4)},{round(y, 4)}</point>"


def format_sample(row: dict) -> GroundingSample | None:
    """Convert a build_grounding_dataset.py row into a training
    sample. Returns None when the row is missing usable fields."""
    img = row.get("image_path")
    query = row.get("query") or ""
    center = row.get("center")
    if not img or not query or not center or len(center) != 2:
        return None
    x, y = float(center[0]), float(center[1])
    return GroundingSample(
        prompt=format_prompt(query),
        response=format_response(x, y),
        image_path=str(img),
        target_xy=(x, y),
    )


_POINT_RE = re.compile(
    r"<\s*point\s*>\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*<\s*/\s*point\s*>",
    re.IGNORECASE,
)


def parse_response(text: str) -> tuple[float, float] | None:
    """Pull ``(x, y)`` out of model output. Returns None on
    unrecoverable garbage. Tolerant of stray whitespace and
    surrounding chatter — the fence is unique enough."""
    if not text:
        return None
    m = _POINT_RE.search(text)
    if m:
        try:
            x = float(m.group(1)); y = float(m.group(2))
        except ValueError:
            return None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return (x, y)
    # Fallback: try the legacy ShowUI style of `[x, y]` JSON list.
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, list) and len(obj) == 2:
            x, y = float(obj[0]), float(obj[1])
            if 0 <= x <= 1 and 0 <= y <= 1:
                return (x, y)
    except Exception:
        pass
    return None
