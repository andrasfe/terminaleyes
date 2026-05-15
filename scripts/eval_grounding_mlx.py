#!/usr/bin/env python3
"""eval_grounding_mlx.py — pixel-distance + in-bbox accuracy for a
grounding adapter produced by ``train_grounding_mlx.py``.

Per-row metrics:
  * **parse_ok** — model produced a syntactically valid
    ``<point>x,y</point>``.
  * **dist_norm** — Euclidean distance in normalised coords.
  * **dist_px** — same distance scaled by frame size.
  * **in_bbox** — predicted point falls inside the gold bbox.

Aggregate:
  * parse_ok rate
  * mean/median normalised distance over parsed rows
  * in-bbox rate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/ml/grounding"))
    ap.add_argument("--split", choices=("train", "val", "test"),
                    default="val")
    ap.add_argument("--runs-root", type=Path,
                    default=Path.home() / ".local/share/terminaleyes/runs")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows (0 = all).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows_path = args.dataset / f"{args.split}.jsonl"
    if not rows_path.exists():
        print(f"split not found: {rows_path}", file=sys.stderr); return 2
    rows = _load_jsonl(rows_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no rows", file=sys.stderr); return 1

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    try:
        from PIL import Image
        from terminaleyes.ml.grounding_format import (
            SYSTEM_PROMPT, format_prompt, parse_response,
        )
        from terminaleyes.agents.ml_planner import _LoadedModel
    except Exception as e:
        print(f"missing deps: {e}", file=sys.stderr); return 2

    loader = _LoadedModel(args.adapter)
    loader.load()

    parsed = 0
    in_bbox = 0
    dists = []  # normalised
    per_row = []

    # Reusable: build the same chat shape the trainer used so
    # train/inference don't drift.
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    for row in rows:
        img_path = args.runs_root / row["image_path"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB").resize((448, 252))
        except Exception:
            continue
        user_text = format_prompt(row["query"])
        # System + user text as a single combined string — keep
        # the trainer's chat template happy without coupling to
        # mlx-vlm's evolving multi-message API.
        prompt = f"{SYSTEM_PROMPT}\n\n{user_text}"
        formatted = apply_chat_template(
            loader.processor, getattr(loader.model, "config", None),
            prompt, num_images=1,
        )
        try:
            out = generate(
                loader.model, loader.processor, formatted,
                image=[img], max_tokens=120, temperature=0.1,
            )
            raw = getattr(out, "text", out) if not isinstance(out, str) else out
        except Exception as e:
            raw = ""
            print(f"  ! gen err: {e}", file=sys.stderr)
        pred = parse_response(raw)
        gold = tuple(row["center"])
        bbox = row.get("bbox") or []
        W, H = row.get("image_size") or [1, 1]

        row_record = {
            "image_path": row["image_path"],
            "query": row["query"],
            "gold": gold,
            "raw": raw[:120],
            "pred": pred,
            "dist_norm": None,
            "in_bbox": False,
        }
        if pred is not None:
            parsed += 1
            d = math.hypot(pred[0] - gold[0], pred[1] - gold[1])
            dists.append(d)
            row_record["dist_norm"] = round(d, 4)
            if len(bbox) == 4:
                px = pred[0] * W; py = pred[1] * H
                if bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]:
                    in_bbox += 1
                    row_record["in_bbox"] = True
        per_row.append(row_record)

    total = len(per_row)
    if total == 0:
        print("no rows evaluated", file=sys.stderr); return 1

    print()
    print(f"split={args.split}  rows={total}  adapter={args.adapter}")
    print(f"  parse_ok       : {parsed}/{total} ({100.0 * parsed / total:.1f}%)")
    if dists:
        dists_sorted = sorted(dists)
        median = dists_sorted[len(dists_sorted) // 2]
        mean = sum(dists) / len(dists)
        print(f"  dist (norm)    : mean={mean:.4f}  median={median:.4f}")
        # As an intuitive number — on a 1920×1080 frame, a normalised
        # distance of 0.05 is ~96 px horizontally, ~54 px vertically.
        print(
            f"  dist (px,1920) : mean≈{mean * 1920:.0f}  "
            f"median≈{median * 1920:.0f}"
        )
    print(f"  in-bbox        : {in_bbox}/{total} "
          f"({100.0 * in_bbox / total:.1f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "summary": {
                "rows": total, "parse_ok": parsed,
                "in_bbox": in_bbox,
                "dist_mean": (sum(dists) / len(dists)) if dists else None,
                "dist_median": (
                    sorted(dists)[len(dists) // 2] if dists else None
                ),
            },
            "rows": per_row,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote per-row report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
