#!/usr/bin/env python3
"""build_grounding_dataset.py — synthesise a grounding-fine-tune
dataset from terminaleyes session frames.

The visual-servo homer doesn't persist structured ``(target, xy)``
traces today, only debug PNGs. So we synthesise labels from what's
on disk: every captured frame gets fed through tesseract, and each
detected word/phrase becomes a grounding sample.

Output schema (JSONL, one row per detected region)::

    {
      "image_path": "runs/<run_id>/0003_191230_homer_capture.png",
      "image_size": [1920, 1080],
      "query": "Terminal",
      "bbox": [x0, y0, x1, y1],   // pixel coords on the FRAME
      "center": [cx, cy],         // normalised (0..1, 0..1)
      "conf": 92.5                // tesseract confidence 0-100
    }

The grounding fine-tune target reads ``image_path + query`` and
predicts ``center``. We deliberately keep the bbox so a future
trainer can supervise on bbox-IoU instead of point distance.

Filtering:
  * Drop OCR detections with confidence < ``--min-conf`` (default
    60). Tesseract's confidence floor is 0; anything below ~50 is
    usually noise.
  * Drop tokens shorter than ``--min-chars`` (default 2) — single
    characters are too ambiguous a query.
  * Drop tokens that are pure punctuation / digits-only if
    ``--alpha-only`` is set (default off).

Usage::

    python scripts/build_grounding_dataset.py \\
        --runs-root ~/.local/share/terminaleyes/runs \\
        --out       data/ml/grounding \\
        --min-conf  70 \\
        --max-per-frame 40

Run-time scales linearly with the number of frames * tesseract
latency (~0.3-0.8s per 1920x1080 frame).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _resolve_runs_root() -> Path:
    return Path.home() / ".local/share/terminaleyes/runs"


def _iter_frames(runs_root: Path):
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for png in sorted(run_dir.glob("*.png")):
            yield png


def _ocr_words(image_path: Path) -> list[dict]:
    """Run tesseract --psm 6 with TSV output. Returns one dict per
    detected word: ``{text, conf, left, top, width, height}``."""
    import pytesseract
    from PIL import Image
    try:
        img = Image.open(image_path)
    except Exception:
        return []
    try:
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
    except Exception as e:
        print(f"  ! tesseract failed on {image_path.name}: {e}",
              file=sys.stderr)
        return []
    out: list[dict] = []
    for i, txt in enumerate(data.get("text", [])):
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = 0.0
        out.append({
            "text": txt,
            "conf": conf,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
        })
    return out


def _looks_like_token(text: str, *, min_chars: int,
                      alpha_only: bool) -> bool:
    if len(text) < min_chars:
        return False
    stripped = text.strip()
    if all(not c.isalnum() for c in stripped):
        return False  # pure punctuation
    if alpha_only and not any(c.isalpha() for c in stripped):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", type=Path, default=_resolve_runs_root())
    ap.add_argument("--out", type=Path, default=Path("data/ml/grounding"))
    ap.add_argument("--min-conf", type=float, default=70.0)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--alpha-only", action="store_true")
    ap.add_argument(
        "--max-per-frame", type=int, default=40,
        help="Cap detections per frame to avoid one busy screen "
             "dominating the dataset.",
    )
    ap.add_argument(
        "--limit-frames", type=int, default=0,
        help="Process only the first N frames (debug). 0 = all.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from PIL import Image  # noqa: F401
        import pytesseract  # noqa: F401
    except Exception as e:
        print("missing deps: pip install pillow pytesseract", file=sys.stderr)
        print(f"  ({e})", file=sys.stderr)
        return 2

    if not args.runs_root.exists():
        print(f"runs root not found: {args.runs_root}", file=sys.stderr)
        return 2

    frames = list(_iter_frames(args.runs_root))
    if args.limit_frames:
        frames = frames[: args.limit_frames]
    if not frames:
        print(f"no PNG frames under {args.runs_root}", file=sys.stderr)
        return 1

    print(f"scanning {len(frames)} frame(s) under {args.runs_root}")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    per_frame_counts: Counter[str] = Counter()
    skipped_conf = skipped_short = 0

    for i, png in enumerate(frames, 1):
        if i % 50 == 0 or i == len(frames):
            print(
                f"  [{i:>5}/{len(frames)}] rows={len(rows):>6} "
                f"skipped(conf<{args.min_conf:.0f})={skipped_conf} "
                f"skipped(short)={skipped_short}"
            )
        try:
            from PIL import Image
            with Image.open(png) as im:
                W, H = im.size
        except Exception:
            continue
        words = _ocr_words(png)
        if not words:
            continue
        keep: list[dict] = []
        for w in words:
            if w["conf"] < args.min_conf:
                skipped_conf += 1
                continue
            if not _looks_like_token(
                w["text"],
                min_chars=args.min_chars,
                alpha_only=args.alpha_only,
            ):
                skipped_short += 1
                continue
            keep.append(w)
        if not keep:
            continue
        if len(keep) > args.max_per_frame:
            rng.shuffle(keep)
            keep = keep[: args.max_per_frame]
        rel = png.relative_to(args.runs_root)
        per_frame_counts[str(rel.parent)] += len(keep)
        for w in keep:
            cx = (w["left"] + w["width"] / 2.0) / W
            cy = (w["top"] + w["height"] / 2.0) / H
            rows.append({
                "image_path": str(rel),
                "image_size": [W, H],
                "query": w["text"],
                "bbox": [
                    w["left"], w["top"],
                    w["left"] + w["width"],
                    w["top"] + w["height"],
                ],
                "center": [round(cx, 4), round(cy, 4)],
                "conf": w["conf"],
            })

    if not rows:
        print("no usable detections — try lowering --min-conf", file=sys.stderr)
        return 1

    # Split by run-id (trajectory_id) so all rows of one run land in
    # the same split — same convention as build_ml_dataset.py.
    args.out.mkdir(parents=True, exist_ok=True)
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        run_id = r["image_path"].split("/", 1)[0]
        by_run[run_id].append(r)
    run_ids = sorted(by_run.keys())
    rng.shuffle(run_ids)
    n = len(run_ids)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val_ids = set(run_ids[:n_val])
    test_ids = set(run_ids[n_val: n_val + n_test])
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rid, items in by_run.items():
        if rid in val_ids:
            splits["val"].extend(items)
        elif rid in test_ids:
            splits["test"].extend(items)
        else:
            splits["train"].extend(items)

    for name, items in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "total_rows": len(rows),
        "total_runs": len(by_run),
        "frames_seen": len(frames),
        "splits": {
            name: {"rows": len(items),
                   "runs": len({r["image_path"].split("/", 1)[0]
                                for r in items})}
            for name, items in splits.items()
        },
        "filters": {
            "min_conf": args.min_conf,
            "min_chars": args.min_chars,
            "alpha_only": args.alpha_only,
            "max_per_frame": args.max_per_frame,
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print()
    print(f"wrote {len(rows)} rows across {len(by_run)} runs → {args.out}")
    for name, items in splits.items():
        print(f"  {name:>5}: {len(items):>6} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
