#!/usr/bin/env python3
"""build_pointer_accel_dataset.py — collect homer step records into
a flat training table for the open-loop forward model.

Walks every ``<run>/homer/<*>/history.jsonl`` produced by
``VisualServoHomer`` (persisted via ``_record_step``) and emits one
``data/ml/pointer_accel/{train,val,test}.jsonl`` row per usable step.

Output schema (per row)::

    {
      "trajectory_id": "<run>/homer/<vs-id>",
      "step_idx": 4,
      "hid_dx": 18,            // sent HID delta x (signed)
      "hid_dy": -7,             // sent HID delta y (signed)
      "measured_dx_pct": 0.041, // observed cursor delta x (normalised)
      "measured_dy_pct": -0.018,// observed cursor delta y (normalised)
      "cursor_x_pct": 0.523,    // cursor position BEFORE the step
      "cursor_y_pct": 0.418,
      "note": "hsv_measured"
    }

The "before" cursor position lets a future model condition on edge
proximity (pointer accel is identity-shaped in the middle and gets
weird near borders).

Filtering:
  * Drop steps where ``hid_dx == 0 && hid_dy == 0`` (those are post-
    click "confirm" records, not pointer-accel samples).
  * Drop steps without a measured delta. The homer logs a measured
    delta even when HSV detection fails (note=openloop_fallback)
    via frame-diff fallback — those rows are NOISIER but still
    carry the right shape of the curve, so we keep them and let
    the training MSE handle the noise.

Splits 80/10/10 by trajectory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _iter_history_files(runs_root: Path):
    return runs_root.glob("**/homer/*/history.jsonl")


def _row_from_step(traj_id: str, idx: int, step: dict) -> dict | None:
    hid_dx = step.get("hid_dx")
    hid_dy = step.get("hid_dy")
    if hid_dx is None or hid_dy is None:
        return None
    if hid_dx == 0 and hid_dy == 0:
        return None
    mdx = step.get("measured_dx_pct")
    mdy = step.get("measured_dy_pct")
    if mdx is None or mdy is None:
        return None
    cursor = step.get("cursor_img")
    cx = cy = None
    if isinstance(cursor, list) and len(cursor) == 2:
        cx, cy = float(cursor[0]), float(cursor[1])
    return {
        "trajectory_id": traj_id,
        "step_idx": idx,
        "hid_dx": int(hid_dx),
        "hid_dy": int(hid_dy),
        "measured_dx_pct": float(mdx),
        "measured_dy_pct": float(mdy),
        "cursor_x_pct": cx,
        "cursor_y_pct": cy,
        "note": step.get("note", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/terminaleyes/runs",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("data/ml/pointer_accel"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hsv-only", action="store_true",
        help="Keep only rows where HSV cursor detection succeeded "
             "(note=hsv_measured). Frame-diff fallback rows are "
             "noisier; with the redglass cursor on the target we "
             "get HSV rows directly.",
    )
    ap.add_argument(
        "--since", type=str, default=None,
        help="Drop history.jsonl files older than this mtime "
             "(ISO timestamp or epoch). Use after switching cursor "
             "theme to filter out pre-theme runs.",
    )
    args = ap.parse_args()
    since_ts = None
    if args.since:
        try:
            since_ts = float(args.since)
        except ValueError:
            from datetime import datetime
            since_ts = datetime.fromisoformat(args.since).timestamp()

    if not args.runs_root.exists():
        print(f"runs root not found: {args.runs_root}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    n_files = 0
    n_lines = 0
    n_dropped_note = 0
    n_dropped_age = 0
    for hist_path in _iter_history_files(args.runs_root):
        if since_ts is not None:
            try:
                if hist_path.stat().st_mtime < since_ts:
                    n_dropped_age += 1
                    continue
            except OSError:
                continue
        n_files += 1
        traj_id = str(hist_path.parent.relative_to(args.runs_root))
        try:
            with hist_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        step = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    r = _row_from_step(traj_id, i, step)
                    if r is None:
                        continue
                    if args.hsv_only and r.get("note") != "hsv_measured":
                        n_dropped_note += 1
                        continue
                    rows.append(r)
        except OSError:
            continue
    if since_ts is not None:
        print(f"dropped {n_dropped_age} pre-{args.since} file(s)")
    if args.hsv_only:
        print(f"dropped {n_dropped_note} non-HSV row(s)")

    print(
        f"scanned {n_files} history.jsonl file(s), {n_lines} step "
        f"line(s) → {len(rows)} usable training row(s)"
    )
    if not rows:
        print(
            "no usable rows — run scripts/collect_pointer_accel.sh first.",
            file=sys.stderr,
        )
        return 1

    by_traj: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_traj[r["trajectory_id"]].append(r)
    traj_ids = sorted(by_traj.keys())
    rng = random.Random(args.seed)
    rng.shuffle(traj_ids)
    n = len(traj_ids)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val_ids = set(traj_ids[:n_val])
    test_ids = set(traj_ids[n_val: n_val + n_test])
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for tid, items in by_traj.items():
        if tid in val_ids:
            splits["val"].extend(items)
        elif tid in test_ids:
            splits["test"].extend(items)
        else:
            splits["train"].extend(items)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with (args.out / f"{name}.jsonl").open(
            "w", encoding="utf-8",
        ) as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "total_rows": len(rows),
        "total_trajectories": len(by_traj),
        "splits": {
            name: {
                "rows": len(items),
                "trajectories": len({
                    r["trajectory_id"] for r in items
                }),
            } for name, items in splits.items()
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"wrote {len(rows)} rows across {len(by_traj)} trajectories "
        f"→ {args.out}"
    )
    for name, items in splits.items():
        print(f"  {name:>5}: {len(items):>5d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
