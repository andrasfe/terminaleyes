#!/usr/bin/env python3
"""build_ml_dataset.py — turn terminaleyes run folders into BC dataset.

Walks ``runs_root`` (default ``~/.local/share/terminaleyes/runs``),
reads every ``<run_id>/steps.jsonl`` written by
:meth:`AgentContext.record_step`, resolves the ``frame_before_seq`` /
``frame_after_seq`` integers back to PNG paths, and emits JSONL
training rows in the schema documented in ``docs/ML_ROADMAP.md``.

Splits 80/10/10 train/val/test by *run* (so all steps of one run
land on the same side of the split — important for trajectory-
level eval). Stamps a small manifest with per-split counts and
per-agent class counts so we can spot imbalance early.

Usage::

    python scripts/build_ml_dataset.py \
        --runs-root ~/.local/share/terminaleyes/runs \
        --out       data/ml/dataset \
        --seed      0

No model dependencies — runs on any Python 3.11+.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _find_frame_path(run_dir: Path, seq: int | None) -> str | None:
    """Resolve a sequence number to the PNG path for that step.

    Filenames are ``NNNN_HHMMSS_<label>.png`` so we scan the run
    directory once per call. Returns a path relative to
    ``run_dir.parent`` (the runs root) so the dataset stays
    portable when the absolute path changes.
    """
    if seq is None:
        return None
    prefix = f"{seq:04d}_"
    for p in run_dir.iterdir():
        if p.name.startswith(prefix) and p.suffix.lower() == ".png":
            return str(p.relative_to(run_dir.parent))
    return None


def _iter_run_steps(run_dir: Path):
    steps_path = run_dir / "steps.jsonl"
    if not steps_path.exists():
        return
    try:
        with steps_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def build_rows(runs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        run_id = run_dir.name
        for raw in _iter_run_steps(run_dir):
            frame_before = _find_frame_path(
                run_dir, raw.get("frame_before_seq"),
            )
            frame_after = _find_frame_path(
                run_dir, raw.get("frame_after_seq"),
            )
            rows.append({
                "trajectory_id": run_id,
                "step_idx": raw.get("step_idx"),
                "ts": raw.get("ts"),
                "intent": raw.get("intent", ""),
                "history": raw.get("history", []),
                "frame_before": frame_before,
                "frame_after": frame_after,
                "action": {
                    "agent": raw.get("agent", ""),
                    "kwargs": raw.get("kwargs", {}),
                },
                "outcome": raw.get("outcome", {}),
            })
    return rows


def split_by_run(
    rows: list[dict], *, seed: int = 0,
) -> dict[str, list[dict]]:
    """80/10/10 split keyed by trajectory_id so all steps of a run
    land in the same split."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["trajectory_id"]].append(r)
    run_ids = sorted(by_run.keys())
    rnd = random.Random(seed)
    rnd.shuffle(run_ids)
    n = len(run_ids)
    n_val = max(1, n // 10) if n >= 10 else 0
    n_test = max(1, n // 10) if n >= 10 else 0
    val_ids = set(run_ids[:n_val])
    test_ids = set(run_ids[n_val:n_val + n_test])
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rid, items in by_run.items():
        if rid in val_ids:
            out["val"].extend(items)
        elif rid in test_ids:
            out["test"].extend(items)
        else:
            out["train"].extend(items)
    return out


def write_split(out_dir: Path, split: str, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_manifest(out_dir: Path, splits: dict[str, list[dict]]) -> None:
    manifest = {"splits": {}, "agent_counts": {}, "total_rows": 0}
    overall_agents: Counter[str] = Counter()
    overall_runs = 0
    for name, rows in splits.items():
        ag = Counter(r["action"]["agent"] for r in rows if r["action"]["agent"])
        run_count = len({r["trajectory_id"] for r in rows})
        manifest["splits"][name] = {
            "rows": len(rows), "runs": run_count,
            "agents": dict(ag),
        }
        overall_agents.update(ag)
        overall_runs += run_count
        manifest["total_rows"] += len(rows)
    manifest["agent_counts"] = dict(overall_agents)
    manifest["total_runs"] = overall_runs
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/terminaleyes/runs",
        help="Directory containing per-run subdirs.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path("data/ml/dataset"),
        help="Output directory; train.jsonl / val.jsonl / test.jsonl "
             "+ manifest.json are written here.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.runs_root.exists():
        print(f"runs root not found: {args.runs_root}", file=sys.stderr)
        return 2

    rows = build_rows(args.runs_root)
    if not rows:
        print(
            f"no steps.jsonl rows found under {args.runs_root}. "
            "Run a few intents first so the per-step logger writes "
            "rows.",
            file=sys.stderr,
        )
        return 1

    splits = split_by_run(rows, seed=args.seed)
    for name, items in splits.items():
        write_split(args.out, name, items)
    write_manifest(args.out, splits)
    print(
        f"wrote {sum(len(v) for v in splits.values())} rows "
        f"across {len({r['trajectory_id'] for r in rows})} runs to "
        f"{args.out}"
    )
    for name, items in splits.items():
        print(
            f"  {name:>5}: {len(items):5d} rows · "
            f"{len({r['trajectory_id'] for r in items}):3d} runs"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
