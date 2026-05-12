#!/usr/bin/env python3
"""eval_ml_planner.py — offline replay-style eval of a trained planner.

Loads a LoRA adapter (produced by ``train_ml_planner.py``), runs one
forward pass per row in a held-out split, and reports:

  * top-1 next-action accuracy (agent name match)
  * exact-match accuracy (agent + kwargs structurally identical)
  * per-agent breakdown (precision-ish: how often the model emits
    each agent, and how often that emission is correct)

The model is fed the same prompt the trainer saw, with the same
``frame_before`` image. No physical loop, no Pi, no target machine —
this is purely offline. Use it after training to decide whether the
adapter is good enough to wire into the controller via
:class:`MlPlannerAgent`.

Usage::

    python scripts/eval_ml_planner.py \\
        --dataset    data/ml/dataset \\
        --split      val \\
        --runs-root  ~/.local/share/terminaleyes/runs \\
        --adapter    data/ml/checkpoints/uitars-7b-lora-v1 \\
        --limit      200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _kwargs_equal(a: dict, b: dict) -> bool:
    """Structural equality with light tolerance: missing keys treated
    as missing on the other side too (so model and label agree on
    optional kwargs)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a == b
    return a == b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=Path("data/ml/dataset"))
    ap.add_argument(
        "--split", choices=("train", "val", "test"), default="val",
    )
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/terminaleyes/runs",
    )
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows evaluated (0 = all).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON path for the per-row report.")
    args = ap.parse_args()

    rows_path = args.dataset / f"{args.split}.jsonl"
    if not rows_path.exists():
        print(f"split not found: {rows_path}", file=sys.stderr)
        return 2
    rows = _load_jsonl(rows_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"no rows in {rows_path}", file=sys.stderr)
        return 1

    # Lazy imports — same reason as the trainer.
    try:
        from PIL import Image
        from terminaleyes.ml.format import format_prompt, parse_response
        from terminaleyes.agents.ml_planner import _get_loader, _warp_if_needed
    except Exception as e:
        print("ML deps not installed: " + str(e), file=sys.stderr)
        return 2

    loader = _get_loader(args.adapter)
    loader.load()

    n_total = 0
    n_agent_match = 0
    n_exact = 0
    per_agent_pred: Counter[str] = Counter()
    per_agent_label: Counter[str] = Counter()
    per_agent_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    per_row: list[dict] = []

    for row in rows:
        n_total += 1
        frame_rel = row.get("frame_before")
        if not frame_rel:
            continue
        frame_path = args.runs_root / frame_rel
        if not frame_path.exists():
            continue
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"image load failed for {frame_path}: {e}",
                  file=sys.stderr)
            continue
        if loader.warp_frames:
            try:
                from terminaleyes.commander.visual_servo_homer import (
                    warp_frame_to_screenshot,  # type: ignore[attr-defined]
                )
                img = warp_frame_to_screenshot(img)
            except Exception:
                pass

        prompt = format_prompt(
            intent=str(row.get("intent", "")),
            history=row.get("history") or [],
        )
        try:
            raw = loader.predict(prompt=prompt, image=img)
        except Exception as e:
            print(f"inference failed at row {n_total}: {e}",
                  file=sys.stderr)
            continue
        parsed = parse_response(raw) or {}
        pred_agent = str(parsed.get("agent", ""))
        pred_kwargs = parsed.get("kwargs") or {}

        gold_agent = str(row["action"].get("agent", ""))
        gold_kwargs = row["action"].get("kwargs") or {}

        per_agent_pred[pred_agent] += 1
        per_agent_label[gold_agent] += 1
        confusion[gold_agent][pred_agent] += 1

        agent_match = pred_agent == gold_agent
        kw_match = _kwargs_equal(pred_kwargs, gold_kwargs)
        if agent_match:
            n_agent_match += 1
            per_agent_correct[gold_agent] += 1
        if agent_match and kw_match:
            n_exact += 1

        per_row.append({
            "trajectory_id": row.get("trajectory_id"),
            "step_idx": row.get("step_idx"),
            "intent": row.get("intent"),
            "gold": {"agent": gold_agent, "kwargs": gold_kwargs},
            "pred": {"agent": pred_agent, "kwargs": pred_kwargs},
            "agent_match": agent_match,
            "kwargs_match": kw_match,
            "raw": raw,
        })

    if n_total == 0:
        print("no rows evaluated", file=sys.stderr)
        return 1

    print()
    print(f"split={args.split}  rows={n_total}")
    print(f"  top-1 agent accuracy : "
          f"{n_agent_match}/{n_total} "
          f"({100.0 * n_agent_match / n_total:.1f}%)")
    print(f"  exact (agent+kwargs) : "
          f"{n_exact}/{n_total} "
          f"({100.0 * n_exact / n_total:.1f}%)")
    print()
    print("per-agent breakdown (gold_count → correct/predicted):")
    for agent in sorted(per_agent_label):
        gold = per_agent_label[agent]
        correct = per_agent_correct[agent]
        pred = per_agent_pred[agent]
        rec = 100.0 * correct / gold if gold else 0.0
        prec = 100.0 * correct / pred if pred else 0.0
        print(
            f"  {agent:<14} gold={gold:>4d}  pred={pred:>4d}  "
            f"correct={correct:>4d}  recall={rec:5.1f}%  "
            f"precision={prec:5.1f}%"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({
                "summary": {
                    "rows": n_total,
                    "agent_match": n_agent_match,
                    "exact": n_exact,
                },
                "per_agent": {
                    a: {
                        "gold": per_agent_label[a],
                        "pred": per_agent_pred[a],
                        "correct": per_agent_correct[a],
                    } for a in per_agent_label
                },
                "confusion": {
                    g: dict(c) for g, c in confusion.items()
                },
                "rows": per_row,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nwrote per-row report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
