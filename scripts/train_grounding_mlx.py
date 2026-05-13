#!/usr/bin/env python3
"""train_grounding_mlx.py — fine-tune a small VLM as a grounding head
on the dataset produced by ``scripts/build_grounding_dataset.py``.

Output target is a single `<point>x,y</point>` string in normalised
coords, so the per-sample loss is concentrated on a tiny number of
tokens — exactly the shape where mlx-vlm + LoRA on a 2B base
behaved cleanly in the planner experiment (v2). We reuse the
``mlx_vlm.lora`` CLI under the hood; the bulk of this script is
converting our JSONL into the messages schema mlx-vlm wants.

Default model: `mlx-community/Qwen2-VL-2B-Instruct-4bit`.
Override with ``--model`` to try ShowUI-2B or a 7B variant.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _convert_split(rows, *, runs_root: Path, system_text: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from terminaleyes.ml.grounding_format import format_sample

    out = []
    for row in rows:
        sample = format_sample(row)
        if sample is None:
            continue
        abs_frame = (runs_root / sample.image_path).resolve()
        if not abs_frame.exists():
            continue
        out.append({
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(abs_frame)},
                    {"type": "text", "text": sample.prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample.response},
                ]},
            ],
            "images": [str(abs_frame)],
        })
    return out


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path,
        default=Path("data/ml/grounding"),
        help="Directory with train.jsonl / val.jsonl from build_grounding_dataset.py",
    )
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/terminaleyes/runs",
        help="Root that image_path entries are relative to.",
    )
    ap.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/grounding-v1"),
    )
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--image-resize", type=int, nargs=2, metavar=("W", "H"),
        default=[448, 252],
    )
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument(
        "--train-cap", type=int, default=8000,
        help="Cap training rows to keep wall-clock reasonable. "
             "0 = use all rows.",
    )
    ap.add_argument(
        "--val-cap", type=int, default=500,
        help="Cap val rows during training-time eval.",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from terminaleyes.ml.grounding_format import SYSTEM_PROMPT

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    train_rows = _load_jsonl(args.dataset / "train.jsonl")
    val_path = args.dataset / "val.jsonl"
    val_rows = _load_jsonl(val_path) if val_path.exists() else []
    if args.train_cap and len(train_rows) > args.train_cap:
        import random
        random.Random(0).shuffle(train_rows)
        train_rows = train_rows[: args.train_cap]
    if args.val_cap and len(val_rows) > args.val_cap:
        import random
        random.Random(0).shuffle(val_rows)
        val_rows = val_rows[: args.val_cap]
    if not train_rows:
        print("no train rows", file=sys.stderr); return 2

    train_msgs = _convert_split(
        train_rows, runs_root=args.runs_root, system_text=SYSTEM_PROMPT,
    )
    val_msgs = _convert_split(
        val_rows, runs_root=args.runs_root, system_text=SYSTEM_PROMPT,
    ) if val_rows else []

    mlx_dir = args.output / ".mlx-dataset"
    _write_jsonl(mlx_dir / "train.jsonl", train_msgs)
    if val_msgs:
        _write_jsonl(mlx_dir / "valid.jsonl", val_msgs)
    print(
        f"converted train={len(train_msgs)} valid={len(val_msgs)} → {mlx_dir}"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model": args.model,
        "backend": "mlx",
        "task": "grounding",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "iters": args.iters,
        "train_rows": len(train_msgs),
        "val_rows": len(val_msgs),
    }
    (args.output / "terminaleyes_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", args.model,
        "--dataset", str(mlx_dir),
        "--split", "train",
        "--learning-rate", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--max-seq-length", str(args.max_seq_length),
        "--grad-clip", str(args.grad_clip),
        "--image-resize-shape", str(args.image_resize[0]),
        str(args.image_resize[1]),
        "--iters", str(args.iters),
        "--output-path", str(args.output),
    ]
    if val_msgs:
        cmd += ["--steps-per-eval", "100"]
    print("running:", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
