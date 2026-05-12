#!/usr/bin/env python3
"""train_ml_planner_mlx.py — LoRA fine-tune on Apple Silicon via MLX.

Native Apple-Silicon path (M-series unified memory + ``mlx-vlm``).
No CUDA / bitsandbytes / GPU box required. The CUDA-based
``train_ml_planner.py`` is the alternative when running on an NVIDIA
host.

Flow:
  1. Read the dataset JSONL produced by ``build_ml_dataset.py``.
  2. Re-shape each row into the ``messages`` schema mlx-vlm expects
     (system text + user text + image, then assistant target). Image
     paths become absolute so mlx-vlm's PIL loader can find them.
  3. Write a ``train.jsonl`` / ``valid.jsonl`` HF datasets-style file
     into ``--mlx-dataset-dir`` (mlx-vlm loads with
     ``datasets.load_dataset('json', data_files=...)``).
  4. Invoke ``mlx_vlm.lora`` as a subprocess with the model path,
     dataset dir, and LoRA hyperparameters. The adapter weights land
     in ``--output-path``.

Default model: ``mlx-community/UI-TARS-7B-DPO-4bit`` (Qwen2-VL-7B
backbone, pre-quantised). Fits well within an M4 Max's unified
memory; LoRA training adds ~1–3 GB on top of the base.

Usage::

    python scripts/train_ml_planner_mlx.py \\
        --dataset    data/ml/dataset \\
        --runs-root  ~/.local/share/terminaleyes/runs \\
        --model      mlx-community/UI-TARS-7B-DPO-4bit \\
        --output     data/ml/checkpoints/uitars-7b-mlx-lora-v1 \\
        --epochs     3
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


def _convert_split(
    rows: list[dict], *, runs_root: Path, system_text: str,
) -> list[dict]:
    """Reshape build_ml_dataset.py rows into mlx-vlm messages rows."""
    from terminaleyes.ml.format import format_prompt, format_response

    out: list[dict] = []
    for row in rows:
        frame_rel = row.get("frame_before")
        action = row.get("action") or {}
        agent = action.get("agent")
        if not frame_rel or not agent:
            continue
        abs_frame = (runs_root / frame_rel).resolve()
        if not abs_frame.exists():
            continue
        user_text = format_prompt(
            intent=str(row.get("intent", "")),
            history=row.get("history") or [],
        )
        assistant_text = format_response(
            agent=str(agent),
            kwargs=action.get("kwargs") or {},
        )
        out.append({
            "messages": [
                {"role": "system",
                 "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(abs_frame)},
                    {"type": "text", "text": user_text},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": assistant_text},
                ]},
            ],
            "images": [str(abs_frame)],
        })
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path,
        default=Path("data/ml/dataset"),
        help="Directory with train.jsonl / val.jsonl from build_ml_dataset.py",
    )
    ap.add_argument(
        "--runs-root", type=Path,
        default=Path.home() / ".local/share/terminaleyes/runs",
        help="Root of run dirs (image paths are resolved relative to this).",
    )
    ap.add_argument(
        "--model", type=str,
        default="mlx-community/UI-TARS-7B-DPO-4bit",
        help="MLX-quantised HF model id.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/run-mlx"),
        help="Where to write LoRA adapter weights.",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=0,
                    help="Override --epochs with a fixed iteration count.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="Gradient clipping norm. 0 disables.")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--image-resize", type=int, nargs=2, metavar=("W", "H"),
        default=[896, 504],
        help="Resize each input frame to (W, H) before tokenisation. "
             "Vision-token count grows quadratically — full 1920x1080 "
             "overruns the default sequence length on Qwen2-VL.",
    )
    ap.add_argument(
        "--max-seq-length", type=int, default=4096,
        help="Forwarded to mlx-vlm so vision + prompt tokens fit.",
    )
    ap.add_argument(
        "--mlx-dataset-dir", type=Path, default=None,
        help="Where to write the mlx-vlm-shaped dataset (default: "
             "<output>/.mlx-dataset).",
    )
    ap.add_argument(
        "--system-text", type=str,
        default=(
            "You are the terminaleyes controller. Emit the next "
            "agent call as JSON: {\"agent\": \"<name>\", \"kwargs\": "
            "{...}}."
        ),
        help="System message shown to the model on every sample.",
    )
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    train_rows = _load_jsonl(args.dataset / "train.jsonl")
    val_path = args.dataset / "val.jsonl"
    val_rows = _load_jsonl(val_path) if val_path.exists() else []
    if not train_rows:
        print("no rows in train.jsonl", file=sys.stderr)
        return 2

    # Lazy import — defer to after arg parsing so --help is fast.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    train_msgs = _convert_split(
        train_rows, runs_root=args.runs_root,
        system_text=args.system_text,
    )
    val_msgs = _convert_split(
        val_rows, runs_root=args.runs_root,
        system_text=args.system_text,
    ) if val_rows else []

    mlx_dir = args.mlx_dataset_dir or (args.output / ".mlx-dataset")
    mlx_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(mlx_dir / "train.jsonl", train_msgs)
    if val_msgs:
        _write_jsonl(mlx_dir / "valid.jsonl", val_msgs)
    print(
        f"converted {len(train_msgs)} train + {len(val_msgs)} valid "
        f"rows to {mlx_dir}"
    )

    # Persist a small meta file so MlPlannerAgent can recover the
    # base model id at load time without depending on mlx-vlm's
    # adapter-side serialisation.
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model": args.model,
        "backend": "mlx",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
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
        "--output-path", str(args.output),
    ]
    if args.iters > 0:
        cmd += ["--iters", str(args.iters)]
    else:
        cmd += ["--epochs", str(args.epochs)]
    if val_msgs:
        cmd += ["--steps-per-eval", "20"]
    print("running:", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    res = subprocess.run(cmd, env=env)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
