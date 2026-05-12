#!/usr/bin/env python3
"""train_ml_planner.py — LoRA fine-tune a VLA on terminaleyes traces.

Designed to run on a single 24 GB GPU. Uses 4-bit base weights via
``bitsandbytes`` and LoRA adapters via ``peft``. Tested target:
UI-TARS-7B-DPO (primary) and Qwen2.5-VL-7B-Instruct (backup) — both
HF-hosted, both Qwen2-VL style multimodal processors.

The training loop is intentionally short: ~3 epochs, paged AdamW,
cosine schedule. The expensive parts (data loading + multimodal
processor) match the model architecture; everything else is
boilerplate that can be tuned on a per-run basis.

This script does NOT execute training on the dev Mac (no CUDA).
Push it to a GPU box, install the deps below, run::

    pip install -U transformers accelerate peft bitsandbytes \\
                pillow datasets

    python scripts/train_ml_planner.py \\
        --dataset    data/ml/dataset \\
        --runs-root  ~/.local/share/terminaleyes/runs \\
        --model      bytedance-research/UI-TARS-7B-DPO \\
        --output     data/ml/checkpoints/uitars-7b-lora-v1 \\
        --epochs     3 \\
        --batch-size 1 \\
        --grad-accum 16

The output directory contains LoRA adapter weights consumable by
:class:`terminaleyes.agents.ml_planner.MlPlannerAgent`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# All ML deps are imported lazily inside main() so this file's
# top-level can be inspected on machines without torch installed.


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


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
        help="Root of run dirs (frame paths in the dataset are relative to this).",
    )
    ap.add_argument(
        "--model", type=str,
        default="bytedance-research/UI-TARS-7B-DPO",
        help="HF model id of the base VLA to fine-tune.",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/run"),
        help="Where to write LoRA adapter weights + tokenizer/processor.",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank. 8/16/32 are sensible.",
    )
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--warp-frames", action="store_true",
        help="Pre-warp each input frame via the homer's homography "
             "estimate before feeding to the model. Helps GUI-"
             "pretrained backbones (UI-TARS, OS-Atlas) handle the "
             "webcam-of-screen distribution shift. Slower data path.",
    )
    args = ap.parse_args()

    # ── deferred imports so the script's --help works without torch
    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoProcessor,
            AutoModelForVision2Seq,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as e:
        print(
            "ML deps not installed: " + str(e) + "\n"
            "  pip install -U transformers accelerate peft "
            "bitsandbytes pillow",
            file=sys.stderr,
        )
        return 2

    from terminaleyes.ml.format import format_sample

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    train_path = args.dataset / "train.jsonl"
    val_path = args.dataset / "val.jsonl"
    if not train_path.exists():
        print(f"missing {train_path}", file=sys.stderr)
        return 2

    train_rows = _load_jsonl(train_path)
    val_rows = _load_jsonl(val_path) if val_path.exists() else []
    print(f"loaded {len(train_rows)} train rows, {len(val_rows)} val rows")

    # Format rows → (prompt, response, image_path) triples.
    samples_train = [s for s in (format_sample(r) for r in train_rows) if s]
    samples_val = [s for s in (format_sample(r) for r in val_rows) if s]
    print(
        f"formatted: train={len(samples_train)} "
        f"val={len(samples_val)}"
    )
    if not samples_train:
        print("no usable training samples", file=sys.stderr)
        return 1

    # ── load processor + 4-bit base + attach LoRA
    print(f"loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"loading base model (4-bit): {args.model}")
    base = AutoModelForVision2Seq.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base = prepare_model_for_kbit_training(base)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            # Qwen2-VL / UI-TARS attention projections.
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    # ── lightweight in-memory dataset wrapper
    def _load_image(rel_path: str):
        full = args.runs_root / rel_path
        img = Image.open(full).convert("RGB")
        if args.warp_frames:
            try:
                from terminaleyes.commander.visual_servo_homer import (
                    warp_frame_to_screenshot,  # type: ignore[attr-defined]
                )
                img = warp_frame_to_screenshot(img)
            except Exception:
                # Warp helper may not be present in older builds —
                # silently fall through.
                pass
        return img

    def _encode(sample):
        image = _load_image(sample.image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": sample.prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample.response},
            ]},
        ]
        # Qwen2-VL processors accept image=... + text=... directly.
        text = processor.apply_chat_template(messages, tokenize=False)
        enc = processor(
            text=text, images=[image],
            return_tensors="pt", padding=False, truncation=True,
            max_length=args.max_seq_len,
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

    class _MlDS(torch.utils.data.Dataset):  # type: ignore[name-defined]
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return _encode(self.samples[i])

    train_ds = _MlDS(samples_train)
    val_ds = _MlDS(samples_val) if samples_val else None

    targs = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
    )
    trainer.train()

    # Persist the LoRA adapter + processor. MlPlannerAgent loads
    # both from this directory at inference time.
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))
    meta = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "train_rows": len(samples_train),
        "val_rows": len(samples_val),
        "warp_frames": bool(args.warp_frames),
    }
    (args.output / "terminaleyes_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    print(f"saved adapter + processor to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
