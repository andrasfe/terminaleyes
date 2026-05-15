#!/usr/bin/env python3
"""train_pointer_accel.py — fit a tiny MLX MLP that maps HID deltas
to observed cursor deltas under Ubuntu's pointer-acceleration curve.

Forward model::

    f((hid_dx, hid_dy, cursor_x_pct, cursor_y_pct))
        → (measured_dx_pct, measured_dy_pct)

The inverse — "given a target pixel delta, what HID delta should I
send" — is the actually useful thing for the homer. We get it by
running Newton-style root-finding on the trained forward model
(see :class:`terminaleyes.commander.pointer_accel.PointerAccelModel`).

Why MLP and not a fitted closed-form curve: Ubuntu's libinput
"adaptive" profile is piecewise non-linear AND velocity-dependent
in subtle ways (acceleration scales with sqrt(dx²+dy²)). A 2-layer
MLP with ~50 hidden units fits it cleanly from a few hundred
samples; a closed-form fit would need careful per-axis parameter
search.

Inputs are 4-d: (hid_dx, hid_dy, cursor_x_pct, cursor_y_pct), all
normalised to roughly ``[-1, 1]``. HID values come in ``[-127, 127]``
so we divide by 127. Cursor positions are already in ``[0, 1]`` and
we shift to ``[-1, 1]``.

Outputs are 2-d: measured_dx_pct, measured_dy_pct (already in pct).

Usage::

    python scripts/train_pointer_accel.py \\
        --dataset data/ml/pointer_accel \\
        --output  data/ml/checkpoints/pointer_accel-v1 \\
        --epochs  400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _featurise(rows: list[dict]):
    import numpy as np
    X = []
    Y = []
    for r in rows:
        cx = r.get("cursor_x_pct")
        cy = r.get("cursor_y_pct")
        if cx is None or cy is None:
            # No prior cursor position — drop. With enough rows we
            # don't need the row-level fallback; the corpus is
            # dominated by rows that DO have a cursor position
            # because the homer only sends HID deltas AFTER it has
            # locked onto the cursor visually.
            continue
        X.append([
            r["hid_dx"] / 127.0,
            r["hid_dy"] / 127.0,
            (cx * 2.0) - 1.0,
            (cy * 2.0) - 1.0,
        ])
        Y.append([
            r["measured_dx_pct"],
            r["measured_dy_pct"],
        ])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dataset", type=Path,
        default=Path("data/ml/pointer_accel"),
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/ml/checkpoints/pointer_accel-v1"),
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    try:
        import numpy as np
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as e:
        print(f"missing deps: {e}", file=sys.stderr); return 2

    if not args.dataset.exists():
        print(f"dataset dir not found: {args.dataset}", file=sys.stderr)
        return 2
    train_rows = _load_jsonl(args.dataset / "train.jsonl")
    val_rows = (
        _load_jsonl(args.dataset / "val.jsonl")
        if (args.dataset / "val.jsonl").exists() else []
    )
    if not train_rows:
        print("no train rows", file=sys.stderr); return 1
    Xtr, Ytr = _featurise(train_rows)
    Xv, Yv = _featurise(val_rows) if val_rows else (None, None)
    print(
        f"train shape: {Xtr.shape} → {Ytr.shape};"
        f" val shape: {None if Xv is None else Xv.shape}"
    )

    class _MLP(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.fc1 = nn.Linear(4, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 2)

        def __call__(self, x):
            x = nn.gelu(self.fc1(x))
            x = nn.gelu(self.fc2(x))
            return self.fc3(x)

    model = _MLP(args.hidden)

    def loss_fn(model, x, y):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr)

    n = Xtr.shape[0]
    rng = np.random.default_rng(0)
    for epoch in range(1, args.epochs + 1):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            j = idx[start: start + args.batch_size]
            xb = mx.array(Xtr[j])
            yb = mx.array(Ytr[j])
            loss, grads = loss_and_grad(model, xb, yb)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            epoch_loss += float(loss)
            n_batches += 1
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            tr = epoch_loss / max(1, n_batches)
            line = f"  epoch {epoch:>4d}/{args.epochs}  train_mse={tr:.6f}"
            if Xv is not None and Xv.shape[0] > 0:
                vl = float(loss_fn(model, mx.array(Xv), mx.array(Yv)))
                line += f"  val_mse={vl:.6f}"
            print(line)

    # Persist weights as a tiny safetensors / numpy bundle. We use a
    # simple dict so the runtime wrapper doesn't need mlx_lm's
    # heavier checkpoint utilities.
    args.output.mkdir(parents=True, exist_ok=True)
    weights = {
        f"fc1.weight": np.array(model.fc1.weight),
        f"fc1.bias":   np.array(model.fc1.bias),
        f"fc2.weight": np.array(model.fc2.weight),
        f"fc2.bias":   np.array(model.fc2.bias),
        f"fc3.weight": np.array(model.fc3.weight),
        f"fc3.bias":   np.array(model.fc3.bias),
    }
    np.savez(str(args.output / "weights.npz"), **weights)
    (args.output / "config.json").write_text(json.dumps({
        "hidden": args.hidden,
        "input_features": [
            "hid_dx_norm", "hid_dy_norm",
            "cursor_x_centred", "cursor_y_centred",
        ],
        "output_features": ["measured_dx_pct", "measured_dy_pct"],
        "train_rows": int(Xtr.shape[0]),
        "val_rows": int(0 if Xv is None else Xv.shape[0]),
        "platform": "ubuntu-libinput-adaptive",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved → {args.output}/weights.npz + config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
