# ML roadmap — learning the controller

This document sketches the design for replacing the hand-built
planner + agent registry with one learned vision-language-action
(VLA) model. The loop stays the same: act → observe → act. The
verifier and the typed agent boundaries stay until the learned
model proves out — they become the reward signal and the action
vocabulary.

---

## Goal

One model: `(webcam frame, free-form intent, short history) → next action`.

- **Action vocabulary** = the existing `REGISTRY` (~17 named agents
  with typed kwargs). Mid-level. Low enough to be closed-loop, high
  enough that the model doesn't need to learn HID byte streams from
  scratch.
- **Decision rate** = one model call per agent step (the same place
  the controller currently calls `_llm_plan_chunk` or rule
  matchers). Wall-clock unchanged from today's LLM-planner path.
- **Loop** = unchanged. The model emits an agent call; the existing
  agent executes it; the next frame goes back into the model.

---

## Data format

Each existing run already produces sequenced frames + journal
entries. We need one row per *step*:

```jsonl
{
  "trajectory_id": "run_5b50c0da09ef",
  "step_idx": 3,
  "frame": "0003_191230_homer_capture.png",       // input
  "frame_after": "0004_191232_keys_alt+F4.png",   // for world-model option
  "intent": "unlock the screen",
  "history": [                                    // prior actions this run
    {"agent": "wake",   "kwargs": {}},
    {"agent": "verify", "kwargs": {"question": "..."}}
  ],
  "action": {                                     // label
    "agent": "login",
    "kwargs": {"vault_name": "desktop"}
  },
  "outcome": {"success": true, "reason": "login submitted"},
  "session": {"platform": "linux", "vault": "desktop"}
}
```

### Data sources (no new collection needed yet)

- `~/.local/share/terminaleyes/runs/<run_id>/` — frames already on disk.
- cc `RunRecord` (plan + status + reason) — gives the action
  sequence and outcome.
- `journal.md` — natural-language summary per run, useful as an
  auxiliary text label.

### What we'd need to add

- A **decision-time logger** in `AgentContext` that emits
  `(frame_id, agent_name, kwargs, outcome)` rows atomically, instead
  of relying on filename heuristics to recover the link between a
  frame and the next action.
- A **dataset builder** that joins the per-step log with run
  records and produces the JSONL schema above with 80/10/10 splits.

---

## Training objective

**Phase 1 — behavioural cloning.** Standard VLA cross-entropy on
the action token sequence, conditioned on `<image><intent><history>`.
Action emitted as a small JSON string the tokeniser handles
natively (`{"agent": "login", "kwargs": {"vault_name": "desktop"}}`).
Loss weighted by `outcome.success` (failed steps contribute less,
refused steps zero).

**Phase 2 — preference / outcome reweighting.** Use the verifier's
final verdict to label whole trajectories good/bad; reweight or
filter at the trajectory level. Equivalent to RFT on a self-
collected reward.

**Phase 3 (optional) — RL fine-tune.** Verifier as reward,
GRPO/PPO-style. Slow because each rollout is a real physical loop,
so reserve this for a model that already mostly works.

---

## Open-weight starting points

| Model                       | Why it's interesting here                                                                                                                  | Where it hurts                                                                                                                                       |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **UI-TARS-7B-DPO** (primary)| Current open-weight SOTA on GUI control benchmarks (ByteDance). Action vocabulary (`click`, `type`, `hotkey`, `scroll`, `wait`) maps 1:1.  | Trained on pixel-perfect screenshots, not webcams. Needs the homography pre-warp (see below) or fine-tuning on webcam frames.                        |
| **UI-TARS-2B**              | Same family, small enough for hourly iteration on a single 24 GB GPU. Easy step-up to 7B once the data pipeline is proven.                 | Lower ceiling. Use as a fast-iteration twin to the 7B, not the production choice.                                                                    |
| **Qwen2.5-VL-7B** (backup)  | Strong general VL backbone, great at JSON-shaped outputs. Less brittle on out-of-distribution inputs because it wasn't screenshot-only.    | No GUI prior — needs more demonstrations than a GUI-pretrained model.                                                                                |
| **OS-Atlas-Base-7B**        | Action-grounded, predates UI-TARS. Reasonable second-place GUI model.                                                                      | Smaller community; UI-TARS subsumes most of its benchmark wins.                                                                                      |
| **ShowUI-2B** (already running) | Already loaded in the stack via llama.cpp on `:1235`. Bolt an action head on top of the existing grounder for the fastest iteration loop. | Grounding-only pretraining; ceiling is lower than dedicated VLAs. Best as scaffolding to validate the **dataset shape** before scaling up.           |

**Pick to start: UI-TARS-7B-DPO + LoRA + homography-warped frames.**
Action heads already match terminaleyes' agent vocabulary almost
1:1, and DPO has been done on UI-TARS already so the base behaviour
is sane. Keep a UI-TARS-2B (or ShowUI-2B) twin loaded for fast
iteration cycles; promote to 7B once dataset shape is proven.

Two training backends are provided:

* **Apple Silicon (default for this repo)** —
  `scripts/train_ml_planner_mlx.py` uses `mlx-vlm` (Apple
  ml-explore). Fits an M-series Mac's unified memory; verified
  end-to-end on M4 Max with `mlx-community/Qwen2-VL-2B-Instruct-4bit`
  (and `mlx-community/UI-TARS-7B-DPO-4bit` for the production
  target). Loss converges on a smoke dataset; LoRA adapters land
  as `adapters.safetensors` alongside a `terminaleyes_meta.json`
  the runtime reads at load time.
* **CUDA / Linux** — `scripts/train_ml_planner.py` uses `peft` +
  `bitsandbytes` (4-bit base, LoRA adapters). Same dataset format,
  same runtime adapter shape, just a different trainer.

`MlPlannerAgent` detects the backend via `terminaleyes_meta.json`
and dispatches inference accordingly (`mlx-vlm.load + generate`
or `transformers + peft`), so a checkpoint trained on either
backend works at inference on either platform that ships the
matching runtime.

### The webcam-OOD problem

Every GUI-pretrained model expects screenshots — pixel-perfect,
axis-aligned. The webcam introduces perspective, glare, bezels,
lens curvature, and small-text OCR fuzz. Two responses:

1. **Pre-warp the frame** (cheapest, do first). The existing
   visual-servo homer already estimates a perspective transform
   from frame edges; reusing it as a model pre-processor produces
   a "flat" screenshot-like image that the GUI-pretrained backbone
   can consume on-distribution.
2. **Fine-tune harder on raw webcam frames** (slower, more honest).
   Reserve for after BC plateaus on warped frames.

The roadmap assumes path (1) by default. Path (2) is a follow-up if
warped-frame BC plateaus below a useful action-accuracy floor.

---

## Evaluation

- **Offline replay**: top-1 next-action accuracy against held-out
  trajectories, broken down by intent class (open-app / navigate /
  lock-unlock / type / read).
- **Dry-run live eval**: feed cc with `dry_run=true`; compare the
  model's plan to the current rule+LLM planner's plan and to a
  human verdict.
- **Wall-clock eval** (later): success rate of full intents
  end-to-end against the existing 8/9 gauntlet.

---

## Where a world model fits

A world model becomes worth its weight when *imagining* an action's
outcome is cheaper than *taking* it. For terminaleyes:

- Webcam → frame is slow (~100 ms) and a wrong action can lock the
  user out — both nudge toward "look before you leap."
- A small latent dynamics model `f(z_t, a_t) → z_{t+1}` would let
  the VLA rank candidate actions before committing. DreamerV3
  shape.
- **Don't start here.** Start with BC. Add the world model only if
  BC plateaus and physical-loop rollout cost is the bottleneck.

---

## Live results so far

| Adapter | Base | Rank | Dropout | Iters | Train rows | Val rows | Top-1 agent | Exact (a+kw) |
|---|---|---|---|---|---|---|---|---|
| `qwen2vl-2b-v2`  | `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 16 | 0.05 | 500 | 46 | 5 | 5/5 (100%)  | 3/5 (60%) |
| `uitars-7b-v3`   | `mlx-community/UI-TARS-7B-DPO-4bit`        | 8  | 0.05 | 600 | 57 | 7 | 6/7 (85.7%) | 3/7 (42.9%) |
| `uitars-7b-v4`   | `mlx-community/UI-TARS-7B-DPO-4bit`        | 8  | 0.05 | 600 | 98 | 9 | **0/9** — babble |
| `uitars-7b-v5`   | `mlx-community/UI-TARS-7B-DPO-4bit`        | 16 | 0.10 | 800 | 68 | 9 | **0/9** — empty on long prompts |

### Lesson learned: UI-TARS-DPO is a poor SFT base for our format

Trained four adapters on the same pipeline, only one (Qwen2-VL-2B
base) produces parseable JSON on held-out evals. The three on
`UI-TARS-7B-DPO-4bit` all degraded as we added more data:

- **v3** (57 rows, rank 8) gave 6/7 top-1 agent accuracy on the
  smallest val split — the apparent best result.
- **v4** (same hyperparams, 98 rows) babbled `<SYSTEM>` fragments
  on every row. Hypothesis: the new `__EXEC_SCRIPT__` envelope
  intents leaked template-shaped text into the user prompt, and
  the LoRA's low rank couldn't separate "imitate prompt shape"
  from "emit JSON".
- **v5** (envelope rows filtered, rank 16, dropout 0.10, 800
  iters) went the other way — empty output on all val rows.
  Direct sanity check: the v5-adapter model produces *something*
  for simple short prompts ("What is 2+2?") but emits empty
  strings for our long `<SYSTEM>...</SYSTEM>` system prompt. The
  base UI-TARS-DPO model works fine for both shapes; only the
  LoRA-fine-tuned model breaks on our format.

The common factor: UI-TARS-DPO was DPO'd by ByteDance to respond
to a very specific (image, GUI instruction) → action-call format,
not our `<SYSTEM> / <USER>` template. Our SFT at LR 2e-5 only
*nudges* against that DPO prior; under prompt distribution shift
(our long format) the prior surfaces as degenerate tokens.

**Working baseline**: `qwen2vl-2b-v2`. Generic VL base with weak
priors → easy to override with small SFT.

### What we'd try next (un-touched)

- **Try `mlx-community/UI-TARS-7B-SFT-4bit`** (the non-DPO variant).
  Same backbone, GUI prior preserved, but without the DPO layer
  that's fighting our SFT.
- **Reshape the prompt** to match UI-TARS's expected format more
  closely — drop the `<SYSTEM>` envelope, embed the agent
  registry as a system message via `apply_chat_template`'s system
  role rather than as inline tags. Test whether v3-rerun on
  reshaped prompts climbs back.
- More targeted data collection. Once trajectories cross ~200, mine
  `(success, failure)` pairs from the same `(intent, frame)` and
  add an ORPO pass (`mlx_vlm.lora --train-mode orpo`).

## Minimum viable loop to start

1. **Decision-time logging hook** (1 day): emit
   `(frame_id, agent_name, kwargs, outcome)` rows atomically from
   `AgentContext.record_step()`; backfill old runs from
   `frames + RunRecord`.
2. **Dataset builder** (1 day): JSONL of the schema above; 80/10/10
   split; emits a manifest with counts per intent class.
3. **LoRA fine-tune of OS-Atlas-Base-7B** (~3 days incl. data
   plumbing): single GPU, ~1000 trajectories enough to start.
4. **`MlPlannerAgent`** (1 day): drop-in alternative to the existing
   planner in `controller.py`, gated behind a flag. Same
   `(intent) → plan` interface, internally one forward pass per
   step instead of an LLM JSON call.
5. **Replay eval harness** (1 day): top-1 + verifier-success on
   held-out runs.

That's roughly a 1–2 week sprint to get a baseline model serving
the controller. Steps 1 and 2 are infrastructure with no ML risk
and start producing the dataset the moment they land.
