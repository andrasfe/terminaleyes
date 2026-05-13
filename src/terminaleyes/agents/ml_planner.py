"""MlPlannerAgent — emit the next agent call from a fine-tuned VLA.

Drop-in replacement for the LLM-planner code path in
:class:`ControllerAgent`. Loads a LoRA adapter trained by
``scripts/train_ml_planner.py`` and runs one forward pass per step:

  (current frame, intent, history) → {"agent": "<name>", "kwargs": {...}}

The agent is invoked iteratively by the controller: it returns ONE
step, gets executed, the resulting frame goes back in, repeat until
the model emits a sentinel ``{"agent": "done"}`` or the step cap.

Inputs come from the same :class:`AgentContext` every other agent
uses: ``capture`` for the frame, ``vision_client`` is NOT used (we
go through HF transformers directly so the model can be hosted
locally without LM Studio in the loop).

This module imports torch/transformers/peft lazily — importing it
on a machine without those packages just disables ML planning
without breaking the rest of the agent stack.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from terminaleyes.agents.base import Agent, Outcome
from terminaleyes.ml.format import (
    format_history, format_prompt, parse_response,
)

logger = logging.getLogger(__name__)


@dataclass
class MlPlannerOutcome(Outcome):
    """``data['agent']`` and ``data['kwargs']`` carry the planned step."""


# Sentinel agent name used to signal "we're done — no more steps".
DONE_AGENT = "done"


class _LoadedModel:
    """Lazy holder for the LoRA-adapted base model + processor.

    Loaded once per process and reused across decisions. Two backend
    paths are supported via the ``backend`` field of
    ``terminaleyes_meta.json``:

      * ``"mlx"`` — Apple Silicon path via ``mlx-vlm``. Adapter is
        a ``adapters.safetensors`` file in the same dir.
      * ``"hf"`` (default) — CUDA + ``bitsandbytes`` + ``peft`` path
        for NVIDIA hosts. Same adapter dir.
    """

    def __init__(self, adapter_dir: Path) -> None:
        self.adapter_dir = adapter_dir
        self.processor: Any = None
        self.model: Any = None
        self.base_model_id: str = ""
        self.warp_frames: bool = False
        self.backend: str = "hf"
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        meta_path = self.adapter_dir / "terminaleyes_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"missing terminaleyes_meta.json in {self.adapter_dir} "
                "(was this adapter produced by a terminaleyes trainer?)"
            )
        meta = json.loads(meta_path.read_text("utf-8"))
        self.base_model_id = str(meta.get("base_model", ""))
        self.warp_frames = bool(meta.get("warp_frames", False))
        self.backend = str(meta.get("backend", "hf")).lower()
        if not self.base_model_id:
            raise ValueError("base_model missing from terminaleyes_meta.json")

        if self.backend == "mlx":
            self._load_mlx()
        else:
            self._load_hf()
        self._loaded = True

    # ── MLX backend (Apple Silicon, M-series) ─────────────────────
    def _load_mlx(self) -> None:
        from mlx_vlm import load as _mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template as _act  # noqa: F401
        # mlx-vlm.load(adapter_path=<dir>) looks for a peft-style
        # adapter_config.json next to the weights. The mlx_vlm.lora
        # trainer instead writes plain adapters.safetensors, so we
        # point load() directly at the .safetensors file when no
        # adapter_config.json exists.
        safetensors = self.adapter_dir / "adapters.safetensors"
        adapter_target: str
        if (self.adapter_dir / "adapter_config.json").exists():
            adapter_target = str(self.adapter_dir)
        elif safetensors.exists():
            adapter_target = str(safetensors)
        else:
            raise FileNotFoundError(
                f"no adapter weights in {self.adapter_dir} "
                "(expected adapters.safetensors or adapter_config.json)"
            )
        logger.info(
            "MlPlanner[mlx]: loading %s + adapter %s",
            self.base_model_id, adapter_target,
        )
        self.model, self.processor = _mlx_load(
            self.base_model_id, adapter_path=adapter_target,
        )

    def _predict_mlx(self, *, prompt: str, image: Any) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(
            self.processor, getattr(self.model, "config", None),
            prompt, num_images=1,
        )
        # Tiny temperature instead of pure greedy. Some 4-bit
        # quantised LoRA models (UI-TARS-7B-DPO-4bit observed) hit
        # degenerate token sequences at temp=0 that produce invalid
        # UTF-8, which mlx-vlm's detokenizer silently swallows ->
        # empty string output. A small temperature breaks the
        # degenerate path without meaningfully reducing accuracy
        # on instruction-following tasks.
        out = generate(
            self.model, self.processor,
            formatted,
            image=[image],
            max_tokens=200,
            temperature=0.1,
        )
        # mlx-vlm.generate returns either a string or a structured
        # GenerationResult depending on version. Normalise.
        return getattr(out, "text", out) if not isinstance(out, str) else out

    # ── HF / CUDA backend ─────────────────────────────────────────
    def _load_hf(self) -> None:
        import torch  # noqa: F401
        from transformers import (
            AutoProcessor,
            AutoModelForVision2Seq,
            BitsAndBytesConfig,
        )
        from peft import PeftModel

        logger.info(
            "MlPlanner[hf]: loading processor + 4-bit base %s + LoRA %s",
            self.base_model_id, self.adapter_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(self.adapter_dir), trust_remote_code=True,
        )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=__import__("torch").bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForVision2Seq.from_pretrained(
            self.base_model_id,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=__import__("torch").bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        self.model.eval()

    def _predict_hf(self, *, prompt: str, image: Any) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text, images=[image], return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=200,
                do_sample=False, temperature=0.0,
            )
        gen = output[0, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            gen.unsqueeze(0), skip_special_tokens=True,
        )[0]

    def predict(self, *, prompt: str, image: Any) -> str:
        if not self._loaded:
            self.load()
        if self.backend == "mlx":
            return self._predict_mlx(prompt=prompt, image=image)
        return self._predict_hf(prompt=prompt, image=image)


# Module-level singleton so successive MlPlannerAgent calls share
# the loaded weights. Keyed by adapter directory.
_LOADED: dict[str, _LoadedModel] = {}


def _get_loader(adapter_dir: Path) -> _LoadedModel:
    key = str(adapter_dir.resolve())
    if key not in _LOADED:
        _LOADED[key] = _LoadedModel(adapter_dir)
    return _LOADED[key]


def _warp_if_needed(frame_bgr: np.ndarray, *, warp: bool) -> "Any":
    """Convert a BGR ndarray (cv2 capture) → PIL image, optionally
    pre-warping via the homer's screen-flattening helper."""
    from PIL import Image
    rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
    img = Image.fromarray(rgb)
    if not warp:
        return img
    try:
        from terminaleyes.commander.visual_servo_homer import (
            warp_frame_to_screenshot,  # type: ignore[attr-defined]
        )
        return warp_frame_to_screenshot(img)
    except Exception:
        return img


class MlPlannerAgent(Agent):
    """Predict the next agent call from a frame + intent + history."""

    name = "ml_planner"

    async def run(
        self,
        *,
        intent: str,
        adapter_dir: str | Path,
        history: list[dict] | None = None,
        record_label: str = "ml_planner",
    ) -> MlPlannerOutcome:
        if self.ctx.capture is None:
            return MlPlannerOutcome(
                success=False,
                reason="no capture in context",
                data={"agent": "", "kwargs": {}},
            )
        adapter = Path(adapter_dir)
        if not adapter.exists():
            return MlPlannerOutcome(
                success=False,
                reason=f"adapter dir not found: {adapter}",
                data={"agent": "", "kwargs": {}},
            )
        try:
            frame = await self.ctx.capture.capture_frame()
        except Exception as e:
            return MlPlannerOutcome(
                success=False,
                reason=f"capture failed: {e}",
                data={"agent": "", "kwargs": {}},
            )
        self.ctx.record_frame(frame.image, label=record_label)

        prompt = format_prompt(intent=intent, history=history or [])
        loader = _get_loader(adapter)

        # The HF model runs synchronously; offload to a thread so we
        # don't block the controller's event loop.
        def _infer() -> str:
            try:
                loader.load()
            except Exception as e:
                raise RuntimeError(f"ml load failed: {e}") from e
            pil = _warp_if_needed(frame.image, warp=loader.warp_frames)
            return loader.predict(prompt=prompt, image=pil)

        try:
            raw = await asyncio.to_thread(_infer)
        except Exception as e:
            return MlPlannerOutcome(
                success=False,
                reason=f"inference failed: {e}",
                data={"agent": "", "kwargs": {}, "history": history or []},
            )

        parsed = parse_response(raw)
        if parsed is None:
            return MlPlannerOutcome(
                success=False,
                reason=f"unparseable response: {raw[:120]!r}",
                data={"agent": "", "kwargs": {}, "raw": raw},
            )
        agent = str(parsed.get("agent", "")).strip()
        kwargs = parsed.get("kwargs") or {}
        if not agent:
            return MlPlannerOutcome(
                success=False,
                reason="response missing 'agent' field",
                data={"raw": raw},
            )
        is_done = agent == DONE_AGENT
        return MlPlannerOutcome(
            success=True,
            reason=(
                "done" if is_done
                else f"emit step {agent}({kwargs})"
            ),
            data={
                "agent": agent,
                "kwargs": kwargs,
                "done": is_done,
                "raw": raw,
                "history": history or [],
            },
        )
