# Location: triage_infer.py
# Purpose: Provide a CPU-friendly baseline inference path using lightweight VLM prompts.
# Why: Phase 2 requires a prompted baseline with a vLLM-style interface and JSON outputs.
"""Baseline prompted inference harness for the lightweight VLM classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from PIL import Image

from huggingface_hub import hf_hub_download

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError as exc:  # pragma: no cover - surfaced during runtime configuration
    raise ImportError(
        "triage_infer requires the `transformers`, `torch`, and `huggingface_hub` packages. "
        "Install them via `pip install transformers torch huggingface_hub`."
    ) from exc

from triage.io.structured import PredictionMeta, build_prediction
from triage.prompts import build_baseline_prompt

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("TRIAGE_MODEL_ID", "richardyoung/olmOCR-2-7B-1025-MLX-4bit")
DEFAULT_CALIBRATION_ID = "baseline-temp-v0"
DEFAULT_ADAPTER_ID = "global"
MAX_NEW_TOKENS = 512


@dataclass
class InferenceConfig:
    """Runtime configuration for the baseline predictor."""

    model_id: str = DEFAULT_MODEL_ID
    calibration_id: str = DEFAULT_CALIBRATION_ID
    adapter_id: str = DEFAULT_ADAPTER_ID
    max_new_tokens: int = MAX_NEW_TOKENS


class BaselineInference:
    """CPU-oriented inference wrapper for the baseline vision-language model."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._supports_chat_template = False
        self._device = torch.device("cpu")

    def predict(
        self,
        image: str | Path | Image.Image,
        *,
        extra_guidance: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Predict a label/confidence pair for a single page."""

        processor, model = self._load_components()
        pil_image = self._to_image(image)
        prompt = build_baseline_prompt(extra_guidance=extra_guidance)

        inputs = self._prepare_inputs(processor, pil_image, prompt)
        inputs = _move_to_device(inputs, device=self._device)
        inputs = _ensure_float32(inputs)
        inputs = _force_fp32_pixel_values(inputs)
        pv = inputs.get("pixel_values")
        if isinstance(pv, torch.Tensor) and pv.dtype != torch.float32:
            inputs["pixel_values"] = pv.to(dtype=torch.float32)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.15,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated, strict=False)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_text = decoded[0] if decoded else ""

        label, confidence = self._parse_response(raw_text)
        meta = PredictionMeta(
            model_id=self.config.model_id,
            calibration_id=self.config.calibration_id,
            adapter_id=self.config.adapter_id,
        )
        return build_prediction(label=label, confidence=confidence, meta=meta)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _load_components(self):
        if self._model is None or self._processor is None:
            LOGGER.info("Loading baseline model '%s' on %s", self.config.model_id, self._device)
            processor = AutoProcessor.from_pretrained(
                self.config.model_id,
                trust_remote_code=True,
                use_fast=True,
            )
            _force_processor_fp32(processor)
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", None)
            self._supports_chat_template = hasattr(processor, "apply_chat_template")

            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                ).to(self._device)
                model.float()
                _wrap_vision_tower_fp32(model)

            _configure_generation_padding(model, self._tokenizer)
            self._model = model.eval()
        return self._processor, self._model

    def _prepare_inputs(self, processor, pil_image: Image.Image, prompt: str):
        """Prepare model inputs, honouring chat templates when available."""

        if self._supports_chat_template and hasattr(processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            return processor(
                text=[chat_prompt],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )

        return processor(
            images=[pil_image],
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )

    @staticmethod
    def _to_image(source: str | Path | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        return Image.open(path).convert("RGB")

    @staticmethod
    def _parse_response(raw_text: str) -> tuple[str, float]:
        """Extract label/confidence from the model response."""

        try:
            json_blob = _extract_json(raw_text)
            label = json_blob.get("label", "unknown")
            confidence = float(json_blob.get("confidence", 0.0))
        except ValueError:
            LOGGER.warning("Failed to parse model output; defaulting to unknown. Raw: %s", raw_text)
            return "unknown", 0.0

        # Basic sanitisation
        confidence = max(0.0, min(1.0, confidence))
        return str(label), confidence


def _extract_json(text: str) -> Dict[str, Any]:
    """Naively locate the first JSON object in the text response."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object detected.")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def _move_to_device(value, device: torch.device):
    if hasattr(value, "to") and not isinstance(value, torch.Tensor):
        try:
            return value.to(device=device)
        except TypeError:
            pass
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(v, device) for v in value)
    return value


def _ensure_float32(value):
    import numpy as np

    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32) if value.is_floating_point() else value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.float32)) if np.issubdtype(value.dtype, np.floating) else torch.from_numpy(value)
    if isinstance(value, dict):
        return {k: _ensure_float32(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_ensure_float32(v) for v in value)
    return value


def _force_processor_fp32(processor) -> None:
    """
    Ensure the underlying image processor emits float32 tensors.
    Recent Qwen2-VL fast processors default to bf16; override that knob.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    for attr in ("image_dtype", "dtype"):
        if hasattr(ip, attr):
            try:
                setattr(ip, attr, torch.float32)
            except Exception:
                pass
    if hasattr(ip, "to"):
        try:
            ip.to(dtype=torch.float32)
        except TypeError:
            pass


def _force_fp32_pixel_values(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively cast whatever sits under 'pixel_values' to contiguous float32."""

    if "pixel_values" not in inputs:
        return inputs

    def cast(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.to(dtype=torch.float32).contiguous()
        if isinstance(value, dict):
            return {k: cast(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cast(v) for v in value)
        return value

    inputs["pixel_values"] = cast(inputs["pixel_values"])
    return inputs


def _wrap_vision_tower_fp32(model) -> None:
    """Best-effort coercion of the vision tower to operate in float32 on CPU."""

    import types

    vt = getattr(model, "vision_tower", None)
    if vt is None:
        return

    try:
        vt.to(torch.float32)
    except Exception:
        pass
    if hasattr(vt, "dtype"):
        try:
            vt.dtype = torch.float32  # type: ignore[attr-defined]
        except Exception:
            pass

    original_forward = vt.forward

    def forward_fp32(self, *args, **kwargs):
        new_args = list(args)
        pixel = None
        if new_args:
            pixel = new_args[0]
        elif "pixel_values" in kwargs:
            pixel = kwargs["pixel_values"]
        if isinstance(pixel, torch.Tensor) and pixel.is_floating_point() and pixel.dtype != torch.float32:
            pixel = pixel.float().contiguous()
            if new_args:
                new_args[0] = pixel
            else:
                kwargs["pixel_values"] = pixel
        return original_forward(*new_args, **kwargs)

    vt.forward = types.MethodType(forward_fp32, vt)

    # Safety hook: ensure first conv sees fp32
    try:
        proj = vt.patch_embed.patchifier.proj  # nn.Conv2d

        def pre_hook(_module, inputs):
            if not inputs:
                return inputs
            (x, *rest) = inputs
            if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float32:
                x = x.float().contiguous()
            return (x, *rest)

        proj.register_forward_pre_hook(pre_hook)
    except Exception:
        pass


def _configure_generation_padding(model, tokenizer) -> None:
    """Ensure generation configs have sensible padding/eos identifiers."""

    if tokenizer is None:
        return

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if pad_token_id is None and eos_token_id is not None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.pad_token_id
        except Exception:
            pad_token_id = eos_token_id

    for cfg in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if cfg is None:
            continue
        if pad_token_id is not None and getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = pad_token_id
        if eos_token_id is not None and getattr(cfg, "eos_token_id", None) is None:
            cfg.eos_token_id = eos_token_id


def _load_chat_template(model_id: str) -> Optional[str]:
    try:
        path = hf_hub_download(model_id, "chat_template.json")
    except Exception:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("template")


# ------------------------------------------------------------------------- #
# CLI entrypoint
# ------------------------------------------------------------------------- #

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline VLM inference CLI")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model identifier.")
    parser.add_argument("--calibration-id", default=DEFAULT_CALIBRATION_ID, help="Calibration metadata id.")
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID, help="Adapter identifier stored in metadata.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Generation token budget.")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = _build_cli()
    parsed = parser.parse_args(args=args)
    config = InferenceConfig(
        model_id=parsed.model_id,
        calibration_id=parsed.calibration_id,
        adapter_id=parsed.adapter_id,
        max_new_tokens=parsed.max_new_tokens,
    )
    predictor = BaselineInference(config=config)
    result = predictor.predict(parsed.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
