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
from typing import Any, Dict, Iterable, Optional, Tuple

from PIL import Image

try:  # Optional MLX backend (Apple Silicon)
    from mlx_vlm import generate as mlx_generate
    from mlx_vlm import load as mlx_load
except ImportError:  # pragma: no cover - optional dependency
    mlx_generate = None
    mlx_load = None

try:  # Transformers backend dependencies
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:  # pragma: no cover - optional based on transformers version
        Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment]
except ImportError:  # pragma: no cover - handled per backend
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment]

from triage.io.structured import PredictionMeta, build_prediction
from triage.prompts import build_baseline_prompt

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("TRIAGE_MODEL_ID", "richardyoung/olmOCR-2-7B-1025-MLX-4bit")
DEFAULT_BACKEND = os.environ.get("TRIAGE_BACKEND", "auto")
DEFAULT_CALIBRATION_ID = "baseline-temp-v0"
DEFAULT_ADAPTER_ID = "global"
MAX_NEW_TOKENS = 512


@dataclass
class InferenceConfig:
    """Runtime configuration for the baseline predictor."""

    model_id: str = DEFAULT_MODEL_ID
    backend: str = DEFAULT_BACKEND
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
        self._backend_choice = (self.config.backend or "auto").lower()
        self._active_backend: Optional[str] = None
        self._device = torch.device("cpu") if torch is not None else "cpu"

    def predict(
        self,
        image: str | Path | Image.Image,
        *,
        extra_guidance: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Predict a label/confidence pair for a single page."""

        processor, model, backend = self._load_components()
        pil_image = self._to_image(image)
        prompt = build_baseline_prompt(extra_guidance=extra_guidance)

        if backend == "mlx":
            raw_text = _generate_with_mlx(
                model=model,
                processor=processor,
                image=pil_image,
                prompt=prompt,
                max_tokens=self.config.max_new_tokens,
            )
        else:
            inputs = self._prepare_inputs(processor, pil_image, prompt)
            inputs = _move_to_device(inputs, device=self._device)
            inputs = _ensure_float32(inputs)
            inputs = _force_fp32_pixel_values(inputs)
            pv = inputs.get("pixel_values")
            if torch is not None and isinstance(pv, torch.Tensor) and pv.dtype != torch.float32:
                inputs["pixel_values"] = pv.to(dtype=torch.float32)

            assert torch is not None  # safety: transformers backend implies torch import succeeded
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

    def _load_components(self) -> Tuple[Any, Any, str]:
        backend = self._resolve_backend()
        if backend == "mlx":
            if self._model is None or self._processor is None:
                LOGGER.info("Loading MLX baseline model '%s'", self.config.model_id)
                if mlx_load is None:
                    raise ImportError(
                        "mlx-vlm is required for backend='mlx'. Install via `pip install mlx-vlm`."
                    )
                model, processor = mlx_load(self.config.model_id)
                self._model = model
                self._processor = processor
                self._supports_chat_template = False
            return self._processor, self._model, backend

        # transformers backend
        if self._model is None or self._processor is None:
            if AutoProcessor is None or AutoModelForCausalLM is None or torch is None:
                raise ImportError(
                    "The transformers backend requires `torch` and `transformers`. "
                    "Install them via `pip install torch transformers` or set backend='mlx'."
                )
            model_id = self.config.model_id
            if model_id.endswith("-MLX-4bit"):
                LOGGER.warning(
                    "Model '%s' is MLX-specific. Falling back to base 'allenai/olmOCR-2-7B-1025' "
                    "for the transformers backend.",
                    model_id,
                )
                model_id = "allenai/olmOCR-2-7B-1025"
            LOGGER.info("Loading baseline model '%s' on %s", model_id, self._device)
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
            )
            _force_processor_fp32(processor)
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", None)
            self._supports_chat_template = hasattr(processor, "apply_chat_template")

            model_cls = AutoModelForCausalLM
            if Qwen2_5_VLForConditionalGeneration is not None:
                model_cls = Qwen2_5_VLForConditionalGeneration

            load_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            if torch.cuda.is_available():
                load_kwargs.update(
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                load_kwargs.update(attn_implementation="eager")

            try:
                model = model_cls.from_pretrained(
                    model_id,
                    **load_kwargs,
                ).eval()
            except (ValueError, OSError) as err:
                if model_cls is AutoModelForCausalLM and Qwen2_5_VLForConditionalGeneration is not None:
                    LOGGER.info("Retrying load with Qwen2_5_VLForConditionalGeneration due to: %s", err)
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id,
                        **load_kwargs,
                    )
                    if not torch.cuda.is_available():
                        model = model.to(self._device)
                    model = model.eval()
                else:
                    raise
            if not torch.cuda.is_available():
                model = model.to(self._device)
                model.float()
                _wrap_vision_tower_fp32(model)

            _configure_generation_padding(model, self._tokenizer)
            self._model = model.eval()
        return self._processor, self._model, backend

    def _resolve_backend(self) -> str:
        if self._active_backend:
            return self._active_backend

        choice = self._backend_choice
        if choice not in {"auto", "mlx", "transformers"}:
            raise ValueError(f"Unsupported backend '{choice}'. Expected auto|mlx|transformers.")

        if choice == "auto":
            backend = "mlx" if mlx_load is not None else "transformers"
        else:
            backend = choice

        if backend == "mlx" and mlx_load is None:
            raise ImportError(
                "Requested backend='mlx' but mlx-vlm is not installed. "
                "Install via `pip install mlx-vlm` or choose backend='transformers'."
            )
        if backend == "transformers" and (AutoProcessor is None or AutoModelForCausalLM is None or torch is None):
            raise ImportError(
                "Requested backend='transformers' but torch/transformers are unavailable. "
                "Install them or choose backend='mlx'."
            )

        self._active_backend = backend
        return backend

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


def _move_to_device(value, device: Any):
    if torch is None:
        return value
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
    if torch is None:
        return value
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

    if torch is None:
        return
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

    if torch is None or "pixel_values" not in inputs:
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

    if torch is None:
        return
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


def _generate_with_mlx(model, processor, image: Image.Image, prompt: str, max_tokens: int) -> str:
    if mlx_generate is None:
        raise ImportError(
            "mlx-vlm is required for MLX inference. Install via `pip install mlx-vlm`."
        )
    result = mlx_generate(model=model, processor=processor, image=image, prompt=prompt, max_tokens=max_tokens)
    if isinstance(result, dict):
        return result.get("text", "")
    return str(result)


# ------------------------------------------------------------------------- #
# CLI entrypoint
# ------------------------------------------------------------------------- #


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline VLM inference CLI")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model identifier.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["auto", "mlx", "transformers"],
        help="Inference backend selection.",
    )
    parser.add_argument("--calibration-id", default=DEFAULT_CALIBRATION_ID, help="Calibration metadata id.")
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID, help="Adapter identifier stored in metadata.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Generation token budget.")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = _build_cli()
    parsed = parser.parse_args(args=args)
    config = InferenceConfig(
        model_id=parsed.model_id,
        backend=parsed.backend,
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
