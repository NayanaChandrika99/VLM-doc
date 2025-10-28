# Location: triage/serving/vllm_app.py
# Purpose: Offer a vLLM-like interface that forwards requests to the CPU baseline.
# Why: Phase 2 baseline needs to emulate vLLM behaviour without GPU dependencies.
"""CPU shim that mimics a subset of the vLLM API surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from triage_infer import BaselineInference, InferenceConfig


@dataclass
class VLLMRequest:
    """Minimal request object mirroring vLLM usage."""

    image: Path
    prompt: Optional[Sequence[str]] = None
    extra_guidance: Optional[Iterable[str]] = None


class CpuVLLMAdapter:
    """Drop-in replacement for vLLM-style generation on CPU."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self._predictor = BaselineInference(config=config)

    def generate(self, request: VLLMRequest) -> dict:
        """Process a single request and return structured prediction."""

        with Image.open(request.image) as img:
            pil_image = img.convert("RGB")
        return self._predictor.predict(pil_image, extra_guidance=request.extra_guidance)

    def generate_batch(self, requests: Sequence[VLLMRequest]) -> List[dict]:
        """Process a batch sequentially to mirror vLLM API expectations."""

        return [self.generate(req) for req in requests]


def create_default_engine() -> CpuVLLMAdapter:
    """Factory used by serving scripts."""

    return CpuVLLMAdapter()
