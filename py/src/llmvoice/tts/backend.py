"""TTS backend contract.

The boundary is intentionally primitive — UTF-8 text plus an instruct string in,
float32 PCM frames out — because this is the seam that becomes the C ABI. No
Python objects cross it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np

from ..style import Style


@runtime_checkable
class TtsBackend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def warmup(self) -> None:
        """Pay one-off costs (weight load, CUDA graph capture) before timing."""
        ...

    def synthesize(self, text: str, style: Style) -> Iterator[np.ndarray]:
        """Yield mono float32 PCM chunks for ``text`` spoken in ``style``."""
        ...

    def cancel(self) -> None:
        """Ask an in-flight ``synthesize`` to stop promptly."""
        ...

    def close(self) -> None: ...


def build_backend(cfg) -> TtsBackend:
    """Factory. Imports lazily so the fake path needs no torch.

    The ggml/ONNX backend that becomes the Unity DLL slots in here; see
    docs/native-port.md.
    """
    if cfg.backend == "fake":
        from .fake import FakeTts

        return FakeTts(cfg)
    if cfg.backend == "qwen3_torch":
        from .qwen3_torch import Qwen3TorchTts

        return Qwen3TorchTts(cfg)
    raise ValueError(
        f"unknown tts.backend {cfg.backend!r} (expected 'qwen3_torch' or 'fake')"
    )
