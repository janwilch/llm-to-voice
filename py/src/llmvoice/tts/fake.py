"""Deterministic offline TTS stand-in.

Lets the whole pipeline — threading, backpressure, cancellation, metrics, audio
device — be exercised with no GPU and no model download. Pitch is derived from
the text so segment boundaries are audible when you actually listen to it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

import numpy as np

from ..config import TtsConfig
from ..style import Style

_CHARS_PER_SECOND = 14.0  # roughly natural speaking rate


class FakeTts:
    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg
        self._sample_rate = cfg.sample_rate
        # Match the real backend's chunk granularity: 12.5 codec frames/second.
        self._chunk_frames = max(1, int(self._sample_rate * cfg.chunk_size / 12.5))
        self._cancel = threading.Event()
        self._phase = 0.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def warmup(self) -> None:
        pass

    def synthesize(self, text: str, style: Style) -> Iterator[np.ndarray]:
        self._cancel.clear()
        total = max(self._chunk_frames, int(len(text) / _CHARS_PER_SECOND * self._sample_rate))
        # Deterministic given (text, style): same inputs, same waveform.
        seed = (hash((text, style.instruct, style.seed)) & 0xFFFF) / 0xFFFF
        freq = 110.0 + 110.0 * seed
        emitted = 0
        while emitted < total:
            if self._cancel.is_set():
                return
            n = min(self._chunk_frames, total - emitted)
            t = (np.arange(n, dtype=np.float32) + self._phase) / self._sample_rate
            chunk = 0.05 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
            # Taper the ends so chunk seams do not click.
            fade = min(64, n)
            chunk[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            self._phase += n
            emitted += n
            yield chunk

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        pass
