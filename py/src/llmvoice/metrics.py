"""Latency and health instrumentation.

The numbers that decide whether this works: time-to-first-token (LLM),
time-to-first-segment (segmenter), time-to-first-audio (end to end), real-time
factor (can the GPU keep up), and ring-buffer underruns (did the listener hear a
gap). Underruns are counted, never hidden by growing a buffer.

Aggregation across utterances lives in ``bench/latency.py``, not here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class UtteranceMetrics:
    submitted_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    first_segment_at: float | None = None
    first_audio_at: float | None = None

    segments: int = 0
    audio_frames: int = 0
    sample_rate: int = 24000
    synth_wall_s: float = 0.0
    cancelled: bool = False

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def mark_first_segment(self) -> None:
        if self.first_segment_at is None:
            self.first_segment_at = time.perf_counter()

    def mark_first_audio(self) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()

    # --- derived ---------------------------------------------------------
    def _delta_ms(self, at: float | None) -> float | None:
        return None if at is None else (at - self.submitted_at) * 1000.0

    @property
    def ttft_ms(self) -> float | None:
        """Time to first LLM token."""
        return self._delta_ms(self.first_token_at)

    @property
    def ttfs_ms(self) -> float | None:
        """Time to first speakable segment."""
        return self._delta_ms(self.first_segment_at)

    @property
    def ttfa_ms(self) -> float | None:
        """Time to first audio — the number that matters."""
        return self._delta_ms(self.first_audio_at)

    @property
    def audio_seconds(self) -> float:
        return self.audio_frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def rtf(self) -> float | None:
        """Synthesis wall time / audio produced. Must stay below 1.0."""
        audio = self.audio_seconds
        return (self.synth_wall_s / audio) if audio > 0 else None

    def summary(self) -> str:
        def fmt(value: float | None, unit: str, digits: int = 0) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}{unit}"

        parts = [
            f"ttft={fmt(self.ttft_ms, 'ms')}",
            f"ttfs={fmt(self.ttfs_ms, 'ms')}",
            f"ttfa={fmt(self.ttfa_ms, 'ms')}",
            f"audio={self.audio_seconds:.2f}s",
            f"rtf={fmt(self.rtf, '', 3)}",
            f"segments={self.segments}",
        ]
        if self.cancelled:
            parts.append("CANCELLED")
        return "  ".join(parts)


class MetricsRegistry:
    """Process-wide counters. Per-utterance timings live in UtteranceMetrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.underruns = 0

    def start_utterance(self, sample_rate: int) -> UtteranceMetrics:
        return UtteranceMetrics(sample_rate=sample_rate)

    def note_underrun(self, count: int = 1) -> None:
        with self._lock:
            self.underruns += count


def peak_vram_gb() -> float | None:
    """Torch peak reservation, if torch is loaded. None on the fake/CPU path."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 1024**3
