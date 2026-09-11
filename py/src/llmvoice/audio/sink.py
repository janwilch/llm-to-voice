"""Audio output: a fixed-size ring buffer drained by the sound device callback.

The buffer never grows. If the pipeline cannot keep up, the callback outputs
silence and increments an underrun counter — an underrun is the signal that
something upstream is too slow, and hiding it behind an elastic buffer would
hide the only honest measure of whether this design works.

Underruns are only counted between ``begin_utterance`` and ``end_utterance``.
Without that bracketing the silence before the first chunk (which is TTFA, not a
dropout) and the silence after the last one would both be miscounted.

The ring buffer is also the part that maps most directly onto the native port:
``read_into`` is what ``llmvoice_poll_pcm`` becomes, and Unity drains it from
``OnAudioFilterRead``.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import numpy as np

from ..config import AudioConfig
from ..metrics import MetricsRegistry

log = logging.getLogger(__name__)


class AudioSink(Protocol):
    @property
    def sample_rate(self) -> int: ...
    def begin_utterance(self) -> None: ...
    def end_utterance(self) -> None: ...
    def write(self, chunk: np.ndarray) -> int: ...
    def drain(self, timeout_s: float | None = None) -> bool: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class RingBuffer:
    """Single-producer/single-consumer float32 ring buffer with blocking write."""

    def __init__(self, capacity_frames: int) -> None:
        self._buf = np.zeros(capacity_frames, dtype=np.float32)
        self._capacity = capacity_frames
        self._read = 0
        self._write = 0
        self._available = 0
        self._lock = threading.Lock()
        self._space = threading.Condition(self._lock)
        self._empty = threading.Condition(self._lock)
        self._closed = False

    @property
    def capacity(self) -> int:
        return self._capacity

    def available(self) -> int:
        with self._lock:
            return self._available

    def write(self, chunk: np.ndarray) -> int:
        """Block until the whole chunk is queued. Returns frames actually written."""
        written = 0
        with self._space:
            while written < chunk.size:
                if self._closed:
                    return written
                free = self._capacity - self._available
                if free == 0:
                    self._space.wait(timeout=0.5)
                    continue
                n = min(free, chunk.size - written)
                end = self._write + n
                if end <= self._capacity:
                    self._buf[self._write : end] = chunk[written : written + n]
                else:
                    split = self._capacity - self._write
                    self._buf[self._write :] = chunk[written : written + split]
                    self._buf[: n - split] = chunk[written + split : written + n]
                self._write = end % self._capacity
                self._available += n
                written += n
        return written

    def read_into(self, out: np.ndarray) -> int:
        """Fill ``out`` with up to ``out.size`` frames; zero-pad the remainder."""
        with self._lock:
            n = min(self._available, out.size)
            if n:
                end = self._read + n
                if end <= self._capacity:
                    out[:n] = self._buf[self._read : end]
                else:
                    split = self._capacity - self._read
                    out[:split] = self._buf[self._read :]
                    out[split:n] = self._buf[: n - split]
                self._read = end % self._capacity
                self._available -= n
                self._space.notify_all()
            if n < out.size:
                out[n:] = 0.0
            if self._available == 0:
                self._empty.notify_all()
        return n

    def wait_empty(self, timeout_s: float | None = None) -> bool:
        with self._empty:
            return self._empty.wait_for(
                lambda: self._available == 0 or self._closed, timeout_s
            )

    def clear(self) -> None:
        with self._lock:
            self._read = self._write = 0
            self._available = 0
            self._space.notify_all()
            self._empty.notify_all()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._space.notify_all()
            self._empty.notify_all()


class UnderrunTracker:
    """Decides whether a short read is a real dropout.

    Extracted from the device callback so the rule can be tested without a sound
    card. Two cases are *not* underruns and both were miscounted before this
    existed: silence before the first chunk arrives (that is time-to-first-audio)
    and the final partial block once the producer has finished.
    """

    def __init__(self) -> None:
        self._expecting = False
        self._playing = False

    def begin(self) -> None:
        self._playing = False
        self._expecting = True

    def end(self) -> None:
        """Producer finished; the tail is no longer our fault."""
        self._expecting = False

    def observe(self, got: int, frames: int) -> bool:
        """Record one callback. True if it counts as an underrun."""
        if got:
            self._playing = True
        elif not self._playing:
            return False
        return got < frames and self._expecting


class DeviceSink:
    """Plays PCM through the default (or configured) output device."""

    def __init__(self, cfg: AudioConfig, sample_rate: int, metrics: MetricsRegistry) -> None:
        import sounddevice as sd  # local import: headless CI need not have PortAudio

        self._metrics = metrics
        self._sample_rate = sample_rate
        self._ring = RingBuffer(max(1024, int(sample_rate * cfg.ring_seconds)))
        self._underruns = UnderrunTracker()
        self._started = False
        # Tallied in the realtime callback, reported from close().
        self.portaudio_status_flags = 0
        device = None if cfg.device < 0 else cfg.device
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=cfg.blocksize,
            device=device,
            callback=self._callback,
        )
        self._stream.start()
        self._started = True
        log.info(
            "audio out: device=%s %d Hz ring=%.2fs",
            self._stream.device,
            sample_rate,
            self._ring.capacity / sample_rate,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _callback(self, outdata, frames, time_info, status) -> None:
        # Realtime thread: no logging, no allocation, no blocking. Emitting a log
        # record from here can itself cause the underruns we are trying to count,
        # so PortAudio's own status flags are just tallied and read back later.
        if status:
            self.portaudio_status_flags += 1
        got = self._ring.read_into(outdata[:, 0])
        if self._underruns.observe(got, frames):
            self._metrics.note_underrun(1)

    def begin_utterance(self) -> None:
        self._underruns.begin()

    def end_utterance(self) -> None:
        """Producer is done.

        Call this *before* draining: once no more audio is coming, the final
        partial block is the natural end of the stream, not an underrun. Gaps
        that happen while the producer is still running are still counted.
        """
        self._underruns.end()

    def write(self, chunk: np.ndarray) -> int:
        return self._ring.write(chunk)

    def drain(self, timeout_s: float | None = None) -> bool:
        return self._ring.wait_empty(timeout_s)

    def flush(self) -> None:
        self._ring.clear()

    def close(self) -> None:
        self._underruns.end()
        if self.portaudio_status_flags:
            log.debug(
                "portaudio raised status flags on %d callbacks",
                self.portaudio_status_flags,
            )
        self._ring.close()
        if self._started:
            self._stream.stop()
            self._stream.close()
            self._started = False


class NullSink:
    """Discards audio but keeps accounting. For tests and --no-audio."""

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self.frames = 0
        self.chunks: list[np.ndarray] = []

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def begin_utterance(self) -> None:
        pass

    def end_utterance(self) -> None:
        pass

    def write(self, chunk: np.ndarray) -> int:
        self.frames += int(chunk.size)
        self.chunks.append(chunk)
        return int(chunk.size)

    def drain(self, timeout_s: float | None = None) -> bool:
        return True

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
