"""The streaming pipeline: prompt in, audio out, nothing buffered end-to-end.

Three stages, each on its own thread boundary, connected by *bounded* queues:

    LLM deltas ──text_q──> segmenter ──segment_q──> TTS ──ring buffer──> device
    (thread A)             (thread B)               (caller's thread)

Bounded is the whole point. When the device is behind, the ring buffer's blocking
write stalls the TTS stage, which stops draining ``segment_q``, which stalls the
segmenter, which stops draining ``text_q``, which stops consuming LLM deltas.
Backpressure propagates all the way to the token source and memory stays flat.

Each boundary carries only text or PCM frames — never Python objects — because
these are the seams that become ring buffers behind a C ABI.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator

from .audio.sink import AudioSink
from .config import SegmentationConfig
from .llm.backend import LlmBackend
from .metrics import MetricsRegistry, UtteranceMetrics
from .segmentation import segment_stream
from .style import Style, StyleChannel
from .textprep import TextNormalizer
from .tts.backend import TtsBackend

log = logging.getLogger(__name__)

_DONE = object()  # end-of-stream sentinel
_PUT_TIMEOUT = 0.1


class _Bridge:
    """Bounded queue with a sentinel and cancel-aware blocking operations."""

    def __init__(self, maxsize: int, cancel: threading.Event) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._cancel = cancel

    def put(self, item: object) -> bool:
        """Block until queued. False if cancelled before the item got in."""
        while not self._cancel.is_set():
            try:
                self._q.put(item, timeout=_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    def finish(self) -> None:
        """Queue the sentinel, blocking until there is room.

        This must block. A non-blocking put would silently drop the sentinel
        whenever the queue happened to be full at end-of-stream, and the
        consumer would then wait forever for an end that never came.
        """
        self.put(_DONE)

    def __iter__(self) -> Iterator[str]:
        while not self._cancel.is_set():
            try:
                item = self._q.get(timeout=_PUT_TIMEOUT)
            except queue.Empty:
                continue
            if item is _DONE:
                return
            yield item  # type: ignore[misc]

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return


class Pipeline:
    def __init__(
        self,
        llm: LlmBackend,
        tts: TtsBackend,
        sink: AudioSink,
        style: StyleChannel,
        metrics: MetricsRegistry,
        segmentation: SegmentationConfig | None = None,
        text_queue_size: int = 32,
        segment_queue_size: int = 8,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._sink = sink
        self._style = style
        self._metrics = metrics
        self._segmentation = segmentation or SegmentationConfig()
        self._text_queue_size = text_queue_size
        self._segment_queue_size = segment_queue_size
        self._cancel = threading.Event()
        self._speaking = threading.Lock()
        # Observers, for the CLI's live transcript and the streaming tests.
        self.on_delta: Callable[[str], None] | None = None
        self.on_segment: Callable[[str], None] | None = None

    # --- public API ------------------------------------------------------
    @property
    def metrics(self) -> MetricsRegistry:
        return self._metrics

    def speak(self, prompt: str) -> UtteranceMetrics:
        """Run one utterance to completion. Blocks until the audio has played."""
        if not self._speaking.acquire(blocking=False):
            raise RuntimeError("pipeline is already speaking")
        try:
            return self._run(prompt)
        finally:
            self._speaking.release()

    def say(self, text: str, style: Style | None = None) -> UtteranceMetrics:
        """Speak fixed text with no LLM, as one segment.

        This is the auditioning path: to compare voices you need the *same* words
        every time, and one generation rather than several, so that what differs
        between takes is only the persona and seed being judged.
        """
        if not self._speaking.acquire(blocking=False):
            raise RuntimeError("pipeline is already speaking")
        try:
            self._cancel.clear()
            m = self._metrics.start_utterance(self._tts.sample_rate)
            style = style or self._style.snapshot()
            self._sink.begin_utterance()
            try:
                m.mark_first_segment()
                m.segments += 1
                self._synthesize_segment(text, style, m)
            finally:
                self._sink.end_utterance()
                if not self._cancel.is_set():
                    self._sink.drain(timeout_s=m.audio_seconds + 2.0)
                else:
                    m.cancelled = True
            return m
        finally:
            self._speaking.release()

    def cancel(self) -> None:
        """Stop mid-utterance and drop queued audio (barge-in)."""
        self._cancel.set()
        self._llm.cancel()
        self._tts.cancel()
        self._sink.flush()

    def close(self) -> None:
        self.cancel()
        self._tts.close()
        self._llm.close()
        self._sink.close()

    # --- internals -------------------------------------------------------
    def _run(self, prompt: str) -> UtteranceMetrics:
        self._cancel.clear()
        m = self._metrics.start_utterance(self._tts.sample_rate)
        # One snapshot per utterance: a mid-utterance /style change must not
        # split one answer across two voices.
        style = self._style.snapshot()

        text_q = _Bridge(self._text_queue_size, self._cancel)
        segment_q = _Bridge(self._segment_queue_size, self._cancel)
        errors: list[BaseException] = []

        producer = threading.Thread(
            target=self._pump_llm,
            args=(prompt, text_q, m, errors),
            name="llm",
            daemon=True,
        )
        segmenter = threading.Thread(
            target=self._pump_segments,
            args=(text_q, segment_q, m, errors),
            name="segmenter",
            daemon=True,
        )

        self._sink.begin_utterance()
        producer.start()
        segmenter.start()
        try:
            self._pump_audio(segment_q, style, m, errors)
        finally:
            self._cancel.set()  # release any stage still blocked on a queue
            producer.join(timeout=5.0)
            segmenter.join(timeout=5.0)
            self._sink.end_utterance()
            m.cancelled = m.cancelled or bool(errors)

        if errors:
            raise errors[0]
        return m

    def _pump_llm(
        self, prompt: str, text_q: _Bridge, m: UtteranceMetrics, errors: list[BaseException]
    ) -> None:
        normalizer = TextNormalizer()
        try:
            for delta in self._llm.stream(prompt):
                if self._cancel.is_set():
                    break
                m.mark_first_token()
                if self.on_delta is not None:
                    self.on_delta(delta)
                piece = normalizer.feed(delta)
                if piece and not text_q.put(piece):
                    return
            tail = normalizer.flush()
            if tail:
                text_q.put(tail)
        except BaseException as exc:  # noqa: BLE001 - surfaced via speak()
            errors.append(exc)
        finally:
            text_q.finish()

    def _pump_segments(
        self,
        text_q: _Bridge,
        segment_q: _Bridge,
        m: UtteranceMetrics,
        errors: list[BaseException],
    ) -> None:
        try:
            # Blocking pull generator: it consumes text_q only as fast as the
            # TTS stage consumes segments.
            for segment in segment_stream(iter(text_q), self._segmentation):
                if self._cancel.is_set():
                    break
                m.mark_first_segment()
                m.segments += 1
                if self.on_segment is not None:
                    self.on_segment(segment)
                if not segment_q.put(segment):
                    return
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            segment_q.finish()

    def _pump_audio(
        self, segment_q: _Bridge, style, m: UtteranceMetrics, errors: list[BaseException]
    ) -> None:
        try:
            for segment in segment_q:
                if self._cancel.is_set():
                    break
                self._synthesize_segment(segment, style, m)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            # Stop counting underruns before draining: from here the tail is
            # the natural end of the stream, not the pipeline falling behind.
            self._sink.end_utterance()
            if not self._cancel.is_set():
                # Let the device finish what is already queued.
                self._sink.drain(timeout_s=m.audio_seconds + 2.0)
            else:
                m.cancelled = True

    def _synthesize_segment(self, segment: str, style, m: UtteranceMetrics) -> None:
        stream = self._tts.synthesize(segment, style)
        while True:
            # Time only the generator, not the blocking sink write, so RTF
            # measures synthesis throughput rather than playback pacing.
            t0 = time.perf_counter()
            try:
                chunk = next(stream)
            except StopIteration:
                m.synth_wall_s += time.perf_counter() - t0
                return
            m.synth_wall_s += time.perf_counter() - t0
            if self._cancel.is_set():
                return
            m.mark_first_audio()
            m.audio_frames += int(chunk.size)
            self._sink.write(chunk)
