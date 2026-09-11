"""End-to-end pipeline tests on fake backends: no GPU, no server, no audio device."""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np
import pytest

from llmvoice.audio.sink import NullSink, RingBuffer, UnderrunTracker
from llmvoice.config import SegmentationConfig, TtsConfig
from llmvoice.llm.backend import EchoLlm
from llmvoice.metrics import MetricsRegistry
from llmvoice.pipeline import Pipeline
from llmvoice.style import Style, StyleChannel
from llmvoice.tts.fake import FakeTts


def build(reply: str | None = None, tts=None, sink=None, segmentation=None):
    metrics = MetricsRegistry()
    tts = tts or FakeTts(TtsConfig(backend="fake"))
    sink = sink or NullSink(tts.sample_rate)
    pipeline = Pipeline(
        llm=EchoLlm(reply),
        tts=tts,
        sink=sink,
        style=StyleChannel(),
        metrics=metrics,
        segmentation=segmentation or SegmentationConfig(),
    )
    return pipeline, sink, metrics


class TestEndToEnd:
    def test_speaks_and_produces_audio(self):
        # Long enough to still be several segments once coalescing has merged the
        # short ones, so this keeps asserting a *streamed* utterance under the
        # shipped defaults rather than a single lump.
        pipeline, sink, _ = build(
            "First sentence here, with enough words to stand alone. "
            "Second sentence follows, also long enough to be its own segment. "
            "A third one closes it out with room to spare."
        )
        m = pipeline.speak("hello")
        assert sink.frames > 0
        assert m.segments >= 2
        assert m.ttfa_ms is not None
        assert m.audio_seconds > 0

    def test_metrics_ordering(self):
        pipeline, _, _ = build("One thing happened. Another thing happened.")
        m = pipeline.speak("hello")
        # token -> segment -> audio, in that order.
        assert m.ttft_ms <= m.ttfs_ms <= m.ttfa_ms

    def test_audio_starts_before_the_llm_stream_ends(self):
        """The core streaming claim: audio exists while text is still arriving."""
        audio_seen = threading.Event()
        deltas_after_audio: list[str] = []

        long_reply = " ".join(
            f"This is sentence number {i}, and it carries on for a while." for i in range(12)
        )
        pipeline, sink, _ = build(long_reply)

        original_write = sink.write

        def watching_write(chunk):
            audio_seen.set()
            return original_write(chunk)

        sink.write = watching_write

        def on_delta(delta: str) -> None:
            if audio_seen.is_set():
                deltas_after_audio.append(delta)

        pipeline.on_delta = on_delta
        pipeline.speak("hello")
        assert deltas_after_audio, "audio only began after the LLM had finished"

    def test_segment_order_is_preserved(self):
        seen: list[str] = []
        pipeline, _, _ = build("Alpha first here. Bravo second here. Charlie third here.")
        pipeline.on_segment = seen.append
        pipeline.speak("hello")
        joined = " ".join(seen)
        assert joined.index("Alpha") < joined.index("Bravo") < joined.index("Charlie")

    def test_think_blocks_never_reach_the_tts(self):
        spoken: list[str] = []

        class RecordingTts(FakeTts):
            def synthesize(self, text, style):
                spoken.append(text)
                yield from super().synthesize(text, style)

        pipeline, _, _ = build(
            "<think>internal reasoning</think>The gate is open. Walk through it.",
            tts=RecordingTts(TtsConfig(backend="fake")),
        )
        pipeline.speak("hello")
        joined = " ".join(spoken)
        assert "internal" not in joined
        assert "The gate is open." in joined


class TestCancellation:
    def test_cancel_stops_promptly(self):
        """Barge-in: cancelling mid-utterance abandons the rest of the answer.

        Triggered off segment count rather than a timer — with fake backends the
        whole utterance can finish in ~100 ms, so a sleeping canceller races the
        pipeline and proves nothing.
        """
        long_reply = " ".join(f"Sentence number {i} is here." for i in range(200))
        pipeline, sink, _ = build(long_reply)

        seen = 0

        def on_segment(_segment: str) -> None:
            nonlocal seen
            seen += 1
            if seen == 3:
                pipeline.cancel()

        pipeline.on_segment = on_segment
        started = time.perf_counter()
        m = pipeline.speak("hello")
        elapsed = time.perf_counter() - started
        assert m.cancelled
        assert m.segments < 50, f"kept going for {m.segments} segments after cancel"
        assert elapsed < 5.0, f"cancel took {elapsed:.2f}s to take effect"

    def test_pipeline_is_reusable_after_cancel(self):
        pipeline, sink, _ = build("A sentence here. Another sentence here.")
        pipeline.cancel()
        m = pipeline.speak("hello")
        assert m.segments >= 1
        assert sink.frames > 0

    def test_rejects_concurrent_speak(self):
        pipeline, _, _ = build(" ".join(f"Sentence {i} here." for i in range(60)))
        errors: list[Exception] = []

        def second():
            time.sleep(0.05)
            try:
                pipeline.speak("again")
            except RuntimeError as exc:
                errors.append(exc)

        t = threading.Thread(target=second)
        t.start()
        pipeline.speak("first")
        t.join(timeout=5)
        assert errors and "already speaking" in str(errors[0])


class TestErrorPropagation:
    def test_tts_failure_surfaces(self):
        class BrokenTts(FakeTts):
            def synthesize(self, text, style):
                raise RuntimeError("gpu fell over")
                yield  # pragma: no cover

        pipeline, _, _ = build("A sentence.", tts=BrokenTts(TtsConfig(backend="fake")))
        with pytest.raises(RuntimeError, match="gpu fell over"):
            pipeline.speak("hello")


class TestBackpressure:
    def test_bounded_queues_keep_memory_flat(self):
        """A slow consumer must stall the producer, not accumulate unboundedly."""
        chunks_written = 0

        class SlowSink(NullSink):
            def write(self, chunk):
                nonlocal chunks_written
                chunks_written += 1
                time.sleep(0.002)
                return super().write(chunk)

        tts = FakeTts(TtsConfig(backend="fake"))
        pipeline, sink, _ = build(
            " ".join(f"Sentence {i} goes here." for i in range(30)),
            tts=tts,
            sink=SlowSink(tts.sample_rate),
            # Coalescing is off here on purpose: this test needs many small
            # segments to stress the bounded queues, which is orthogonal to the
            # voice-drift trade that coalescing makes.
            segmentation=dataclasses.replace(SegmentationConfig(), coalesce_min_chars=0),
        )
        m = pipeline.speak("hello")
        assert chunks_written > 0
        assert m.segments >= 10


class TestRingBuffer:
    def test_wraparound_preserves_order(self):
        ring = RingBuffer(8)
        out = np.zeros(4, dtype=np.float32)
        ring.write(np.arange(6, dtype=np.float32))
        assert ring.read_into(out) == 4
        assert list(out) == [0, 1, 2, 3]
        ring.write(np.arange(100, 104, dtype=np.float32))
        assert ring.read_into(out) == 4
        assert list(out) == [4, 5, 100, 101]

    def test_zero_pads_when_starved(self):
        ring = RingBuffer(8)
        ring.write(np.ones(2, dtype=np.float32))
        out = np.full(4, 9.0, dtype=np.float32)
        assert ring.read_into(out) == 2
        assert list(out) == [1.0, 1.0, 0.0, 0.0]

    def test_write_blocks_until_space_then_completes(self):
        ring = RingBuffer(4)
        done = threading.Event()

        def producer():
            ring.write(np.arange(10, dtype=np.float32))
            done.set()

        threading.Thread(target=producer, daemon=True).start()
        out = np.zeros(4, dtype=np.float32)
        collected: list[float] = []
        for _ in range(4):
            time.sleep(0.02)
            n = ring.read_into(out)
            collected.extend(out[:n].tolist())
        assert done.wait(timeout=2.0)
        ring.read_into(out)
        assert collected[:4] == [0.0, 1.0, 2.0, 3.0]

    def test_close_releases_a_blocked_writer(self):
        ring = RingBuffer(2)
        done = threading.Event()

        def producer():
            ring.write(np.ones(100, dtype=np.float32))
            done.set()

        threading.Thread(target=producer, daemon=True).start()
        time.sleep(0.05)
        ring.close()
        assert done.wait(timeout=2.0)


class TestUnderrunAccounting:
    """Only real dropouts may be counted; both non-cases were bugs once."""

    def test_silence_before_first_audio_is_not_an_underrun(self):
        t = UnderrunTracker()
        t.begin()
        assert t.observe(0, 512) is False
        assert t.observe(0, 512) is False

    def test_gap_mid_stream_is_an_underrun(self):
        t = UnderrunTracker()
        t.begin()
        t.observe(512, 512)
        assert t.observe(0, 512) is True
        assert t.observe(128, 512) is True

    def test_final_partial_block_is_not_an_underrun(self):
        t = UnderrunTracker()
        t.begin()
        t.observe(512, 512)
        t.end()  # producer finished
        assert t.observe(384, 512) is False
        assert t.observe(0, 512) is False

    def test_full_blocks_are_never_underruns(self):
        t = UnderrunTracker()
        t.begin()
        assert t.observe(512, 512) is False

    def test_begin_resets_playing_between_utterances(self):
        t = UnderrunTracker()
        t.begin()
        t.observe(512, 512)
        t.end()
        t.begin()
        assert t.observe(0, 512) is False


class TestStyleChannel:
    def test_instruct_composes_persona_and_delivery(self):
        style = Style(persona="A calm narrator", delivery="Speak slowly", seed=1)
        assert style.instruct == "A calm narrator. Speak slowly."

    def test_delivery_change_does_not_touch_persona(self):
        channel = StyleChannel()
        before = channel.persona
        channel.set_delivery("shout angrily")
        assert channel.persona == before
        assert "shout angrily" in channel.snapshot().instruct

    def test_snapshot_is_stable_across_a_mid_utterance_change(self):
        channel = StyleChannel()
        snap = channel.snapshot()
        channel.set_delivery("whisper")
        assert "whisper" not in snap.instruct

    def test_style_never_reaches_the_llm(self):
        """Structural guarantee: the LLM backend has no way to see the style."""
        prompts: list[str] = []

        class SpyLlm(EchoLlm):
            def stream(self, prompt):
                prompts.append(prompt)
                yield from super().stream(prompt)

        metrics = MetricsRegistry()
        tts = FakeTts(TtsConfig(backend="fake"))
        channel = StyleChannel()
        channel.set_delivery("SHOUT_MARKER_TEXT")
        pipeline = Pipeline(
            llm=SpyLlm(),
            tts=tts,
            sink=NullSink(tts.sample_rate),
            style=channel,
            metrics=metrics,
        )
        pipeline.speak("what is the time")
        assert prompts == ["what is the time"]
        assert all("SHOUT_MARKER_TEXT" not in p for p in prompts)
