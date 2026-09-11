"""Segmentation tests.

These assert the behaviour *we* depend on, not stream2sentence's internals: that
speech can start on a clause rather than a full sentence, that the guards we
enabled actually hold, and that nothing is silently dropped.
"""

from __future__ import annotations

import dataclasses

from llmvoice.config import SegmentationConfig
from llmvoice.segmentation import coalesce, segment_stream


def chunks(text: str, size: int = 3):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def segments(text: str, cfg: SegmentationConfig | None = None, size: int = 3) -> list[str]:
    return list(segment_stream(chunks(text, size), cfg))


class TestCompleteness:
    def test_no_text_is_lost(self):
        text = "First sentence here. Second sentence follows. Third one ends it."
        joined = " ".join(segments(text))
        for word in ("First", "Second", "Third", "ends"):
            assert word in joined

    def test_trailing_fragment_is_flushed(self):
        # A stream that stops mid-sentence must still be spoken.
        out = segments("A complete sentence. And a dangling tail")
        assert "dangling tail" in " ".join(out)

    def test_single_short_input(self):
        assert " ".join(segments("Hi.")).strip() == "Hi."


class TestLatencyBehaviour:
    """The segmenter's own incremental behaviour, with coalescing out of the way.

    Coalescing deliberately trades these properties for voice consistency (see
    SegmentationConfig), so it is disabled here: what is under test is that
    stream2sentence can still yield early when asked, which is what any future
    latency/identity retuning depends on.
    """

    def test_first_fragment_arrives_before_the_full_sentence(self):
        cfg = SegmentationConfig(
            quick_yield_single_sentence_fragment=True,
            minimum_first_fragment_length=10,
            coalesce_min_chars=0,
        )
        text = (
            "Well, that depends entirely on how much time you have available "
            "before the deadline arrives."
        )
        out = segments(text, cfg)
        assert len(out) >= 2, "expected the clause to be yielded before the sentence end"
        assert len(out[0]) < len(text)

    def test_multiple_sentences_yield_multiple_segments(self):
        cfg = dataclasses.replace(SegmentationConfig(), coalesce_min_chars=0)
        out = segments("One thing happened here. Another thing happened later.", cfg)
        assert len(out) >= 2

    def test_the_default_config_merges_short_sentences_into_one_segment(self):
        """The shipped trade: two short sentences are one voice draw, not two."""
        out = segments("One thing happened here. Another thing happened later.")
        assert len(out) == 1


class TestGuards:
    def test_decimals_are_not_split(self):
        cfg = SegmentationConfig(never_split_numbers=True)
        out = segments("The value is 3.14 exactly.", cfg)
        assert any("3.14" in s for s in out), out

    def test_segments_are_stripped_and_non_empty(self):
        out = segments("One. Two. Three.")
        assert all(s == s.strip() and s for s in out)


class TestConfigTolerance:
    def test_unknown_parameters_do_not_raise(self, caplog):
        # The installed stream2sentence may not have every parameter we tuned
        # against; the wrapper must warn and continue, never crash.
        cfg = SegmentationConfig()
        object.__setattr__(cfg, "language", "en")
        out = segments("A sentence to segment.", cfg)
        assert out


# --- coalescing -----------------------------------------------------------
# Each segment is its own TTS generation and VoiceDesign redraws the speaker on
# every one, so short segments cost voice consistency. See SegmentationConfig.


def test_coalesce_merges_until_the_threshold():
    out = list(coalesce(["Hi!", "How are you?", "and you?", "I am well."], min_chars=20))
    assert out == ["Hi! How are you? and you?", "I am well."]


def test_coalesce_flushes_the_tail_so_no_text_is_lost():
    assert list(coalesce(["one", "two", "three"], min_chars=1000)) == ["one two three"]


def test_coalesce_disabled_passes_segments_through():
    segs = ["a", "bb", "ccc"]
    assert list(coalesce(segs, min_chars=0)) == segs
    assert list(coalesce(segs, min_chars=-1)) == segs


def test_coalesce_leaves_long_segments_alone():
    long_a, long_b = "x" * 80, "y" * 80
    assert list(coalesce([long_a, long_b], min_chars=60)) == [long_a, long_b]


def test_coalesce_merges_the_first_segment_like_any_other():
    """A short opener has the least context to fix a voice, so it is not special."""
    assert list(coalesce(["Hi!", "a" * 80], min_chars=60)) == ["Hi! " + "a" * 80]


def test_passthrough_first_opts_back_into_early_first_audio():
    """Available, but off: it reintroduces the audible opening-voice switch."""
    out = list(coalesce(["Hi!", "a" * 80], min_chars=60, passthrough_first=True))
    assert out == ["Hi!", "a" * 80]


def test_coalesce_is_lazy():
    """It must not drain the source to buffer: the pipeline needs backpressure."""
    consumed = []

    def source():
        for s in ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"]:
            consumed.append(s)
            yield s

    it = coalesce(source(), min_chars=5)
    assert next(it) == "aaaaaaaaaa"
    assert consumed == ["aaaaaaaaaa"]  # not the whole source


def test_coalesce_empty_source():
    assert list(coalesce([], min_chars=60)) == []


def test_segment_stream_applies_coalescing():
    text = " ".join(f"Sentence {i} goes here." for i in range(30))
    cfg = dataclasses.replace(SegmentationConfig(), coalesce_min_chars=0)
    apart = list(segment_stream(iter([text]), cfg))
    merged = list(segment_stream(iter([text]), dataclasses.replace(cfg, coalesce_min_chars=60)))

    assert len(merged) < len(apart)
    # Coalescing regroups words, it must not lose or reorder any.
    assert " ".join(merged).split() == " ".join(apart).split()
