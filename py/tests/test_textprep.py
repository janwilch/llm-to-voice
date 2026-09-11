"""textprep strips <think> blocks and nothing else, correctly, on a stream."""

from __future__ import annotations

import pytest

from llmvoice.textprep import TextNormalizer, normalize_text


def feed_in_pieces(text: str, size: int) -> str:
    n = TextNormalizer()
    out = [n.feed(text[i : i + size]) for i in range(0, len(text), size)]
    out.append(n.flush())
    return "".join(out).strip()


class TestThinkBlocks:
    def test_removed(self):
        raw = "<think>I should be helpful here.</think>The answer is four."
        assert normalize_text(raw) == "The answer is four."

    def test_multiple_blocks_mid_text(self):
        assert normalize_text("<think>a</think>One. <think>b</think>Two.") == "One. Two."

    def test_unterminated_block_is_dropped(self):
        # A truncated response must not leak reasoning into the audio.
        assert normalize_text("Hello. <think>secret reasoning") == "Hello."


class TestStreaming:
    """The actual difficulty: markers arriving split across deltas."""

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 100])
    def test_chunk_size_does_not_change_the_result(self, size):
        raw = "<think>reasoning here</think>Here is the answer. It uses `code`.\nDone."
        assert feed_in_pieces(raw, size) == normalize_text(raw)

    @pytest.mark.parametrize("size", [1, 2, 3, 4])
    def test_tags_split_across_chunks(self, size):
        raw = "A.<think>drop me</think>B.<think>and me</think>C."
        assert feed_in_pieces(raw, size) == "A.B.C."

    def test_partial_tag_is_held_not_emitted(self):
        n = TextNormalizer()
        # "<thi" could still become "<think>", so it must not be spoken yet.
        assert "<thi" not in n.feed("Hello. <thi")
        assert n.feed("nk>drop me</think> Bye.").strip().endswith("Bye.")

    def test_held_partial_tag_is_released_on_flush(self):
        # "<thi" that never completes is real text and must survive.
        n = TextNormalizer()
        assert n.feed("Compare a <thi") + n.flush() == "Compare a <thi"

    def test_empty_input(self):
        assert normalize_text("") == ""
