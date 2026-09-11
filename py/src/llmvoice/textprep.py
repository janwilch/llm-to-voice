"""Strip ``<think>`` blocks from an LLM response. Nothing else.

Everything the model writes is forwarded verbatim — markdown punctuation, source
code, symbols, all of it. If the model chose to output it, it gets spoken.

An earlier version also stripped markdown and fenced code. Measurement killed
that off:

* markdown does not disturb segmentation — ``1. Open the door. 2. Walk inside.``
  stays a single segment, and emphasis, inline code, parentheses and decimals all
  pass through cleanly,
* stream2sentence already removes leading markers via
  ``filter_first_non_alnum_characters`` and URLs via ``cleanup_text_links``,
* and there is no evidence Qwen3-TTS is disturbed by any of it.

``<think>`` blocks are the sole exception, and a categorical one: they are not
speech at all, and leaking model reasoning into the audio is a real defect. They
are dropped even when a truncated response never closes the tag.

The only real difficulty is that this must work on a *stream*: ``<think>`` can
arrive split across two deltas. So we hold back the smallest tail that could
still become the marker and emit everything before it immediately.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

_OPEN = "<think>"
_CLOSE = "</think>"
_MAX_MARKER = max(len(_OPEN), len(_CLOSE))


def _held_prefix_len(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``marker``.

    That suffix cannot be emitted yet: another delta may complete it.
    """
    for size in range(min(len(text), _MAX_MARKER - 1), 0, -1):
        if marker.startswith(text[-size:]) and size < len(marker):
            return size
    return 0


class TextNormalizer:
    """Stateful, streaming ``<think>`` filter. One instance per utterance.

    ``feed`` may return "" while holding back a possible partial tag; ``flush``
    releases whatever is left.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        return self._process(final=False)

    def flush(self) -> str:
        out = self._process(final=True)
        tail, self._buf = self._buf, ""
        # An unterminated block stays dropped: truncated reasoning must not leak.
        if tail and not self._in_think:
            out += tail
        self._in_think = False
        return out

    def _process(self, *, final: bool) -> str:
        out: list[str] = []
        while self._buf:
            marker = _CLOSE if self._in_think else _OPEN
            idx = self._buf.find(marker)
            if idx < 0:
                keep = 0 if final else _held_prefix_len(self._buf, marker)
                if not self._in_think:
                    out.append(self._buf[: len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                break
            if not self._in_think:
                out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(marker) :]
            self._in_think = not self._in_think
        return "".join(out)


def normalize_text(text: str) -> str:
    """One-shot convenience wrapper, mostly for tests."""
    n = TextNormalizer()
    return (n.feed(text) + n.flush()).strip()


def normalize_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Wrap a delta iterator, yielding only non-empty pieces."""
    n = TextNormalizer()
    for chunk in chunks:
        piece = n.feed(chunk)
        if piece:
            yield piece
    piece = n.flush()
    if piece:
        yield piece
