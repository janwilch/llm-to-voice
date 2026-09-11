"""Incremental text -> speakable segments, via stream2sentence.

stream2sentence exists for exactly this job (real-time sentences from a
character/chunk feed, built for driving TTS from LLM output), so we do not
hand-roll a splitter. This module is only glue:

* forward our tuned defaults,
* filter kwargs against the installed signature, because these parameter names
  have drifted between releases (the README documents some that the shipped
  source does not have, and vice versa),
* degrade the nltk tokenizer to rule-based if the punkt data is unavailable,
  rather than dying on first run.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict

from stream2sentence import generate_sentences

from .config import SegmentationConfig

log = logging.getLogger(__name__)

_warned_unsupported: set[str] = set()
_tokenizer_checked = False


def _supported_kwargs(func, requested: dict[str, object]) -> dict[str, object]:
    params = inspect.signature(func).parameters
    accepted = {k: v for k, v in requested.items() if k in params}
    dropped = sorted(set(requested) - set(accepted))
    for name in dropped:
        if name not in _warned_unsupported:
            _warned_unsupported.add(name)
            log.warning(
                "stream2sentence %s has no parameter %r; ignoring it "
                "(installed version differs from the one this was tuned against)",
                getattr(func, "__name__", func),
                name,
            )
    return accepted


def _resolve_tokenizer(name: str) -> str:
    """Fall back to rule-based if the nltk punkt data is not present."""
    global _tokenizer_checked
    if "nltk" not in name:
        return name
    try:
        import nltk  # noqa: PLC0415  (optional dependency, checked lazily)

        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                return name
            except LookupError:
                continue
        for package in ("punkt_tab", "punkt"):
            if nltk.download(package, quiet=True):
                return name
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("nltk unavailable (%s)", exc)
    if not _tokenizer_checked:
        _tokenizer_checked = True
        log.warning(
            "nltk punkt data unavailable; falling back to the rule-based "
            "tokenizer. Abbreviations like 'Dr.' may split a sentence early."
        )
    return "rule-based"


def warmup(cfg: SegmentationConfig | None = None) -> None:
    """Pay the tokenizer's one-off load cost before the first real utterance.

    Loading punkt costs ~150 ms, which would otherwise be charged to the first
    measured TTFA and make the pipeline look slower than it is.
    """
    cfg = cfg or SegmentationConfig()
    tokenizer = _resolve_tokenizer(cfg.tokenizer)
    try:
        from stream2sentence import init_tokenizer

        init_tokenizer(tokenizer, cfg.language)
    except Exception as exc:  # pragma: no cover - best effort only
        log.debug("tokenizer warmup skipped: %s", exc)
    # Also exercise the generator once so any lazy imports are resolved.
    list(segment_stream(iter(["Warm up the tokenizer. "]), cfg))


def coalesce(
    segments: Iterable[str], min_chars: int, *, passthrough_first: bool = False
) -> Iterator[str]:
    """Merge consecutive segments until each is at least ``min_chars`` long.

    Each segment becomes an independent TTS generation, and VoiceDesign draws a
    fresh speaker every time, so a stream of short fragments is a stream of
    slightly different voices. Fewer, longer segments means fewer draws.

    ``passthrough_first`` emits the opening segment untouched, however short, to
    buy back the time-to-first-audio that buffering costs. It is off because that
    is the worst trade available here: the opener has the least context to fix a
    voice, so it is precisely the segment that comes out as somebody else, and the
    listener hears that switch as the utterance settles. Turn it on only to trade
    identity for latency deliberately.

    Lazy and order-preserving: it yields as soon as the threshold is met, so it
    costs latency only on segments that were too short to send anyway. Whatever
    is buffered at end-of-stream is always flushed, so no text is ever dropped.
    """
    if min_chars <= 0:
        yield from segments
        return
    remaining = iter(segments)
    if passthrough_first:
        for first in remaining:
            yield first
            break
    buffered: list[str] = []
    pending = 0
    for segment in remaining:
        buffered.append(segment)
        pending += len(segment)
        if pending >= min_chars:
            yield " ".join(buffered)
            buffered, pending = [], 0
    if buffered:
        yield " ".join(buffered)


def segment_stream(
    chunks: Iterable[str], cfg: SegmentationConfig | None = None
) -> Iterator[str]:
    """Yield speakable segments from a stream of text chunks.

    Blocking pull generator: it consumes ``chunks`` only as fast as the caller
    consumes segments, which is what gives the pipeline its backpressure.
    """
    cfg = cfg or SegmentationConfig()
    requested = asdict(cfg)
    requested["tokenizer"] = _resolve_tokenizer(cfg.tokenizer)
    # coalesce_min_chars is ours, not stream2sentence's; applied after it.
    requested.pop("coalesce_min_chars", None)
    kwargs = _supported_kwargs(generate_sentences, requested)
    stripped = (s.strip() for s in generate_sentences(chunks, **kwargs))
    yield from coalesce((s for s in stripped if s), cfg.coalesce_min_chars)
