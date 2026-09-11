"""The out-of-band voice-style channel.

This is the whole point of the project: a control channel the application owns,
which steers emotion, volume and delivery *without* touching the words the LLM
produced. Qwen3-TTS takes it as an ``instruct`` string prepended to the model
input in ChatML form, so it is structurally separate from the text to be spoken.

The persona/delivery split is not cosmetic. In VoiceDesign mode the ``instruct``
string *is* the voice identity, so rewriting it wholesale between utterances can
drift the perceived speaker. Keeping ``persona`` byte-identical and pinning the
seed holds identity still while ``delivery`` moves.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import StyleConfig

# The model authors' guidance for VoiceDesign descriptions: a focused 30-80 words
# gives the model enough to hold a voice steady without contradicting itself.
_RECOMMENDED_MAX_WORDS = 80
_RECOMMENDED_MIN_WORDS = 8


@dataclass(frozen=True, slots=True)
class Style:
    """An immutable snapshot handed to the TTS backend for one utterance."""

    persona: str
    delivery: str
    seed: int

    @property
    def instruct(self) -> str:
        parts = [p.strip().rstrip(".") for p in (self.persona, self.delivery) if p.strip()]
        return ". ".join(parts) + "." if parts else ""

    def word_count(self) -> int:
        return len(self.instruct.split())



class StyleChannel:
    """Mutable, thread-safe holder for the current style.

    The CLI mutates it between utterances; the pipeline snapshots it once per
    utterance so a mid-utterance change cannot split a voice across segments.
    """

    def __init__(self, cfg: StyleConfig | None = None, seed: int = 1234) -> None:
        cfg = cfg or StyleConfig()
        self._lock = threading.Lock()
        self._persona = cfg.persona
        self._delivery = cfg.delivery
        self._seed = seed

    def snapshot(self) -> Style:
        with self._lock:
            return Style(self._persona, self._delivery, self._seed)

    @property
    def persona(self) -> str:
        with self._lock:
            return self._persona

    @property
    def delivery(self) -> str:
        with self._lock:
            return self._delivery

    @property
    def seed(self) -> int:
        with self._lock:
            return self._seed

    def set_delivery(self, delivery: str) -> None:
        with self._lock:
            self._delivery = delivery.strip()

    def set_persona(self, persona: str) -> None:
        """Changing this changes *who* is speaking, not just how. Rare by design."""
        with self._lock:
            self._persona = persona.strip()

    def set_seed(self, seed: int) -> None:
        with self._lock:
            self._seed = seed

    def warnings(self) -> list[str]:
        """Advisory checks on the composed instruct string."""
        style = self.snapshot()
        words = style.word_count()
        out: list[str] = []
        if words > _RECOMMENDED_MAX_WORDS:
            out.append(
                f"instruct is {words} words; Qwen recommends under "
                f"{_RECOMMENDED_MAX_WORDS} to avoid contradictory attributes"
            )
        if words < _RECOMMENDED_MIN_WORDS:
            out.append(
                f"instruct is only {words} words; short descriptions give the "
                "model little to hold the voice steady"
            )
        return out
