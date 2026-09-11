"""A named voice library: the savable half of a VoiceDesign character.

There are no per-voice weights to save. VoiceDesign does not fine-tune anything —
it *samples* a speaker at generation time from (instruct, seed, text), which is
why the same seed and text reproduce bit-identical audio while a different seed
gives a different person entirely. So a "voice" here is a recipe, not a model:
the ``persona`` half of the instruct string plus the sampling ``seed``.

That makes a character about 300 bytes. Twenty of them is a ~6 KB JSON file, with
no extra checkpoint and no extra VRAM. The tradeoff is honest: the recipe pins the
voice *exactly* only for identical text. Across new text the speaker is redrawn
near the same description, which is close but not immovable — pinning identity
outright would need the Base checkpoint and a cloned reference clip.

``delivery`` is deliberately *not* stored. It is the out-of-band channel that
moves per utterance; a voice is who is speaking, not how they feel right now.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

log = logging.getLogger(__name__)

_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class Voice:
    """One saved character. ``note`` is free-text for your own bookkeeping."""

    name: str
    persona: str
    seed: int
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "persona": self.persona, "seed": self.seed, "note": self.note}

    @classmethod
    def from_dict(cls, raw: object) -> Voice:
        if not isinstance(raw, dict):
            raise ValueError(f"expected a voice object, got {type(raw).__name__}")
        missing = [k for k in ("name", "persona", "seed") if k not in raw]
        if missing:
            raise ValueError(f"voice entry is missing {', '.join(missing)}")
        try:
            seed = int(raw["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"voice {raw['name']!r} has a non-integer seed") from exc
        return cls(
            name=str(raw["name"]),
            persona=str(raw["persona"]),
            seed=seed,
            note=str(raw.get("note", "")),
        )


class VoiceLibrary:
    """JSON-backed set of named voices, keyed by name.

    A missing file is an empty library rather than an error: the first
    ``/voice save`` creates it. A *malformed* file does raise, because silently
    starting empty would look like the characters had been lost.
    """

    def __init__(self, path: str | Path, voices: dict[str, Voice] | None = None) -> None:
        self.path = Path(path)
        self._voices: dict[str, Voice] = dict(voices or {})

    # --- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> VoiceLibrary:
        p = Path(path)
        if not p.exists():
            return cls(p)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p} is not valid JSON: {exc}") from exc
        entries = raw.get("voices", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError(f"{p}: expected a list of voices under 'voices'")
        voices: dict[str, Voice] = {}
        for entry in entries:
            voice = Voice.from_dict(entry)
            if voice.name in voices:
                raise ValueError(f"{p}: duplicate voice name {voice.name!r}")
            voices[voice.name] = voice
        return cls(p, voices)

    def save(self) -> None:
        """Write the library, atomically: a crash mid-write must not lose voices."""
        payload = {
            "version": _FORMAT_VERSION,
            "voices": [self._voices[n].to_dict() for n in sorted(self._voices)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # --- access ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._voices)

    def __contains__(self, name: object) -> bool:
        return name in self._voices

    def names(self) -> list[str]:
        return sorted(self._voices)

    def all(self) -> list[Voice]:
        return [self._voices[n] for n in self.names()]

    def get(self, name: str) -> Voice:
        try:
            return self._voices[name]
        except KeyError:
            known = ", ".join(self.names()) or "none saved yet"
            raise KeyError(f"no voice named {name!r} (have: {known})") from None

    def add(self, voice: Voice) -> None:
        """Add or replace. Replacing is the expected way to refine a character."""
        self._voices[voice.name] = voice

    def remove(self, name: str) -> Voice:
        voice = self.get(name)
        del self._voices[name]
        return voice

    def rename(self, old: str, new: str) -> Voice:
        voice = replace(self.remove(old), name=new)
        self.add(voice)
        return voice
