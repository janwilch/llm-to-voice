"""The voice library: round-tripping, failure modes, and applying a voice."""

from __future__ import annotations

import json

import pytest

from llmvoice.app import apply_voice
from llmvoice.config import StyleConfig
from llmvoice.style import StyleChannel
from llmvoice.voices import Voice, VoiceLibrary


def test_missing_file_is_an_empty_library(tmp_path):
    """A first run has no voices.json; that is not an error."""
    lib = VoiceLibrary.load(tmp_path / "voices.json")
    assert len(lib) == 0
    assert lib.names() == []


def test_round_trip(tmp_path):
    path = tmp_path / "voices.json"
    lib = VoiceLibrary.load(path)
    lib.add(Voice("captain", "A gruff old sailor.", 7, note="likes shouting"))
    lib.add(Voice("clerk", "A nervous young clerk.", 99))
    lib.save()

    again = VoiceLibrary.load(path)
    assert again.names() == ["captain", "clerk"]
    assert again.get("captain").seed == 7
    assert again.get("captain").note == "likes shouting"
    assert again.get("clerk").persona == "A nervous young clerk."


def test_save_is_sorted_and_versioned(tmp_path):
    path = tmp_path / "voices.json"
    lib = VoiceLibrary.load(path)
    for name in ("zeta", "alpha", "mid"):
        lib.add(Voice(name, f"persona {name}", 1))
    lib.save()
    raw = json.loads(path.read_text())
    assert raw["version"] == 1
    assert [v["name"] for v in raw["voices"]] == ["alpha", "mid", "zeta"]


def test_add_replaces_so_a_character_can_be_refined(tmp_path):
    lib = VoiceLibrary.load(tmp_path / "v.json")
    lib.add(Voice("captain", "first draft", 1))
    lib.add(Voice("captain", "second draft", 2))
    assert len(lib) == 1
    assert lib.get("captain").persona == "second draft"
    assert lib.get("captain").seed == 2


def test_get_unknown_lists_what_is_available(tmp_path):
    lib = VoiceLibrary.load(tmp_path / "v.json")
    lib.add(Voice("captain", "p", 1))
    with pytest.raises(KeyError, match="captain"):
        lib.get("nope")


def test_remove_and_rename(tmp_path):
    lib = VoiceLibrary.load(tmp_path / "v.json")
    lib.add(Voice("old", "p", 5))
    lib.rename("old", "new")
    assert lib.names() == ["new"]
    assert lib.get("new").seed == 5
    lib.remove("new")
    assert len(lib) == 0


def test_malformed_json_raises_rather_than_starting_empty(tmp_path):
    """Silently starting empty would look like the characters had been lost."""
    path = tmp_path / "voices.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        VoiceLibrary.load(path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"voices": [{"name": "a"}]}, "missing"),
        ({"voices": [{"name": "a", "persona": "p", "seed": "abc"}]}, "non-integer seed"),
        ({"voices": "nope"}, "expected a list"),
        ({"voices": [{"name": "a", "persona": "p", "seed": 1}] * 2}, "duplicate"),
    ],
)
def test_bad_entries_are_reported(tmp_path, payload, match):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        VoiceLibrary.load(path)


def test_save_creates_parent_directories(tmp_path):
    lib = VoiceLibrary.load(tmp_path / "nested" / "deeper" / "voices.json")
    lib.add(Voice("a", "p", 1))
    lib.save()
    assert lib.path.exists()


def test_save_leaves_no_temp_file(tmp_path):
    lib = VoiceLibrary.load(tmp_path / "voices.json")
    lib.add(Voice("a", "p", 1))
    lib.save()
    assert [p.name for p in tmp_path.iterdir()] == ["voices.json"]


def test_apply_voice_sets_persona_and_seed_but_not_delivery():
    """Delivery is the out-of-band channel: it belongs to the moment, not the character."""
    style = StyleChannel(StyleConfig(), seed=1)
    style.set_delivery("whisper urgently")
    apply_voice(style, Voice("captain", "A gruff old sailor.", 42))

    snap = style.snapshot()
    assert snap.persona == "A gruff old sailor."
    assert snap.seed == 42
    assert snap.delivery == "whisper urgently"
    assert "gruff old sailor" in snap.instruct
    assert "whisper urgently" in snap.instruct
