"""Anchor for the paths in ``config.py``.

``models/`` lives at the repository root rather than inside ``py/`` because the
native port in ``cpp/`` loads the same GGUF and ONNX files. Resolving config
paths against the working directory therefore stopped being safe once the
Python project root moved to ``py/``: ``cd py && uv run`` and
``uv run --project py`` have different working directories, and the models are
in neither of them.

So relative config paths are resolved against the checkout root, found by
walking up from this file. Absolute paths are passed through untouched, which
is what ``--voices`` and a configured ``local_dir`` rely on.
"""

from __future__ import annotations

from pathlib import Path

# Both exist only at the checkout root, and never inside site-packages.
_ROOT_MARKERS = (".git", "cpp")


def repo_root() -> Path:
    """The checkout root, or the working directory if we are not in one."""
    for parent in Path(__file__).resolve().parents:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    # Installed as a wheel outside a checkout: the caller's CWD is the only
    # meaningful anchor left, which is the pre-``py/`` behaviour.
    return Path.cwd()


def resolve(path: str) -> Path:
    """Absolute path for a config value, anchored at the checkout root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root() / candidate
