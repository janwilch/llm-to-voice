"""Qwen3-TTS VoiceDesign over faster-qwen3-tts (PyTorch/CUDA).

Why this runtime rather than the official package: the official ``qwen-tts``
release exposes no streaming API at all — its generate calls return fully
rendered waveforms, despite the model card advertising low-latency streaming.
The dual-track streaming capability is real in the architecture but unexposed in
the reference code, so a third-party runtime is required. ``faster-qwen3-tts``
exposes it as pull-based generators and adds CUDA-graph decode capture.

Optional keyword arguments (``seed``, ``non_streaming_mode``) are passed only if
the installed signature accepts them, so a version bump degrades rather than
crashes.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Iterator

import numpy as np

from ..config import TtsConfig
from ..paths import resolve as resolve_path
from ..style import Style

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "faster-qwen3-tts is not installed. Install the GPU extra:\n"
    "    uv sync --extra cuda\n"
    "On RTX 50-series (sm_120) torch must come from the cu128/cu130 index — "
    "pyproject.toml already pins that — and attn_implementation must stay "
    "'sdpa'; flash-attn 2.x does not build for sm_120.\n"
    "To develop without a GPU, run with --tts-backend fake."
)


class Qwen3TorchTts:
    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._lock = threading.Lock()  # one GPU consumer at a time
        self._cancel = threading.Event()
        self._stream_kwargs: dict[str, object] = {}

    @property
    def sample_rate(self) -> int:
        return self._cfg.sample_rate

    # --- lifecycle -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(_INSTALL_HINT) from exc

        source = self._resolve_source()
        log.info("loading Qwen3-TTS VoiceDesign from %s", source)
        kwargs = self._load_kwargs(FasterQwen3TTS.from_pretrained)
        self._model = FasterQwen3TTS.from_pretrained(source, **kwargs)
        self._stream_kwargs = self._optional_stream_kwargs()

    def _resolve_source(self) -> str:
        local = resolve_path(self._cfg.local_dir)
        if local.exists() and any(local.iterdir()):
            return str(local)
        log.info(
            "%s is empty; falling back to the hub id %s "
            "(run scripts/download_models.py to pre-fetch)",
            local,
            self._cfg.model_id,
        )
        return self._cfg.model_id

    def _load_kwargs(self, func) -> dict[str, object]:
        """Only pass load-time options the installed signature knows about."""
        wanted: dict[str, object] = {
            "device_map": self._cfg.device,
            "attn_implementation": self._cfg.attn_implementation,
        }
        try:
            import torch

            wanted["dtype"] = getattr(torch, self._cfg.dtype)
        except ImportError:  # pragma: no cover - torch ships with the extra
            pass
        params = inspect.signature(func).parameters
        accepted = {k: v for k, v in wanted.items() if k in params}
        # Older/newer releases spell these differently; try the aliases.
        if "dtype" in wanted and "dtype" not in accepted and "torch_dtype" in params:
            accepted["torch_dtype"] = wanted["dtype"]
        if "device_map" not in accepted and "device" in params:
            accepted["device"] = self._cfg.device
        for name in sorted(set(wanted) - set(accepted) - {"dtype"}):
            log.warning("FasterQwen3TTS.from_pretrained has no %r parameter", name)
        return accepted

    def _optional_stream_kwargs(self) -> dict[str, object]:
        func = self._model.generate_voice_design_streaming
        params = inspect.signature(func).parameters
        extra: dict[str, object] = {}
        if "non_streaming_mode" in params:
            # False => feed text progressively during decode.
            extra["non_streaming_mode"] = False
        else:
            log.info("installed runtime has no non_streaming_mode; using its default")
        return extra

    def warmup(self) -> None:
        self._ensure_loaded()
        style = Style(persona="A neutral test voice.", delivery="Speak plainly.", seed=self._cfg.seed)
        for _ in self.synthesize("Warm up.", style):
            pass

    # --- synthesis -------------------------------------------------------
    def synthesize(self, text: str, style: Style) -> Iterator[np.ndarray]:
        self._ensure_loaded()
        self._cancel.clear()
        with self._lock:
            self._seed(style.seed)
            stream = self._model.generate_voice_design_streaming(
                text=text,
                language=self._cfg.language,
                instruct=style.instruct,
                chunk_size=self._cfg.chunk_size,
                **self._stream_kwargs,
            )
            for item in stream:
                if self._cancel.is_set():
                    break
                chunk = _as_mono_float32(_unpack(item))
                if chunk.size:
                    yield chunk

    def _seed(self, seed: int) -> None:
        """Pin sampling so a fixed instruct yields a stable voice."""
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:  # pragma: no cover
            pass

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._model = None


def _unpack(item) -> object:
    """The runtime yields ``(audio_chunk, sample_rate, timing)``; tolerate plain arrays."""
    if isinstance(item, tuple):
        return item[0]
    return item


def _as_mono_float32(chunk) -> np.ndarray:
    if hasattr(chunk, "detach"):  # torch tensor
        chunk = chunk.detach().to("cpu").float().numpy()
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim > 1:
        # (channels, frames) or (frames, channels) -> mono
        array = array.mean(axis=0) if array.shape[0] < array.shape[-1] else array.mean(axis=-1)
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float32)
