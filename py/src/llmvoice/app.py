"""Assembly: turn an AppConfig into a running Pipeline, and check the VRAM budget."""

from __future__ import annotations

import logging

from .audio.sink import DeviceSink, NullSink
from .config import AppConfig
from .llm.backend import EchoLlm
from .llm.llama_server import LlamaServerLlm
from .metrics import MetricsRegistry
from .paths import resolve as resolve_path
from .pipeline import Pipeline
from .segmentation import warmup as warmup_segmentation
from .style import StyleChannel
from .tts.backend import build_backend
from .voices import Voice, VoiceLibrary

log = logging.getLogger(__name__)


def apply_voice(style: StyleChannel, voice: Voice) -> None:
    """Make ``voice`` the active speaker.

    Persona and seed together are the voice; delivery is left alone because it
    is the out-of-band channel and belongs to the moment, not the character.
    """
    style.set_persona(voice.persona)
    style.set_seed(voice.seed)


class App:
    """Owns every resource for one session and tears them down in order."""

    def __init__(self, cfg: AppConfig, *, fake_llm: bool = False) -> None:
        self.cfg = cfg
        self.metrics = MetricsRegistry()
        self.style = StyleChannel(cfg.style, seed=cfg.tts.seed)
        # Saved characters. Loaded even when none is selected, so /voice save
        # during a session appends to the existing file instead of clobbering it.
        self.voices = VoiceLibrary.load(resolve_path(cfg.style.voices_path))
        if cfg.style.voice:
            apply_voice(self.style, self.voices.get(cfg.style.voice))

        check_vram_budget(cfg)
        warmup_segmentation(cfg.segmentation)

        self.llm = EchoLlm() if fake_llm else LlamaServerLlm(cfg.llm)
        self.tts = build_backend(cfg.tts)
        self.sink = (
            DeviceSink(cfg.audio, self.tts.sample_rate, self.metrics)
            if cfg.audio.enabled
            else NullSink(self.tts.sample_rate)
        )
        self.pipeline = Pipeline(
            llm=self.llm,
            tts=self.tts,
            sink=self.sink,
            style=self.style,
            metrics=self.metrics,
            segmentation=cfg.segmentation,
        )

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> App:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def check_vram_budget(cfg: AppConfig) -> None:
    """Refuse to start a configuration that would blow the VRAM cap.

    Advisory when torch is absent (the fake backend needs none), but it catches
    the obvious mistake of loading the 1.7B TTS model onto an already-busy card.
    """
    if not cfg.budget.enforce or cfg.tts.backend == "fake":
        return
    try:
        import torch
    except ImportError:
        log.info("torch not installed; skipping VRAM budget check")
        return
    if not torch.cuda.is_available():
        log.warning("no CUDA device visible; the qwen3_torch backend will fail")
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
    log.info(
        "VRAM: %.1f GB free of %.1f GB, cap %.1f GB",
        free_gb,
        total_gb,
        cfg.budget.vram_cap_gb,
    )
    if free_gb < 5.5:
        raise RuntimeError(
            f"only {free_gb:.1f} GB VRAM free; the 1.7B VoiceDesign model needs "
            f"~5 GB. Close other GPU work, or lower --n-gpu-layers on llama-server "
            f"so the LLM keeps less on the card."
        )
