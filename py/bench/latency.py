"""Latency/throughput sweep and acceptance check.

    uv run python bench/latency.py --tts-backend fake  # pipeline overhead only
    uv run python bench/latency.py                     # real LLM + real TTS
    uv run python bench/latency.py --chunk-sizes 2,4,8 # sweep TTS chunk size

Acceptance targets from the plan, on an RTX 5070 Ti:
    TTFA < 500 ms, RTF < 0.5, zero underruns, peak VRAM < 10 GB.

NOTE: the 500 ms target predates `segmentation.coalesce_min_chars`, which now
buys voice consistency with latency: TTFA includes waiting for ~60 characters of
LLM output before the first generation starts, instead of firing on a clause. At
30-60 tok/s that wait is roughly 250-500 ms on its own, so this target may no
longer be reachable with a real LLM and has not been re-validated since the
change. Measure before treating a failure here as a regression.

The pre-buffer and chunk-size knobs exist because the advertised 97 ms
first-packet latency is a lab number: at least one community streaming
implementation buffers ~38 tokens (~3 s) before emitting to keep the voice
stable. Measure, do not assume.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass

from llmvoice.app import App
from llmvoice.config import AppConfig
from llmvoice.metrics import peak_vram_gb

DEFAULT_PROMPTS = [
    "In two or three sentences, explain why the sky looks blue.",
    "Describe a thunderstorm rolling over a small coastal town.",
    "What is the difference between a sword and a spear in close combat?",
]

TARGET_TTFA_MS = 500.0
TARGET_RTF = 0.5


@dataclass
class Result:
    chunk_size: int
    ttfa_ms: list[float]
    rtf: list[float]
    underruns: int
    audio_s: float

    @property
    def median_ttfa(self) -> float:
        return statistics.median(self.ttfa_ms) if self.ttfa_ms else float("nan")

    @property
    def worst_ttfa(self) -> float:
        return max(self.ttfa_ms) if self.ttfa_ms else float("nan")

    @property
    def median_rtf(self) -> float:
        return statistics.median(self.rtf) if self.rtf else float("nan")


def run_one(cfg: AppConfig, prompts: list[str], fake_llm: bool) -> Result:
    with App(cfg, fake_llm=fake_llm) as app:
        app.tts.warmup()  # never time the weight load or CUDA graph capture
        before = app.metrics.underruns
        ttfa: list[float] = []
        rtf: list[float] = []
        audio = 0.0
        for prompt in prompts:
            m = app.pipeline.speak(prompt)
            if m.ttfa_ms is not None:
                ttfa.append(m.ttfa_ms)
            if m.rtf is not None:
                rtf.append(m.rtf)
            audio += m.audio_seconds
        return Result(
            chunk_size=cfg.tts.chunk_size,
            ttfa_ms=ttfa,
            rtf=rtf,
            underruns=app.metrics.underruns - before,
            audio_s=audio,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--tts-backend",
        choices=["qwen3_torch", "fake"],
        help="override tts.backend; 'fake' measures pipeline overhead only",
    )
    p.add_argument("--fake-llm", action="store_true", help="canned LLM replies")
    p.add_argument("--no-audio", action="store_true", help="no output device")
    p.add_argument(
        "--chunk-sizes",
        default="8",
        help="comma-separated tts.chunk_size values to sweep",
    )
    p.add_argument("--prompt", action="append", help="override the prompt set")
    args = p.parse_args(argv)

    prompts = args.prompt or DEFAULT_PROMPTS
    sizes = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]

    results: list[Result] = []
    for size in sizes:
        cfg = AppConfig()  # fresh per sweep step, so runs cannot bleed into each other
        if args.tts_backend:
            cfg.tts.backend = args.tts_backend
        if args.no_audio:
            cfg.audio.enabled = False
        cfg.tts.chunk_size = size
        print(f"\n=== chunk_size={size} ===")
        results.append(run_one(cfg, prompts, args.fake_llm))

    print("\n chunk  ttfa_med  ttfa_max   rtf_med  underruns   audio")
    for r in results:
        print(
            f" {r.chunk_size:>5}  {r.median_ttfa:>8.0f}  {r.worst_ttfa:>8.0f}  "
            f"{r.median_rtf:>8.3f}  {r.underruns:>9}  {r.audio_s:>6.1f}s"
        )

    vram = peak_vram_gb()
    print(f"\npeak VRAM: {'n/a' if vram is None else f'{vram:.2f} GB'}")

    best = min(results, key=lambda r: r.median_ttfa)
    failures: list[str] = []
    if best.median_ttfa > TARGET_TTFA_MS:
        failures.append(f"median TTFA {best.median_ttfa:.0f}ms > {TARGET_TTFA_MS:.0f}ms")
    if best.median_rtf > TARGET_RTF:
        failures.append(f"median RTF {best.median_rtf:.3f} > {TARGET_RTF}")
    if best.underruns:
        failures.append(f"{best.underruns} underruns")
    if vram is not None and vram > 10.0:
        failures.append(f"peak VRAM {vram:.2f} GB > 10 GB")

    if failures:
        print("\nFAIL (best chunk_size=%d):" % best.chunk_size, file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nPASS at chunk_size={best.chunk_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
