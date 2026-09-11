"""Fetch the TTS checkpoint and (optionally) an LLM GGUF, with a budget check.

    uv run python scripts/download_models.py            # TTS only
    uv run python scripts/download_models.py --llm 4b   # TTS + Qwen3-4B Q4_K_M

VoiceDesign only exists at 1.7B — the 0.6B releases ship Base and CustomVoice
only — so the TTS side of the budget is fixed at roughly 5 GB and the LLM is the
part that has to flex.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# This file is py/scripts/download_models.py, so the checkout root is three
# levels up. models/ sits there, not under py/, because cpp/ reads it too.
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

TTS_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TTS_APPROX_GB = 4.0

# Rough resident VRAM, weights plus working set, for the budget table.
LLM_CHOICES: dict[str, dict[str, object]] = {
    "4b": {
        "repo": "Qwen/Qwen3-4B-GGUF",
        "pattern": "*Q4_K_M.gguf",
        "vram_gb": 3.0,
        "note": "default: comfortable inside a 10 GB cap",
    },
    "8b": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "pattern": "*Q4_K_M.gguf",
        "vram_gb": 5.0,
        "note": "lands right at the 10 GB cap; expect no headroom",
    },
}
TTS_VRAM_GB = 5.0


def budget_table(llm: str | None, cap_gb: float) -> tuple[str, bool]:
    rows = [("Qwen3-TTS-12Hz-1.7B-VoiceDesign (bf16)", TTS_VRAM_GB)]
    total = TTS_VRAM_GB
    if llm:
        spec = LLM_CHOICES[llm]
        rows.append((f"{spec['repo']} Q4_K_M + 4k KV", float(spec["vram_gb"])))
        total += float(spec["vram_gb"])
    width = max(len(name) for name, _ in rows)
    lines = [f"  {name:<{width}}  {gb:>5.1f} GB" for name, gb in rows]
    lines.append(f"  {'total':<{width}}  {total:>5.1f} GB   (cap {cap_gb:.1f} GB)")
    return "\n".join(lines), total <= cap_gb


def free_disk_gb(path: Path) -> float:
    target = path
    while not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free / 1024**3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--llm", choices=sorted(LLM_CHOICES), help="also fetch an LLM GGUF")
    p.add_argument("--cap-gb", type=float, default=10.0, help="VRAM cap to check against")
    p.add_argument("--dest", type=Path, default=MODELS_DIR)
    p.add_argument("--skip-tts", action="store_true", help="only fetch the LLM")
    p.add_argument("--force", action="store_true", help="download even if over the cap")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="print the budget and disk check, then stop without downloading",
    )
    args = p.parse_args(argv)

    table, within = budget_table(args.llm, args.cap_gb)
    print("VRAM budget:")
    print(table)
    if not within:
        print("\nThis combination exceeds the cap.", file=sys.stderr)
        if not args.force:
            print("Pick a smaller LLM, raise --cap-gb, or pass --force.", file=sys.stderr)
            return 2
        print("Continuing anyway (--force).", file=sys.stderr)

    args.dest.mkdir(parents=True, exist_ok=True)
    needed = (0.0 if args.skip_tts else TTS_APPROX_GB) + (3.0 if args.llm else 0.0)
    free = free_disk_gb(args.dest)
    print(f"\nDisk: {free:.1f} GB free at {args.dest}, need roughly {needed:.1f} GB")
    if free < needed:
        print("Not enough free disk space.", file=sys.stderr)
        return 2

    if args.check_only:
        print("\n--check-only: nothing downloaded.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is missing. Run 'uv sync' first.",
            file=sys.stderr,
        )
        return 2

    if not args.skip_tts:
        target = args.dest / TTS_REPO.split("/")[-1]
        print(f"\nDownloading {TTS_REPO} -> {target}")
        snapshot_download(repo_id=TTS_REPO, local_dir=str(target))
        print(f"  done: {target}")

    if args.llm:
        spec = LLM_CHOICES[args.llm]
        target = args.dest / str(spec["repo"]).split("/")[-1]
        print(f"\nDownloading {spec['repo']} ({spec['pattern']}) -> {target}")
        snapshot_download(
            repo_id=str(spec["repo"]),
            local_dir=str(target),
            allow_patterns=[str(spec["pattern"])],
        )
        ggufs = sorted(target.rglob("*.gguf"))
        print(f"  done: {ggufs[0] if ggufs else target}")
        if ggufs:
            print("\nStart the LLM server with something like:")
            print(
                f"  llama-server --model {ggufs[0]} --ctx-size 4096 "
                f"--n-gpu-layers 999 --port 8080"
            )

    print("\nNext: uv run python -m llmvoice.cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
