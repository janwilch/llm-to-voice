# llm-to-voice

Type text, hear the answer **as it is generated** — with emotion, volume and
delivery steered by a side channel that never becomes part of what the LLM says.

Everything runs locally. Total VRAM stays under 10 GB including KV cache.

```
typed text
    │
    ▼
llama.cpp  /v1/chat/completions (stream=true)      ← system prompt: WHAT is said
    │  token deltas
    ▼
textprep       strips <think> blocks; everything else forwarded verbatim
    ▼
stream2sentence  whole sentences, coalesced to ~60 chars (one voice draw each)
    │  segments                      ← style channel: HOW it is said (out-of-band)
    ▼
Qwen3-TTS VoiceDesign  instruct= persona + delivery, streaming
    │  24 kHz mono PCM chunks
    ▼
fixed ring buffer → audio device
```

## Status

The real TTS path is implemented but needs the model download and the
`cuda` extra. The ggml/ONNX backend — the one that becomes a Unity DLL — is not
written yet; [docs/native-port.md](docs/native-port.md) has the plan, the C ABI,
and the two alternatives that were rejected.

## Repository layout

```
py/       the Python prototype — this is what runs today
cpp/      the native port, empty until Stage 1 in docs/native-port.md passes
models/   shared by both; not in git
docs/
```

`py/` and `cpp/` are peers, and the prototype is not deleted by the port: it
stays as the voice-auditioning and tuning harness. **uv runs from `py/`**, and
paths in `config.py` resolve against the checkout root, so `models/` and
`voices.json` are the same files either way.

## Quick start

```bash
cd py
uv sync --extra dev

# No GPU, no server, no model: proves the pipeline streams.
uv run python -m llmvoice.cli --fake-llm --tts-backend fake --no-audio -p "hello there"

uv run pytest
```

For the real thing, still from `py/`:

```bash
uv sync --extra cuda --extra dev   # both: a bare --extra cuda drops pytest
uv run python scripts/download_models.py --llm 4b   # prints a VRAM budget table first
llama-server --model ../models/Qwen3-4B-GGUF/*Q4_K_M.gguf --ctx-size 4096 --n-gpu-layers 999 --port 8080
uv run python -m llmvoice.cli
```

Then, in the REPL, the test that matters:

```
> what happened at the bridge last night?
> /style shout angrily, much louder than normal
> what happened at the bridge last night?
```

Same words, different delivery, same voice. If that holds, the style channel is
genuinely out-of-band.

## GPU setup — read this first on an RTX 50-series card

The RTX 50-series is **sm_120**. Two things bite because of that, and one
is a `transformers` incompatibility that bites everywhere:

1. **Install torch *and torchaudio* from the cu128 or cu130 index.**
   `pyproject.toml` pins the cu128 index for both. A cu128 wheel will not load
   against a cu130 build or vice versa. See
   [pytorch/pytorch#164342](https://github.com/pytorch/pytorch/issues/164342).
   `torchaudio` matters because it arrives transitively via `faster-qwen3-tts`,
   and the plain PyPI wheel is built against CUDA 13: paired with a cu128 torch
   it fails at import with `OSError: libcudart.so.13: cannot open shared object
   file` out of `torchaudio/lib/_torchaudio.abi3.so`. It is listed explicitly in
   the `cuda` extra because `[tool.uv.sources]` only binds the project's own
   dependencies, not transitive ones.
2. **Do not use FlashAttention 2.** Building flash-attn 2.x with
   `FLASH_ATTN_CUDA_ARCHS=120` segfaults nvcc on the backward kernels, which
   blocks all RTX 50-series users
   ([#2361](https://github.com/Dao-AILab/flash-attention/issues/2361),
   [#2535](https://github.com/Dao-AILab/flash-attention/issues/2535),
   [#1987](https://github.com/Dao-AILab/flash-attention/issues/1987)).
   `tts.attn_implementation` defaults to `"sdpa"` for this reason — PyTorch SDPA
   with cuDNN flash reaches ~160 TFLOPS on a 5070 Ti, so nothing is lost.
3. **Do not raise `transformers` above 5.15.1.** `qwen-tts-hf` requires
   `>=5.15.1,<6`, but its rotary modules still want the
   `ROPE_INIT_FUNCTIONS["default"]` entry 5.x deleted, so it reinstalls one that
   reads the 4.x `config.rope_theta`. Through 5.15.1 that is harmless —
   `_init_weights` routes rope_type `"default"` to each model's own
   `compute_default_rope_parameters` and never consults the registry. 5.16+
   refactored that lookup to put `**ROPE_INIT_FUNCTIONS` *last*, so the
   reinstalled entry shadows every model's own default and loading the Mimi codec
   fails with `AttributeError: 'MimiConfig' object has no attribute 'rope_theta'`
   (5.x moved theta into `config.rope_parameters`).

The `cuda` extra therefore uses exact `==` pins rather than ranges. Verified
working on an RTX 5070 Ti / Python 3.13:

| package | version |
| --- | --- |
| torch | 2.11.0+cu128 |
| torchaudio | 2.11.0+cu128 |
| transformers | 5.15.1 |
| qwen-tts-hf | 0.1.1.post1 |
| faster-qwen3-tts | 0.4.0 |
| accelerate | 1.12.0 |

Cold load 2.6 s, and ~740 ms from calling the TTS to the first audio chunk at
`chunk_size=8` — synthesis only, not end-to-end TTFA, which additionally waits on
the LLM and on segment coalescing. Change one of these versions only deliberately
— the threads above show the combination is version-fragile.

Note the two extras are separate `uv sync` invocations and each drops the
other's packages, so use `uv sync --extra cuda --extra dev` to keep both; a bare
`uv run` re-syncs and uninstalls whatever the last sync omitted.

## VRAM budget (hard cap 10 GB)

| Component | Default | Ceiling |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign (bf16) | ~5.0 GB | ~5.0 GB |
| LLM (Q4_K_M, 4k KV) | Qwen3-4B ≈ 3.0 GB | Qwen3-8B ≈ 5.0 GB |
| **Total** | **≈ 8.0 GB** | **≈ 10.0 GB** |

`App` refuses to start if too little VRAM is free. The ggml path later cuts the
TTS side to roughly 1.8 GB.

## Why this stack (2026)

**Qwen3-TTS** (Alibaba, Jan 2026, Apache-2.0) is the only current open-weight
model that satisfies all four requirements at once:

- **Streaming text input, not just streaming output.** Its dual-track
  architecture predicts acoustic tokens *per incoming text token*: 97 ms (0.6B) /
  101 ms (1.7B) first-packet latency. Same property as CosyVoice-2's "bistream",
  but faster and with better WER (1.24, ahead of CosyVoice 3 and Seed-TTS).
- **Instruction control that is structurally out-of-band.** Control signals are
  prepended in ChatML form, separate from the text to be spoken.
- **Small.** 0.6B / 1.7B, 24 kHz output.
- **A real native path** — see [docs/native-port.md](docs/native-port.md). This
  is what decided it over better-sounding alternatives.

Checkpoints are separate per mode: `Base` (3 s voice cloning), `CustomVoice`
(9 fixed speakers + instruct), `VoiceDesign` (voice invented from a description).
**VoiceDesign is 1.7B-only.** This project uses VoiceDesign.

### Two things worth knowing before you dig in

**The official `qwen-tts` package exposes no streaming API.** Despite the model
card advertising low-latency streaming, its generate calls return fully rendered
waveforms
([discussion](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/discussions/4)).
The capability is real in the architecture but unexposed in the reference code,
so a third-party runtime is mandatory — hence `faster-qwen3-tts` (MIT), which
also adds CUDA-graph decode (156 ms TTFA, RTF 4.78 on a 4090). Treat the
advertised 97 ms as a lab number: one community implementation buffers ~38 tokens
(~3 s) before emitting to keep the voice stable. `py/bench/latency.py` measures what
we actually get.

**Inline emotion tags do not exist here.** Qwen3-TTS has no trained support for
positional tags like `(happy)` or `[whisper]` mid-sentence — it is an open
feature request with no maintainer response
([#238](https://github.com/QwenLM/Qwen3-TTS/discussions/238),
[#218](https://github.com/QwenLM/Qwen3-TTS/discussions/218)). Beware third-party
claims: the "80+ emotions" ComfyUI node is temperature/top-p/repetition-penalty
modulation, not an emotion channel. Emotion here is the `instruct` string, which
changes per utterance but not mid-phrase. Models with genuinely trained inline
tags (Fish Audio S2/OpenAudio, Breeze TTS 2) each give up either streaming text
input or open local weights.

### Considered and rejected

- **Fun-CosyVoice3-0.5B** (Apache-2.0, proven bistream, 150 ms) — the plan B.
- **IndexTTS-2** — best emotion/timbre disentanglement, no streaming story.
- **Kokoro-82M** — excellent and tiny, but no instruction control at all.
- **Qwen3-Omni 30B-A3B** — one model, text in / speech out, persona via system
  prompt; far over the VRAM budget.
- **sherpa-onnx** — the best Unity story by far, but no Qwen3-TTS support
  ([#3104](https://github.com/k2-fsa/sherpa-onnx/issues/3104), unanswered) and
  its TTS models have no instruct channel.

## Design notes

**The persona / delivery split.** In VoiceDesign the `instruct` string *is* the
voice identity, so rewriting it wholesale between utterances drifts the speaker.
`persona` is held byte-identical and the seed is pinned; only `delivery` moves.
`/persona` exists but will change who is speaking.

**Bounded queues, everywhere.** When the device falls behind, the ring buffer's
blocking write stalls TTS, which stalls the segmenter, which stops consuming LLM
deltas. Backpressure reaches the token source and memory stays flat.

**Underruns are counted, never hidden.** The ring buffer is fixed size. An
underrun is the only honest signal that the pipeline cannot keep up, so growing a
buffer to paper over one would destroy the measurement.

## Layout

| Path | What it is |
|---|---|
| `py/src/llmvoice/textprep.py` | streaming `<think>` stripper |
| `py/src/llmvoice/segmentation.py` | stream2sentence wrapper + tuned defaults |
| `py/src/llmvoice/style.py` | the out-of-band style channel |
| `py/src/llmvoice/pipeline.py` | three stages, two bounded queues |
| `py/src/llmvoice/audio/sink.py` | ring buffer, device output, underrun tracking |
| `py/src/llmvoice/tts/qwen3_torch.py` | Qwen3-TTS VoiceDesign streaming |
| `py/src/llmvoice/config.py` | every tunable, as dataclass defaults — edit here |
| `py/src/llmvoice/paths.py` | resolves config paths against the checkout root |
| `py/bench/latency.py` | TTFA / RTF / underrun sweep with acceptance targets |
| `docs/native-port.md` | the port plan, the C ABI, and what was rejected |
| `cpp/` | the native port — empty until the ggml backend is measured |

## Licensing

Qwen3-TTS Apache-2.0, Qwen3 LLM Apache-2.0, llama.cpp MIT, faster-qwen3-tts MIT,
stream2sentence MIT. All usable in a commercial game.
