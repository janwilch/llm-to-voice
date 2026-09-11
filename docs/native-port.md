# Porting to a native DLL for Unity

The Python prototype in `py/` is disposable. This document records the
constraints it was built under so the port into `cpp/` is a translation rather
than a redesign, and records the two alternatives that were rejected so they are
not re-litigated.

## The shape of the problem

`py/` is ~2,100 lines, and almost none of it is compute. The compute already
lives in two native C/C++ runtimes: llama.cpp behind `py/src/llmvoice/llm/llama_server.py`,
and PyTorch's CUDA kernels behind `py/src/llmvoice/tts/qwen3_torch.py`. Everything
written here is orchestration — a `<think>` stripper, a segmenter, a coalescer, a
ring buffer, an underrun rule.

## Why Qwen3-TTS makes this possible

The Qwen3-TTS backbone *is* a Qwen3 LLM, so it runs on ggml. That means the LLM
and the TTS model can share one inference runtime, and neither needs Python:

| Component | Native form |
|---|---|
| Chat LLM | GGUF via llama.cpp |
| TTS talker (1.42B) | GGUF via llama.cpp |
| TTS MTP code predictor (142M) | GGUF via llama.cpp |
| Speech encoder / code2wav decoder | fp16 ONNX via onnxruntime, or ggml |

Both llama.cpp and onnxruntime are plain C libraries with Windows builds, which
is the entire reason this model was chosen over better-sounding alternatives.

### Reference implementations

- **[ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp)
  — the target.** A C++17 port of Qwen3-TTS on GGML, MIT over the Apache-2.0
  model. It closes what used to be the open gap in this document:
  - **VoiceDesign** from a free-text attribute instruct string — our style
    channel, not just Base voice cloning.
  - **Streaming**, described as stateful frame-by-frame codec decode where the
    first audio callback fires one frame after the first Talker step, and the
    streamed output matches the offline full decode exactly.
  - **A plain C header** (`qwen.h`, single prefix, C linkage) written for
    ctypes / bindgen / cgo consumers, plus `-DQWEN_SHARED=ON` for a real `.dll`.
  - CPU / CUDA / Vulkan / ROCm / Metal, with Windows CUDA and Vulkan build
    scripts.

  Caveat, and it is the main risk in this plan: it is young and effectively
  single-maintainer (~159 stars, ~133 commits as of Sept 2026). Vendor it at a
  pinned commit. A frozen C++17 + GGML tree is *genuinely* stable in a way the
  frozen torch / transformers / qwen-tts-hf combination in `py/pyproject.toml`
  is not, and that asymmetry is most of the argument for porting at all.

- **[HaujetZhao/Qwen3-TTS-GGUF](https://github.com/HaujetZhao/Qwen3-TTS-GGUF)** —
  the same split (llama.cpp for talker/predictor, onnxruntime for
  encoder/decoder) with a Python `TTSEngine` on top. Supports VoiceDesign and
  reports **~1.8 GB VRAM at RTF 0.35 on an RTX 5050**, ~300 ms first packet.
  Useful as a cross-check on qwentts.cpp's numbers and as a fallback split.

- **[ggml-org/llama.cpp PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254)**
  — merged to master Aug 2026. Brings Qwen3-TTS into mainline with a
  `llama-tts` binary and a follow-up `POST /tts` server endpoint, but **Base
  only**: voice cloning from a reference clip, no VoiceDesign, no instruct
  channel. So mainline is the fallback for the *LLM* half, not the TTS half.
  Worth re-checking before Stage 2 — if VoiceDesign lands upstream, the
  single-maintainer risk above disappears.

### Prior art for the Unity shape

[UndreamAI's LLMUnity](https://github.com/undreamai/LLMUnity) ships llama.cpp
into Unity as a standalone C++/C# library (`LlamaLib`) and has shipped in
commercial titles — *Verbal Verdict*, *Case Closed* — on Unity 2021 LTS through
Unity 6, across Nvidia / AMD / Metal. The pattern is proven. We are doing the
same thing with a second model attached.

## The C ABI

```c
typedef struct llmvoice_handle llmvoice_handle;

llmvoice_handle* llmvoice_create (const llmvoice_config* cfg);
void  llmvoice_set_style (llmvoice_handle*, const char* persona_utf8,
                                            const char* delivery_utf8);
void  llmvoice_submit    (llmvoice_handle*, const char* prompt_utf8);
int   llmvoice_poll_pcm  (llmvoice_handle*, float* dst, int max_frames);
void  llmvoice_cancel    (llmvoice_handle*);
void  llmvoice_destroy   (llmvoice_handle*);
```

### `poll_pcm` is a pull, and that is deliberate

Unity drains it from `OnAudioFilterRead` or a `PCMReaderCallback` `AudioClip`.
There is no callback from native code into managed code anywhere in this API.

That is not a style preference, and it is now confirmed rather than assumed. The
sherpa-onnx Unity plugins document that their chunk-callback API wraps the user
callback in a closure / instance-method delegate that **IL2CPP cannot marshal**,
so on IL2CPP builds they auto-fall-back to the callback-less non-streaming
`Generate` — which would defeat the entire purpose of this project. A pull-based
boundary cannot inherit that bug. Keep it pull-based even if it looks awkward
from C++.

## What the prototype already got right

Each of these exists to make the port mechanical:

- **Stage boundaries carry only UTF-8 bytes or PCM frames.** No Python objects
  cross a queue, so every queue becomes a lock-free ring buffer.
- **Bounded queues everywhere.** Memory is flat under load and backpressure
  reaches the token source. A native port with unbounded queues would drift.
- **`TtsBackend` / `LlmBackend` protocols.** The pipeline never learns which
  backend it is driving, so a ggml backend can replace `qwen3_torch` underneath
  it — add it as a third branch in `py/src/llmvoice/tts/backend.py:build_backend`.
  This is what makes Stage 1 below cheap.
- **Flat scalar config.** `AppConfig` is a tree of scalar dataclasses precisely
  so it can become `llmvoice_config` without inventing a serialisation format.
- **`cancel()` on every stage.** Barge-in is a hard requirement in a game, and
  retrofitting cancellation into a threaded pipeline is painful.
- **`UnderrunTracker`.** A pure function of (frames requested, frames available,
  producing?, playing?). Port it verbatim; the two edge cases it encodes were
  both bugs at some point.

## What will need rethinking

- **`RingBuffer.read_into` does *not* port verbatim** — this document used to say
  it did, and that was wrong. `py/src/llmvoice/audio/sink.py` acquires a mutex and
  calls `notify_all()` on the read side. That is fine on a PortAudio callback in
  a CLI. On Unity's audio thread, a blocking mutex acquire plus a condvar signal
  is a priority-inversion hazard that will manufacture exactly the underruns the
  tracker exists to detect. In C++ it becomes an SPSC ring with atomic read/write
  indices and no mutex on the consumer side. The *semantics* port as-is: fill a
  caller-owned buffer, return the frame count, zero-pad the remainder — that is
  `llmvoice_poll_pcm`.
- **Sample rate.** The model emits 24 kHz mono; a Unity mixer typically runs at
  48 kHz. Either set the project's `AudioSettings` output rate to 24 kHz, which
  affects every other sound in the game, or resample in the native layer. Decide
  this early — it changes which side of the ring buffer the resampler sits on.
- **GPU contention is the unsolved problem, and it is unsolved in all three
  options.** ggml's kernel queue knows nothing about a 16.6 ms frame budget, so
  long TTS decodes will cause frame hitches, and there is no clean fix on
  consumer Windows. The levers are coarse: cap `n_gpu_layers`, move the LLM to
  CPU when the GPU is contended, or gate speech to moments where a hitch is
  cheap. Measure frame-time percentiles at Stage 1, not at Stage 4.
- **Threading.** The prototype uses three OS threads and blocking queues. In a
  game, prefer the engine's job system or a single dedicated worker thread with
  lock-free SPSC queues; blocking a Unity thread is not acceptable.
- **The segmenter.** `stream2sentence` is Python. Its *logic* must be
  reimplemented in C++ — the parameters worth keeping are a minimum sentence
  length, the never-split-numbers guard, and the segment coalescer. Quick-yield
  of the first fragment is the classic TTFA lever and is deliberately **off**:
  each segment is a separate VoiceDesign generation that re-samples the speaker,
  and a clause-length opener has so little context that it comes out as a
  different person before the voice settles. Port the coalescer with it, or the
  port will sound worse than the prototype for a latency win nobody asked for.
  See `py/src/llmvoice/segmentation.py` for the tuned values and
  `py/tests/test_segmentation.py` for the behaviour to preserve.
- **`textprep`.** Straight port, and small: it strips `<think>` blocks and
  nothing else. A pure state machine with a bounded holdback, so a partial
  `<think>` split across two deltas is never emitted. `py/tests/test_textprep.py`
  pins that behaviour.
- **Model loading time.** Loading two GGUF models plus the codec is slow enough
  to need a loading screen or a background warmup.
- **VRAM sharing.** The game itself wants VRAM. The ggml path's ~1.8 GB for TTS
  is what makes this viable at all; budget for the renderer too, and consider
  keeping the LLM on CPU if the GPU is contended.

## The order of work

The staging matters more than the destination, because each stage de-risks the
next and none of it is throwaway.

**Stage 1 — swap the TTS backend, stay in Python.** `faster-qwen3-tts` ships an
experimental adapter for the qwentts.cpp runtime (`qwentts-cpp-python`; GGML is
opt-in, Torch/CUDA-graph stays the default). Add it as a third branch in
`tts/backend.py:build_backend`, which is what that indirection was for. Then
measure ggml VoiceDesign quality, TTFA, RTF and VRAM with `py/bench/latency.py`
against the existing acceptance targets — **before writing a line of C++ or
opening Unity.**

This stage is the gate. If ggml VoiceDesign does not hold the speaker identity
across segments, we find out here for the price of one backend class, and
Fun-CosyVoice3-0.5B (plan B in the README) is still on the table. Everything
after this point assumes Stage 1 passed.

**Stage 2 — port the glue into `cpp/` behind the C ABI above.** Link `libqwen`
and `libllama`. It is roughly 300 lines of real logic: textprep, the segmenter,
the coalescer, the ring buffer, the underrun rule. Port the *test vectors* from
`py/tests/` first and make the C++ pass them — those tests are the specification,
and several of them encode bugs that were fixed once already.

**Stage 3 — thin C# P/Invoke wrapper plus `OnAudioFilterRead`.** Blittable types
only. No delegates cross the boundary, for the IL2CPP reason above.

**Stage 4 — keep the Python CLI.** It is the voice-auditioning and tuning
harness, and it stays useful for the rest of the project's life. That is why
`py/` is a peer of `cpp/` rather than something the port deletes.

## Rejected: ship the Python prototype as a sidecar

Run `py/` under uv in the background and stream to Unity over REST or a pipe.
Feasible, and it could demo next week. Rejected for shipping:

- **Transport is not the problem.** 24 kHz mono float32 is 96 KB/s; loopback
  adds well under a millisecond. Anyone arguing REST is too slow for this is
  wrong.
- **VRAM is the problem.** It keeps the PyTorch path: ~5.0 GB for TTS against
  ~1.8 GB for ggml, plus a second CUDA context (~300–600 MB), in a separate
  process the renderer cannot coordinate with. That is 3–4 GB of a player's card
  handed to a subprocess — most of an 8 GB card.
- **Payload and startup.** torch+cu128 is ~5.7 GB installed; with transformers,
  nltk and an embeddable interpreter it is 6–7 GB on top of the game and the
  models. Cold start is interpreter + `import torch` (1–3 s alone) + CUDA init +
  the 2.6 s model load.
- **Fragility.** The README's "GPU setup" section is the evidence: six exact
  `==` pins, an sm_120 wheel-index trap, a flash-attn build that segfaults nvcc,
  and a `transformers` ceiling where 5.16 breaks Mimi loading. That is an
  honestly documented *development* environment. It is not something to put on a
  stranger's machine.
- **Barge-in gets harder.** `cancel()` is one synchronous call across three
  stages today. Across a process boundary it becomes a cancel endpoint racing an
  in-flight audio stream, with Unity discarding PCM already in transit — new
  correctness surface on a hard requirement.
- If it *is* used (dev tool, internal build, early-access experiment), use stdio
  or a named pipe rather than a TCP listener: a listening socket in a game
  directory draws a Windows Defender Firewall prompt and antivirus attention.

## Rejected: rebuild in managed C# for IL2CPP

IL2CPP transpiles C# to C++ and compiles AOT. It does not produce CUDA kernels.
So "rebuild it in C#" resolves to one of two things, and neither is what it
sounds like:

**C# that P/Invokes native inference libraries** is Stage 3 above with a
different name. The inference is still native C++. This is the right *delivery*
shape, not a separate option.

**Genuinely managed inference via Unity Inference Engine** (the renamed Sentis)
fails on three independent counts:

1. **Quantization ceiling.** It supports None (fp32), Float16 and Uint8 —
   [nothing below 8-bit](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/quantize-a-model.html).
   There is no Q4_K_M equivalent. The 4B LLM at uint8 is ~4 GB and the 1.7B TTS
   ~1.7 GB, against 3.0 + 1.8 GB today, before KV cache and activations, in a
   process that also has to render a game.
2. **Quantization buys no speed there.** Unity's own docs frame it as reducing
   storage and memory "without significantly affecting inference speed" — it
   dequantizes to compute. So fp16 compute cost at uint8 accuracy, and per-tensor
   linear uint8 on an acoustic-token predictor is likely audibly destructive.
3. **The ONNX export does not exist.** Nobody has exported Qwen3-TTS's dual-track
   streaming architecture with an incremental KV cache to ONNX. On Unity's own
   forums, users cannot get plain Phi-3 / SmolLM through the importer and no
   Unity staffer answers with a supported path. We would be pioneering the export
   *and* the runtime support.

The honest version of the managed path is: write a GGUF loader, an attention and
KV cache implementation, and a code2wav vocoder as HLSL compute shaders — i.e.
reimplement ggml's Vulkan backend. Person-years, landing slower than the library
we can link today.

And note the sting: if the appeal was "portable GPU compute without a CUDA
dependency", **ggml already has a Vulkan backend**, inside the option we chose.
