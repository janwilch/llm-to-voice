# Porting to a native DLL for Unity

The Python prototype in `py/` is disposable and stays as a voice-auditioning harness. The compute lives in two native runtimes: llama.cpp (LLM) and qwentts.cpp (TTS, itself llama.cpp + a codec). Everything in `cpp/` is orchestration around them, exposed to Unity as a single C ABI.

## Architecture

| Component | Native form |
|---|---|
| Chat LLM | GGUF via llama.cpp |
| TTS talker (1.42B) | GGUF via llama.cpp |
| TTS MTP code predictor (142M) | GGUF via llama.cpp |
| Speech encoder / code2wav decoder | fp16 ONNX via onnxruntime, or ggml |

TTS is provided by [ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), vendored at a pinned commit (`cpp/bin/_deps/qwentts_cpp-src`). It supports VoiceDesign (free-text style instruct, not just voice cloning) and streaming via a frame-ramping `on_chunk` callback. Re-check [llama.cpp PR #26254](https://github.com/ggml-org/llama.cpp/pull/26254) (mainline Qwen3-TTS, Base-only today) periodically — if VoiceDesign lands there, qwentts.cpp's single-maintainer dependency can be dropped.

## The C ABI

```c
typedef struct llmvoice_handle llmvoice_handle;

typedef struct {
    const char* llm_model_path;
    const char* tts_talker_path;
    const char* tts_codec_path;
    int         sample_rate_hint;   // 0 = native 24000
} llmvoice_config;

llmvoice_handle* llmvoice_create   (const llmvoice_config* cfg);
void              llmvoice_warmup   (llmvoice_handle*);
void              llmvoice_destroy  (llmvoice_handle*);

void  llmvoice_set_style (llmvoice_handle*, const char* persona_utf8,
                                             const char* delivery_utf8);
void  llmvoice_submit    (llmvoice_handle*, const char* prompt_utf8);
void  llmvoice_cancel    (llmvoice_handle*);

int   llmvoice_poll_text (llmvoice_handle*, char* dst_utf8, int max_bytes);
int   llmvoice_poll_pcm  (llmvoice_handle*, float* dst, int max_frames);
int   llmvoice_is_done   (llmvoice_handle*);
```

Both `poll_*` calls drain whatever is ready since the last call and zero-pad/truncate as needed; they never block and never call back into the caller. No delegates cross the boundary in either direction — a callback into managed code cannot be marshaled reliably under IL2CPP, so this API stays pull-only even though the native side is callback-driven internally. `poll_pcm` is called from Unity's audio thread (`OnAudioFilterRead` or a `PCMReaderCallback` `AudioClip`); `poll_text` is called from `Update()`.

Model loading happens exactly once, in `llmvoice_create` — the GGUF weights, codec and tokenizer are loaded into `qt_context`/the llama.cpp context and stay resident for the handle's entire lifetime, across as many `submit` calls as the game session needs. This isn't new: `Qwen3TtsBackend` already works this way — `qt_init` runs once in its constructor and every `synthesizeToQueue` call reuses the same `_context`. `llmvoice_warmup` exposes the existing `ITtsBackend::warmup()` (and, once added, an equivalent on the LLM backend) through the ABI: it runs one throwaway generation through the already-loaded context to prime GPU kernels and allocators, so the *first real* `submit` isn't the one that eats the cold-start latency. Call it once, right after `create`, before the player's first prompt — not before every `submit`.

`llmvoice_handle` is intentionally an opaque type: the public header only forward-declares `struct llmvoice_handle;` and never defines it, so callers (Unity included) hold nothing but a pointer they pass back into every call — they can't see or touch its contents, which is what lets the real definition be arbitrary C++ on the other side of the ABI. The actual struct is defined only in the `.cpp` that implements these functions, roughly:

```cpp
// llmvoice.cpp — the real definition; nothing outside this file ever sees it
struct llmvoice_handle {
    std::unique_ptr<ITtsBackend> tts;
    std::unique_ptr<ILlmBackend> llm;
    std::jthread                 worker;
    ma_rb                        pcm_ring;    // lock-free, read from the audio thread
    std::mutex                   text_mutex;  // fine here — read from Update(), not real-time
    std::string                  text_buffer;
};

llmvoice_handle* llmvoice_create(const llmvoice_config* cfg) {
    auto* h = new llmvoice_handle{ /* construct tts/llm backends from cfg, start worker */ };
    return h;
}
```

This is the standard "opaque pointer" pattern (also called PImpl) for wrapping C++ in a C ABI: it lets `llmvoice.h` stay pure C with no exposed C++ types, while the implementation is free to use `std::jthread`, `std::unique_ptr`, templates, exceptions internally — none of that crosses the boundary, only the pointer does.

## Internal design

- **qwentts's `on_chunk` → `BlockingQueue<std::vector<float>>` → llmvoice worker thread.** This is already how `Qwen3TtsBackend::synthesizeToQueue` works and stays as-is: qwentts's callback fires on its own internal compute thread and must not block, so it just pushes into the bounded queue. The same pattern applies to LLM text deltas once that backend exists — `BlockingQueue<std::string>` (or a small delta+metadata struct) from the llama.cpp callback thread into the same llmvoice worker.
- **Chunk accumulation happens behind the ABI, not in Unity.** The llmvoice worker thread drains both `BlockingQueue`s and writes into a ring buffer that `poll_pcm`/`poll_text` read from. qwentts's audio chunks are irregular by design (first chunk is one 12.5 Hz frame, then doubles up to 8 frames as the stream settles), so normalizing them into a flat, poll-able buffer is native-library responsibility. Unity only ever sees fixed-size pulls; a native CLI test harness consumes the identical, already-normalized stream instead of re-deriving qwentts's framing.
- **Only the PCM ring needs to be lock-free.** `poll_pcm` is called from Unity's real-time audio thread, so that ring must be a true SPSC structure with no mutex on the read side. `poll_text` is called from `Update()` on Unity's main thread, which has no real-time constraint — a plain mutex-guarded buffer (the same style `BlockingQueue` already uses elsewhere in this codebase) is simpler and sufficient there; don't build a second lock-free ring for text.
- **Take the PCM ring from a library, don't hand-roll the atomics.** Vendor one via `FetchContent`, the same pattern already used for CLI11/llama.cpp/qwentts.cpp. miniaudio's `ma_rb` is the best fit: single header, MIT, and its acquire/commit contract (reserve a write region, commit what was actually written, same for reads) maps directly onto `poll_pcm`'s "fill up to N frames, return how many, zero-pad the rest" contract — closer to what's needed than a generic per-item lock-free queue (e.g. rigtorp/SPSCQueue, moodycamel::ReaderWriterQueue), which would still need a wrapper to turn "pop one item" into "fill this buffer." It also doubles as the CLI's playback backend (§ below), so it's one dependency serving both jobs.
- **`RingBuffer.read_into`'s semantics port, its locking does not.** Keep "fill a caller-owned buffer, return the frame count, zero-pad the remainder" from `py/src/llmvoice/audio/sink.py`; drop the mutex + `notify_all()` — a blocking acquire on Unity's audio thread is a priority-inversion hazard.
- **`UnderrunTracker` ports verbatim.** Pure function of (frames requested, frames available, producing?, playing?); reuse the bracket-and-count logic as-is so pre-roll silence and end-of-stream tail aren't miscounted as dropouts.

## The CLI as the ABI's test and voice-design harness

`cpp/src/main.cpp` currently talks to `ITtsBackend`/`BlockingQueue` directly. Once the ABI exists, migrate it to be a client of `llmvoice_*` instead — this makes the CLI a real integration test of the exact path Unity uses, and keeps it as the voice-audition tool:

- `llmvoice_create` from the existing `--tts-backend`/model-path flags, followed by one `llmvoice_warmup` call before the poll loop — the CLI should pay the same warmup cost Unity will pay once per session, not once per prompt.
- `llmvoice_set_style` from a new `--persona`/`--delivery` flag pair, for auditioning VoiceDesign prompts without opening Unity.
- `llmvoice_submit(prompt)` in place of the direct `synthesizeToQueue` call.
- A poll loop replacing today's `queue.pop()` loop: call `llmvoice_poll_pcm` and `llmvoice_poll_text` on an interval, print text deltas as they arrive (as today), and either play the PCM (via miniaudio, already vendored for the ring buffer above) or accumulate it purely for the CLI's own purposes (e.g. write a WAV for review) — the ABI itself stays streaming-only, only the CLI-side consumer buffers to disk.
- Loop until `llmvoice_is_done`, then `llmvoice_destroy`.

## Remaining steps

1. **LLM backend.** Add an `ILlmBackend`/llama.cpp wrapper mirroring `ITtsBackend`, feeding token deltas into a `BlockingQueue<std::string>` the same way `Qwen3TtsBackend` feeds PCM, including its own `warmup()` (one throwaway generation through the loaded context) for `llmvoice_warmup` to call alongside the TTS one.
2. **`textprep`.** Port the `<think>`-stripping state machine from `py/src/llmvoice/`. Pin behavior against `py/tests/test_textprep.py`.
3. **Segmenter + coalescer.** Port `py/src/llmvoice/segmentation.py`'s logic (minimum sentence length, never-split-numbers guard, no quick-yield of the first fragment — each segment is a separate VoiceDesign generation and a clause-length opener resamples the speaker as a different voice). No library covers this: generic sentence-boundary detectors (e.g. ICU's `BreakIterator`) don't know the never-split-numbers or coalescing rules, and pulling one in would still need the same custom logic on top. Port the ~100 lines directly; pin against `py/tests/test_segmentation.py`.
4. **The PCM ring (vendored, e.g. `ma_rb`) + the mutex-guarded text buffer + `UnderrunTracker` port**, sitting behind `poll_pcm`/`poll_text` as described above.
5. **The C ABI itself** (`cpp/include/llmvoice.h` + `SHARED` CMake target; currently `BUILD_SHARED_LIBS` is forced off).
6. **Migrate `main.cpp`** to the ABI, per above.
7. **Decide the sample-rate resample point**: native layer vs. Unity's `AudioClip` resampling against the project's mixer rate. Affects which side of the ring buffer the resampler sits on — decide before step 8.
8. **Thin C# P/Invoke wrapper + `OnAudioFilterRead`/`PCMReaderCallback`.** Blittable types only, no delegates crossing the boundary.
9. **Loading-time UX.** Two GGUF models + codec load slowly enough to need a loading screen or background warmup.
10. **GPU frame-budget measurement.** ggml's kernel queue doesn't know about a 16.6 ms frame budget. Measure frame-time percentiles as soon as the LLM+TTS pipeline runs end-to-end (steps 1-4), not after Unity integration. Levers if hitches show up: cap `n_gpu_layers`, move the LLM to CPU under GPU contention, or gate speech to moments where a hitch is cheap.
