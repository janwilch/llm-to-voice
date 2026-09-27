# llm-to-voice

A native C/C++ library that turns a prompt into a spoken answer, fully offline. It streams the LLM's reply as text and as voice (mono float PCM, 24 kHz) at the same time, so playback starts after the first (length-configurable) segment instead of after the whole answer.

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (Qwen3 LLM) and [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) (Qwen3-TTS VoiceDesign and Base), sharing a single ggml runtime and GPU context. Everything ships as one self-contained shared library with a small C API that any language or engine with a C FFI can load, e.g. to give interactive characters a reactive voice.

## Features

- **Streaming pipeline:** LLM → segmenter → TTS on separate threads; poll text and audio independently.
- **Voice design:** describe the speaker in plain language ("a mature woman with a warm voice…") and fix it with a seed.
- **Voice cloning:** clone a speaker from a WAV sample and its transcript.
- **Speakable segments:** the segmenter drops `<think>` blocks and emoji, splits on sentence ends (aware of abbreviations, initials, numbers and list markers) and merges short sentences for more natural TTS.
- **One library, one ggml:** llama.cpp and qwentts.cpp are linked statically into `llmvoice` (`llmvoice.so` / `.dll` / `.dylib`), which exports only the `llmvoice*` C functions.
- **Cancellable:** `llmvoiceCancel` stops every stage, e.g. when the user interrupts.

## Getting started

### Requirements

- CMake ≥ 3.20, Ninja, and a C++23 compiler (the presets use Clang)
- Git (dependencies are fetched and patched at configure time)
- About 5 GB of disk space for the models

### Models

The models go into `models/` at the repo root:

| File | Source | Folder |
|---|---|---|
| `Qwen3-4B-Q4_K_M.gguf` | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | `models/Qwen3-4B-GGUF/` |
| `qwen-talker-1.7b-base-Q4_K_M.gguf` (cloning), `qwen-talker-1.7b-voicedesign-Q4_K_M.gguf` (voice design), `qwen-tokenizer-12hz-Q4_K_M.gguf` | [Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) | `models/Qwen3-TTS-GGUF/` |

### Build

```sh
cd cpp
cmake --preset release          # or: debug, release-vulkan
cmake --build --preset release
ctest --test-dir build/release  # optional
```

The library and the CLI end up in `cpp/build/<preset>/bin/`.

### Try it

Design a voice, then let the pipeline answer with it on the default audio device (run from the repo root):

```sh
cpp/build/release/bin/llmvoice_cpp voice-design -p "Hello there. Sit down, and I will tell you about the sea." --instruct "An old man with a deep, calm voice."
cpp/build/release/bin/llmvoice_cpp pipeline -p "Tell me a short story about a lighthouse."
```

| Subcommand | Effect |
|---|---|
| `voice-design --instruct <text> [--out-path out/voice]` | speak the prompt in the described voice (random seed per run) and save it as `sample.wav` + `transcript.txt` |
| `pipeline [--sample-wav out/voice/sample.wav] [--sample-text out/voice/transcript.txt] [--no-think]` | answer the prompt in the cloned voice; any WAV with a matching transcript works as sample |
| `llm-only [--skip-segmenter] [--no-think]` | run only the LLM, optionally without segmentation |

`--no-think` skips the model's reasoning block (faster first audio, lower answer quality).

## Usage

The whole API is in [`cpp/src/llmvoice.h`](cpp/src/llmvoice.h). A minimal pipeline run:

```c
LlmvoiceConfig config = {
    .llmModelPath = "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
    .llmContextSize = 8192,
    .ttsTalkerPath = "models/Qwen3-TTS-GGUF/qwen-talker-1.7b-base-Q4_K_M.gguf",
    .ttsCodecPath = "models/Qwen3-TTS-GGUF/qwen-tokenizer-12hz-Q4_K_M.gguf",
    .ttsMode = 0,   // 0 = Base (cloning), 1 = VoiceDesign; must match the talker model
};

LlmvoiceHandle* voice = llmvoiceCreate(&config);   // NULL on failure, see llmvoiceLastError()

llmvoiceCreateLlmContext(voice, "You are a friendly lighthouse keeper.");
llmvoiceCreateTtsBaseContext(voice, "out/voice/sample.wav", "out/voice/transcript.txt");
llmvoiceWarmup(voice);   // after the contexts: the warmup uses the voice

llmvoiceSubmitPipeline(voice, "Hello! Who are you?", false, 1024);

char text[256];
float pcm[1024];
while (!llmvoiceIsDone(voice)) {
    size_t bytes = llmvoicePollText(voice, text, sizeof text);   // UTF-8, not NUL-terminated
    size_t frames = llmvoicePollPcm(voice, pcm, 1024);           // mono float, 24 kHz
    // show the text, play the audio; both calls are non-blocking
}
// llmvoiceLastError() is now "" on success, or the reason a stage failed

llmvoiceDestroy(voice);
```

With `.ttsMode = 1` and the voicedesign talker, use `llmvoiceCreateTtsVoiceDesignContext(voice, seed, instruct)` instead.

Both outputs must be polled: the TTS stalls if the text is not read. In the pipeline, a segment's text is released once the start of its audio has been polled, so text stays in sync with playback. `llmvoiceSubmitLlm` and `llmvoiceSubmitTts` run a single stage, and `llmvoiceSetSegmenterConfig` tunes how text is split before it is spoken.

## How it works

```
// before: llmvoiceCreate (TTS mode is fixed here)

llmvoiceCreateLlmContext(systemPrompt):    // fresh chat context
  ILlmBackend.createFreshContext(systemPrompt)
llmvoiceCreateTtsBaseContext(wav, transcript) / llmvoiceCreateTtsVoiceDesignContext(seed, instruct):
  ITtsBackend.createBaseContext / createVoiceDesignContext  // speaker for following generations
llmvoiceWarmup
llmvoiceSetSegmenterConfig(minSentenceLength, coalesceMinChars):  // from next submit on

llmvoiceSubmitPipeline(prompt):  // also SubmitLlm (LLM [+ segmenter]) and SubmitTts (TTS only)
  LLM THREAD:        ILlmBackend.synthesizeToQueue(prompt, llmTokenQueue)
  SEGMENTER THREAD:  llmTokenQueue -> cleaned segments (w/o <think>) -> segmentsToTtsQueue
  TTS THREAD:        for each segment:
                       segmentsOutQueue.push(segment, pcmOut.totalWritten)
                       ITtsBackend.synthesizeToBuffer(segment, pcmOut)  // ring buffer

llmvoiceCancel:    cancel backends, close queues, join threads
llmvoicePollText:  RETURN bytes from segmentsOutQueue, once pcmOut.totalRead > the segment's start
llmvoicePollPcm:   RETURN frames from pcmOut
llmvoiceIsDone:    RETURN no thread running AND all outputs drained

// after: llmvoiceDestroy
```

## Project layout

```
cpp/
├── CMakeLists.txt      dependency setup (one shared ggml) and targets
├── patches/            patches applied to the fetched ggml and llama.cpp
├── src/
│   ├── llmvoice.h/.cpp public C API and pipeline orchestration
│   ├── llm/            LLM backend (llama.cpp)
│   ├── tts/            TTS backend (qwentts.cpp)
│   ├── segmenter/      text → speakable segments
│   ├── threading/      blocking queue, lock-free PCM ring buffer
│   ├── benchmark/      timing and memory output of the CLI
│   ├── helpers/        file and enum helpers
│   └── main.cpp        CLI
└── tests/              Catch2 tests
```

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); the models are licensed under Apache-2.0.

## Running on Windows

Build with the [MSYS2](https://www.msys2.org/) **CLANG64** toolchain. The presets call plain `clang`/`clang++`, so run CMake from the CLANG64 shell or put `C:\msys64\clang64\bin` first on `PATH`.

```sh
pacman -S mingw-w64-clang-x86_64-{clang,cmake,ninja}
pacman -S mingw-w64-clang-x86_64-{vulkan-headers,vulkan-loader,shaderc}   # release-vulkan only
```

- Build as described above; the output is `cpp\build\<preset>\bin\llmvoice_cpp.exe` and `llmvoice.dll`.
- The binaries are self-contained: they need no MSYS2 DLLs at runtime, only `vulkan-1.dll` from the GPU driver (for `release-vulkan`).
- The `[bench]` output reports GPU memory and utilisation only with an NVIDIA driver (`nvml.dll`). CPU time on Windows advances in ~15.6 ms steps, so it is only meaningful for longer stages.
