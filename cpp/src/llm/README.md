# Qwen3Backend implementation notes

Research findings on the llama.cpp API surface needed to implement `Qwen3Backend.cpp`, gathered from the vendored llama.cpp source, `ILlmBackend.hpp`, the `Qwen3TtsBackend` sibling, and the Python prototype. Kept here so the implementation can be written directly against real, current signatures instead of possibly-outdated memory of the llama.cpp API.

## Vendored llama.cpp version

`cpp/CMakeLists.txt` pins llama.cpp at commit `982937a3337f7e97ef08fd5603f4157575ece7e1` (`b10909-6-g982937a33`, dated 2026-09-12) via `FetchContent`, checked out at `cpp/bin/_deps/llama_cpp-src`. This is a recent, post-sampler-rewrite, post-vocab-API-split version, so use the "modern" API (`llama_model_*`, vocab-scoped functions, sampler chains) — the old `llama_token_*`/`llama_new_context_with_model`-style functions still exist but are marked `DEPRECATED`.

## CMake prerequisite: llama is not linked into the app yet

`cpp/CMakeLists.txt` currently links `llmvoice_cpp` only against `CLI11::CLI11 qwen-core` — the `llama` target (which publicly exposes `llama.h`/`llama-cpp.h` and links `ggml`) is only linked into a separate `deps_smoke` test target, just to prove there's no ggml-symbol collision. Before `Qwen3Backend.cpp` can `#include <llama.h>` and link, `llama` needs to be added to `target_link_libraries(llmvoice_cpp ...)`.

Also, llama.cpp is built here as a subproject (`LLAMA_STANDALONE` is `OFF`), so `LLAMA_BUILD_COMMON` defaults to `OFF` and the `common` library (`common/common.h`, `common/sampling.h`, `common_sampler_*`, `common_chat_*`) is **not built or linked**. Implement directly against the raw `llama.h` C API — do not assume `common/` helpers are available.

## `ILlmBackend.hpp` contract

```cpp
virtual void warmup() = 0;
virtual void synthesizeToQueue(const std::string& prompt, BlockingQueue<std::string>& queue) = 0;
virtual void cancel() = 0;
```

This is a streaming, not synchronous-return, interface: push text deltas as `std::string` into `queue`, and `.close()` the queue when generation ends — mirror `Qwen3TtsBackend`'s `QueueCloser` RAII-close-on-exit pattern. `cancel()` should set a mutex-guarded flag checked during the generation loop, the same way `Qwen3TtsBackend` guards its `_cancelled` flag. Note `Qwen3Backend.hpp`'s factory `createQwen3Backend()` currently takes no model-path argument (unlike TTS's `createQwen3TtsBackend(talkerPath, codecPath)`) — it will need a model path parameter added to load a GGUF file. `cpp/src/main.cpp` parses a `--fake-llm` flag but doesn't wire up any LLM backend yet.

## `Qwen3TtsBackend` is an architectural reference only, not an API reference

`qwentts.cpp` (used by the TTS backend) builds its own custom ggml graphs directly and never calls llama.cpp's `llama_decode`/`llama_batch`/sampler API — it only shares the underlying ggml library. What's worth imitating from `Qwen3TtsBackend` is the class shape: an opaque handle wrapped in the class, constructor loads the model once, `warmup()` runs a throwaway generation, `synthesizeToQueue` pushes into the `BlockingQueue` behind a `QueueCloser` guard, a mutex-guarded `_cancelled` flag checked mid-generation, and `std::runtime_error` thrown on failure. The actual llama.cpp call sequence has to come from `llama.h` itself, below.

## Exact llama.h API to use

**Model load** (modern, non-deprecated):
```c
struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
```
Defaults via `llama_model_default_params()`; free with `llama_model_free()`. (`llama_load_model_from_file`/`llama_free_model` are deprecated wrappers.)

**Context creation** (modern):
```c
struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
```
Defaults via `llama_context_default_params()`; free with `llama_free(ctx)`. Relevant `llama_context_params` fields: `n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`, `n_threads`/`n_threads_batch`, `flash_attn_type`.

**Vocab** is its own opaque type fetched off the model, not the context:
```c
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
```
Use `llama_vocab_is_eog(vocab, token)` to check both EOS and EOT per generated token (not the deprecated `llama_token_eos`), plus `llama_vocab_bos/eos/eot(vocab)` and `llama_vocab_n_tokens(vocab)` if needed.

**Tokenization / detokenization**:
```c
int32_t llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len,
                        llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
int32_t llama_token_to_piece(const struct llama_vocab * vocab, llama_token token, char * buf,
                              int32_t length, int32_t lstrip, bool special);
int32_t llama_detokenize(const struct llama_vocab * vocab, const llama_token * tokens, int32_t n_tokens,
                          char * text, int32_t text_len_max, bool remove_special, bool unparse_special);
```
All three follow the standard two-call idiom: call with a buffer, and on a negative return re-allocate to `-result` and retry. For streaming output, call `llama_token_to_piece` per generated token and push the resulting UTF-8 fragment into the queue as one delta — this is the direct equivalent of the Python prototype's SSE `delta.content` chunks.

**Chat template**:
```c
int32_t llama_chat_apply_template(const char * tmpl, const struct llama_chat_message * chat,
                                   size_t n_msg, bool add_ass, char * buf, int32_t length);
const char * llama_model_chat_template(const struct llama_model * model, const char * name);
```
This does not use a Jinja parser — it pattern-matches a pre-defined list of known templates (Qwen's ChatML-style format is one of the built-ins). Get the model's own template via `llama_model_chat_template(model, nullptr)`, build a `llama_chat_message[]` from system+user roles, and apply the same two-call size idiom as tokenize. There is no direct equivalent of the Python side's `chat_template_kwargs: {enable_thinking: false}` in this API, since Qwen3's Jinja template branches on an `enable_thinking` variable this non-Jinja matcher doesn't expose. Pragmatic options: append the literal `/no_think` soft switch Qwen3 recognizes in-content, or rely on downstream `<think>...</think>` stripping (already planned per `docs/native-port.md`, ported from `py/src/llmvoice/textprep.py`) as a safety net either way.

**Batch + decode**:
```c
struct llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens); // convenience, single seq
struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(struct llama_batch batch);
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch); // 0 = success
```
Prefill the tokenized prompt with one `llama_decode` call (chunked if longer than `n_batch`); `llama_batch_get_one` is fine here since this project only needs a single sequence. Then loop: sample the next token, `llama_decode` again with a 1-token batch containing just that token, repeat.

**Sampler chain**, straight from llama.h's own usage example:
```cpp
auto sparams = llama_sampler_chain_default_params();
llama_sampler * smpl = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));  // must be last: picks the actual token
...
llama_decode(ctx, batch);
const llama_token id = llama_sampler_sample(smpl, ctx, -1);
...
llama_sampler_free(smpl);
```
`llama_sampler_sample(smpl, ctx, idx)` already fetches logits, applies the chain, picks, and accepts the token internally — no need to manage a candidate array by hand. `llama_sampler_init_penalties(...)` and `llama_sampler_init_dry(...)` exist for repetition control but aren't used by the Python prototype today, so they're optional/future rather than required for parity.

`llama-cpp.h` supplies RAII wrapper aliases worth using for cleanup safety, matching this codebase's general RAII style (`std::jthread`, `std::unique_ptr`, per `docs/native-port.md`):
```cpp
typedef std::unique_ptr<llama_model,   llama_model_deleter>   llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
```
With these as members, no manual cleanup is needed in the destructor at all (rule of zero) — the compiler-generated destructor frees model, context, and sampler automatically in reverse declaration order.

## Lifecycle: construct once, reuse across calls

Model loading reads the whole GGUF file into memory, so the model, context, and sampler chain should be constructed once in `Qwen3Backend`'s constructor and reused for the object's lifetime — not rebuilt per `synthesizeToQueue` call, the same way `Qwen3TtsBackend` builds `_context` once. Reusing the context and sampler across independent generations does mean each one needs to be reset at the start of `synthesizeToQueue` so state doesn't leak from the previous call: clear the KV cache with `llama_memory_clear(llama_get_memory(ctx), true)` (`llama.h:739`) so the new prompt isn't conditioned on the prior generation's tokens, and reset the sampler's internal state (repeat/penalty history etc.) with `llama_sampler_reset(smpl)` (`llama.h:1338`). Both are cheap compared to reconstruction.

## Python prototype's generation behavior to match

The prototype (`py/src/llmvoice/llm/llama_server.py`) doesn't embed llama.cpp in-process — it talks to a separately-run `llama-server` over OpenAI-compatible `/v1/chat/completions` with `"stream": true`, parsing SSE `data:` lines and yielding `delta.content`. Its config (`py/src/llmvoice/config.py`, `LlmConfig`) sets: a fixed system prompt sent as a `system` message alongside the user prompt; `max_tokens: int = 400`; `temperature: float = 0.7`; `top_p: float = 0.95`; no `top_k` or repetition penalty set anywhere (server defaults apply); `disable_thinking: bool = True`, sent as `chat_template_kwargs: {enable_thinking: false}` (see the chat-template caveat above for why this doesn't map 1:1 in llama.cpp's C API). A C++ sampler chain of `top_p(0.95, 1)` + `temp(0.7)` + `dist(seed)` is the direct equivalent (an optional no-op-sized `top_k` can be added since llama.cpp's own examples always chain one, but it isn't required for parity). Generation should end on `llama_vocab_is_eog(vocab, token)` or after emitting `max_tokens` tokens — there's no custom stop-string list in the prototype's config. `<think>` stripping happens in a separate downstream stage (`py/src/llmvoice/textprep.py`, a streaming state machine), not in the LLM backend itself, so `Qwen3Backend` should forward raw text deltas verbatim, `<think>` blocks included.

## Concrete shape for `Qwen3Backend.cpp`

Private members: `llama_model_ptr _model`, `llama_context_ptr _ctx`, `const llama_vocab* _vocab` (non-owning, from `llama_model_get_vocab`), `llama_sampler_ptr _sampler` (built once, top_p + temp + dist chain), plus a `std::mutex _mutex; bool _cancelled = false;` pair matching `Qwen3TtsBackend`.

Constructor: call `llama_backend_init()` once process-wide (not called anywhere in the codebase yet — needs adding, along with `llama_backend_free()` at shutdown), `llama_model_load_from_file(path, llama_model_default_params())`, `llama_model_get_vocab`, `llama_init_from_model(model, llama_context_default_params())` with `n_ctx`/`n_batch` set appropriately, then build the sampler chain.

`warmup()`: run one throwaway short generation, matching `Qwen3TtsBackend::warmup()`.

`synthesizeToQueue`: reset `_cancelled`, install a `QueueCloser`-style RAII guard on `queue`, clear the KV cache and reset the sampler (see Lifecycle above), build the `llama_chat_message[]` from system+user prompt, apply the chat template (two-call size idiom), `llama_tokenize` the formatted string, prefill via `llama_decode`, then loop: `llama_sampler_sample` → check `llama_vocab_is_eog` / the cancellation flag / the token budget → `llama_token_to_piece` → push the piece to the queue → `llama_decode` the single new token → repeat.

`cancel()`: mutex-guarded flag set, identical to `Qwen3TtsBackend`.
