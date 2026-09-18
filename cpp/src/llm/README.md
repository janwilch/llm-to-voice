# Qwen3Backend implementation notes

Research findings on the llama.cpp API surface needed to implement `Qwen3Backend.cpp`, gathered from the vendored llama.cpp source, `ILlmBackend.hpp`, the `Qwen3TtsBackend` sibling, and the Python prototype. Kept here so the implementation can be written directly against real, current signatures instead of possibly-outdated memory of the llama.cpp API.

## Vendored llama.cpp version

`cpp/CMakeLists.txt` pins llama.cpp at commit `982937a3337f7e97ef08fd5603f4157575ece7e1` (`b10909-6-g982937a33`, dated 2026-09-12) via `FetchContent`, checked out at `cpp/bin/_deps/llama_cpp-src`. This is a recent, post-sampler-rewrite, post-vocab-API-split version, so use the "modern" API (`llama_model_*`, vocab-scoped functions, sampler chains) — the old `llama_token_*`/`llama_new_context_with_model`-style functions still exist but are marked `DEPRECATED`.

llama.cpp is still built here as a subproject (`LLAMA_STANDALONE` is `OFF` and the `common` library (`common/common.h`, `common/sampling.h`, `common_sampler_*`, `common_chat_*`) is **not built or linked**).

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
int32_t llama_tokenize(const struct llama_vocab * vocab, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
int32_t llama_token_to_piece(const struct llama_vocab * vocab, llama_token token, char * buf, int32_t length, int32_t lstrip, bool special);
int32_t llama_detokenize(const struct llama_vocab * vocab, const llama_token * tokens, int32_t n_tokens, char * text, int32_t text_len_max, bool remove_special, bool unparse_special);
```
All three follow the standard two-call idiom: call with a buffer, and on a negative return re-allocate to `-result` and retry. For streaming output, call `llama_token_to_piece` per generated token and push the resulting UTF-8 fragment into the queue as one delta — this is the direct equivalent of the Python prototype's SSE `delta.content` chunks.

**Chat template**:
```c
int32_t llama_chat_apply_template(const char * tmpl, const struct llama_chat_message * chat, size_t n_msg, bool add_ass, char * buf, int32_t length);
const char * llama_model_chat_template(const struct llama_model * model, const char * name);
```
This does not use a Jinja parser — it pattern-matches a pre-defined list of known templates (Qwen's ChatML-style format is one of the built-ins). Get the model's own template via `llama_model_chat_template(model, nullptr)`, build a `llama_chat_message[]` from system+user roles, and apply the same two-call size idiom as tokenize. Rely on downstream `<think>...</think>` stripping in addition to sentence segmentation.

**Batch + decode**: `llama_batch_get_one` exists but its own doc comment says not to use it (`llama.h:944`, "this is a helper function to facilitate transition to the new batch API - avoid using it"). Build batches with `llama_batch_init` instead and fill the fields by hand:
```c
typedef struct llama_batch {
    int32_t n_tokens;
    llama_token  *  token;
    float        *  embd;
    llama_pos    *  pos;
    int32_t      *  n_seq_id;
    llama_seq_id ** seq_id;
    int8_t       *  logits;   // which positions' logits llama_decode should compute
} llama_batch;

struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(struct llama_batch batch);
int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch); // 0 = success
```

A logit is the model's raw, unnormalized per-vocab-token score at a given sequence position — one float per vocabulary entry, higher meaning the model rates that token more likely to come next. They're what a sampler consumes: the chain below (top_k/top_p/temp/dist) filters and reshapes this distribution, applies softmax, and draws a token from it. `batch.logits[i]` is the flag controlling whether `llama_decode` bothers computing that position's logits at all — only positions about to be sampled from need it (the prompt's last prefill token, then every generated token), so leaving it `false` elsewhere saves compute.

`llama_batch_init(n, 0, 1)` heap-allocates a batch sized for up to `n` tokens on a single sequence (`embd = 0` selects token-id input over raw embeddings; `n_seq_max = 1` since this project never needs multi-sequence batching). It must be released with `llama_batch_free` — unlike `llama_batch_get_one`, whose returned struct borrows the caller's token buffer and owns no heap memory of its own.

Per token `i` written into the batch (mirroring what `common_batch_add` does at `common/common.cpp:1838-1855`, unavailable here since `common/` isn't linked — see above):
```c
batch.token   [i] = id;
batch.pos     [i] = pos;        // position in the sequence — see below
batch.n_seq_id[i] = 1;
batch.seq_id  [i][0] = 0;       // this project's one sequence id
batch.logits  [i] = want_logits_for_this_token;
```

then set `batch.n_tokens` to the count filled and call `llama_decode`.

The position field is the one behavior `llama_batch_get_one` used to paper over: its doc comment notes that with it "the position of the tokens will be tracked automatically by llama_decode" (`llama.h:945`); a manually built batch gets no such tracking, so the caller must maintain its own running position counter — start at 0 for a freshly cleared KV cache, advance it by the number of tokens decoded on every `llama_decode` call (chunk size during prefill, then 1 per step in the generation loop).

Prefill the tokenized prompt across one or more `llama_decode` calls (chunked to `n_batch` tokens each, `batch.logits` true only on the final token of the final chunk since only its logits are needed to start sampling); then loop: sample the next token, refill the batch with that single token at the next position with `logits` true, `llama_decode` again, repeat.

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

**Multi-turn KV cache reuse**: `createFreshContext`'s own doc comment (`ILlmBackend.hpp:13`, "Clears existing context & memory") is the only place that should call `llama_memory_clear(llama_get_memory(ctx), true)` (`llama.h:739`) and `llama_sampler_reset(smpl)` (`llama.h:1338`) — `synthesizeToQueue` must not repeat either. Reprocessing the whole conversation from scratch every turn costs O(n) work per turn (O(n²) total over an n-turn conversation) instead of O(1) amortized, and resetting the sampler mid-conversation throws away state (repetition-penalty history etc.) that should span the whole session, not just one turn.

Track the current KV position as state that survives across `synthesizeToQueue` calls, not a local reset to 0 each call: either a persistent `llama_pos` member advanced by however many tokens get decoded, or query it directly with `llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1` (`llama.h:800`; returns `-1` on an empty sequence, so treat that as position `0` right after `createFreshContext`).

On each `synthesizeToQueue` call, append the new user message to `_history`, then re-apply `llama_chat_apply_template` (`add_ass = true`) over the whole updated history to get a new formatted string. Because llama.cpp's built-in ChatML formatting (what Qwen's template resolves to) wraps and concatenates each message independently, the previous `_lastFormatted` string is guaranteed to be a strict prefix of the new one — the only text that actually needs tokenizing and decoding is the suffix `newFormatted.substr(_lastFormatted.size())` (`add_special = false`, BOS already went in during `createFreshContext`; `parse_special = true` for the `<|im_start|>`-style markers), decoded starting at the current position, extending the existing KV cache rather than rebuilding it. After generation, append the assistant's reply to `_history` and re-run `llama_chat_apply_template` once more to refresh `_lastFormatted` for the next turn's diff.

For extra robustness beyond the string-prefix assumption (relevant only if history could ever be edited rather than strictly appended to), the same trick can run at the token level instead: tokenize the full new prompt every call (cheap — CPU-only BPE, not a GPU decode), diff it against the previously-decoded token sequence to find the longest common prefix, and trim any stale KV tail with `llama_memory_seq_rm(mem, 0, n_keep, -1)` (`llama.h:748`) before decoding from the divergence point. This is the same technique llama.cpp's own `server` tool uses for prompt-cache/slot reuse; not required for this project's always-append conversation shape, but the mechanism to reach for if that assumption ever breaks.

## Concrete shape for `Qwen3Backend.cpp`

Private members: `llama_model_ptr _model`, `llama_context_ptr _ctx`, `const llama_vocab* _vocab` (non-owning, from `llama_model_get_vocab`), `llama_sampler_ptr _sampler` (built once, top_p + temp + dist chain), `std::vector<std::pair<std::string, std::string>> _history` and `std::string _lastFormatted` (conversation so far and its last applied-template rendering, diffed each turn — see Multi-turn KV cache reuse above), a persistent `llama_pos` position counter (or none, if reading it back via `llama_memory_seq_pos_max` instead), plus a `std::mutex _mutex; bool _cancelled = false;` pair matching `Qwen3TtsBackend`.

Constructor: call `llama_backend_init()` once process-wide (not called anywhere in the codebase yet — needs adding, along with `llama_backend_free()` at shutdown), `llama_model_load_from_file(path, llama_model_default_params())`, `llama_model_get_vocab`, `llama_init_from_model(model, llama_context_default_params())` with `n_ctx`/`n_batch` set appropriately, then build the sampler chain.

`warmup()`: run one throwaway short generation, matching `Qwen3TtsBackend::warmup()`.

`createFreshContext(systemPrompt)`: the only place allowed to reset session state, per its own contract (`ILlmBackend.hpp:13`, "Clears existing context & memory") — `llama_memory_clear(llama_get_memory(ctx), true)` and `llama_sampler_reset(smpl)`, then reset `_history` to just the system message, tokenize and decode its formatted rendering from position 0 (`add_special = true` — the one call that adds BOS), and store that rendering in `_lastFormatted` as the baseline the first `synthesizeToQueue` call will diff against.

`synthesizeToQueue`: reset `_cancelled`, install a `QueueCloser`-style RAII guard on `queue`. Does **not** clear the KV cache or reset the sampler — only `createFreshContext` does that (see Multi-turn KV cache reuse above). Append `prompt` to `_history` as a user message, re-apply the chat template to the full updated history, and tokenize only the new suffix past `_lastFormatted` (`add_special = false`, `parse_special = true`). Allocate one `llama_batch` via `llama_batch_init` sized to `n_batch` (an RAII wrapper freeing it with `llama_batch_free` on scope exit, same shape as `QueueCloser`, keeps this exception-safe), and start the position counter from wherever the KV cache currently ends rather than 0. Prefill: for each `n_batch`-sized chunk of the new tokens, fill the batch fields by hand (see Batch + decode above), set `logits` true only on the chunk's last token when it's also the last chunk, `llama_decode`, advance the position by the chunk length. Then loop: `llama_sampler_sample` → check `llama_vocab_is_eog` / the cancellation flag / the token budget → `llama_token_to_piece` → push the piece to the queue → refill the batch with that one token at the current position with `logits` true, `llama_decode`, advance the position by 1 → repeat. On completion, append the generated text to `_history` as the assistant turn and re-apply the chat template once more to refresh `_lastFormatted`.

`cancel()`: mutex-guarded flag set, identical to `Qwen3TtsBackend`.
