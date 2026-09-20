# `segmenter` — LLM tokens to speakable segments

This stage sits between `ILlmBackend` and `ITtsBackend`. It consumes the
`BlockingQueue<std::string>` of raw token pieces that `Qwen3Backend::synthesizeToQueue`
produces and emits a `BlockingQueue<std::string>` of complete, speakable segments
that `Qwen3TtsBackend::synthesizeToQueue` can take one at a time.

It is steps 2 and 3 of `docs/native-port.md`'s remaining work, merged into one
component because they share a buffer and a lifetime.

Nothing in here touches ggml, llama, or a GPU. It is pure string state machine
work, which makes it the one part of the pipeline that is fully unit-testable on
any machine — and the Python prototype it ports already has the tests.

## Why it exists

Three separate problems, all of which have to be solved on a *stream*, because
the point of the pipeline is that audio starts before the LLM has finished.

### 1. Token pieces are bytes, not characters

`llama_token_to_piece` writes raw UTF-8 bytes. A single code point can be split
across two tokens, so an individual piece is not necessarily valid UTF-8 — "ü",
an em dash, or an emoji can arrive as two fragments, and the first one alone is
a broken sequence.

The Python prototype never had this problem: `llama_server.py` deals in `str`,
so the HTTP layer had already reassembled everything. This job is new in the
native port and has no prototype to copy.

The segmenter must hold back an incomplete trailing sequence until the
continuation bytes arrive, emitting everything before it immediately. Lead-byte
lengths are fixed (`0xxxxxxx` = 1, `110xxxxx` = 2, `1110xxxx` = 3,
`11110xxx` = 4), so at most three bytes are ever held. Decide and document a
policy for genuinely malformed input — dropping it and replacing with U+FFFD are
both defensible; silently forwarding invalid bytes into the TTS tokenizer is not.

This has to run first, before any tag or punctuation matching, or a split code
point can hide a boundary from the stages below.

### 2. `<think>` blocks are not speech

Qwen3 emits its reasoning inside `<think>…</think>`. Leaking that into the audio
is a real defect, not a cosmetic one.

It cannot be suppressed upstream: `llama_chat_apply_template` is not a Jinja
engine, so the GGUF's real template (with its thinking-mode branches) degrades to
plain chatml and there is no `enable_thinking: false` to set. The filtering has
to happen here.

Port `py/src/llmvoice/textprep.py` — the whole state machine is ~50 lines and the
approach carries over directly:

- One flag, `inThink`, and a held-back buffer.
- Search for `<think>` when outside, `</think>` when inside; toggle on each hit.
- When the marker is not found, hold back the longest suffix of the buffer that
  is a proper prefix of the marker (`_held_prefix_len` in the prototype) and emit
  everything before it. This is what makes a tag split across two tokens work.
- On flush, release the held tail **only if outside a think block**. An
  unterminated block — a truncated or cancelled response — stays dropped. Ending
  in the middle of reasoning must not leak it.
- Reset per utterance. The prototype's docstring says "one instance per
  utterance" and means it.

**Check before implementing:** whether `<think>` and `</think>` are ordinary
vocabulary tokens or control tokens in the GGUF you ship.
`Qwen3Backend::tokenToPiece` calls `llama_token_to_piece` with `special = false`,
which renders control tokens as *empty*. If these two are control tokens, the
tags never reach the segmenter while the reasoning text between them does — the
worst possible outcome, and invisible until you hear it. Confirm against the
actual model file rather than assuming.

### 3. TTS wants sentences, not fragments

Each segment becomes an independent `qt_synthesize` call, and VoiceDesign draws a
fresh speaker per generation. A stream of short fragments is therefore a stream
of *slightly different voices*. This is the constraint that shapes every tuning
decision below, and it was settled by ear, not by theory — see the comments in
`py/src/llmvoice/config.py`, which are the real record.

Port `py/src/llmvoice/segmentation.py`'s `coalesce` directly; it is about 20
lines and translates cleanly.

**The scoping risk to be aware of up front:** `segmentation.py` is *glue*, not an
algorithm. The actual sentence splitting is done by the `stream2sentence`
package, which has no C++ equivalent. `docs/native-port.md` step 3 lists
"minimum sentence length" and the "never-split-numbers guard" as things to port,
but those are stream2sentence *parameters* — the behaviour behind them has to be
written from scratch here. Budget for that: it is the bulk of the work in this
folder, and it is larger than the "~100 lines" the port plan estimates.

What the splitter has to do:

- Break on sentence-final punctuation (`.`, `!`, `?`, `…`) followed by
  whitespace or end of input.
- **Never split inside a number.** `3.14` is one token of speech, not two
  sentences. This is `never_split_numbers` and it is non-negotiable — it is
  pinned by a test.
- Not break on common abbreviations (`Dr.`, `Mr.`, `e.g.`). In Python this came
  free from nltk's punkt data; with a rule-based splitter it needs an
  abbreviation list. The prototype already logs a warning when it degrades to
  rule-based for exactly this reason, so the degraded behaviour is understood
  and acceptable — just make the list explicit.
- Enforce a minimum segment length so a lone "Yes." does not become its own
  generation.
- Strip each segment and drop empties.
- Flush whatever is buffered at end of stream. **No text is ever dropped** —
  this is the property the completeness tests exist to guard.

Then `coalesce`: merge consecutive segments until each is at least
`coalesceMinChars` long, joining with a single space. It must be lazy — yield as
soon as the threshold is met, never buffer more than necessary — so it costs
latency only on segments that were too short to send anyway.

## Tuned constants

Carry these over with their rationale intact. They were chosen by listening, and
the reasoning is not recoverable from the numbers alone.

| Constant | Value | Why |
|---|---|---|
| `coalesceMinChars` | 60 | Six 9–21 char segments drift audibly in voice identity; three 56–65 char segments much less; one call not at all. This is also the whole time-to-first-audio cost — roughly 60 characters of LLM output before speech starts. |
| `minimumSentenceLength` | 24 | Floor before coalescing gets involved. |
| `quickYieldFirstFragment` | **off** | Speaking on the first clause is the biggest available lever on time-to-first-audio, and it is deliberately not pulled. The opener has the least context to fix a speaker, so it is precisely the segment that comes out as somebody else, and the listener hears the switch as the utterance settles. Keep the flag, keep it off. |
| `minimumFirstFragmentLength` | 12 | Inert while quick-yield is off. Keep for whoever trades identity for latency later. |
| `neverSplitNumbers` | on | `3.14`. |

Applying coalescing to the *first* segment like any other is intentional — a
short opener is exactly the one that comes out as a different voice.

## Explicit non-goals

An earlier version of the prototype stripped markdown and fenced code blocks.
Measurement removed it, and the native port should not reintroduce it:

- Markdown does not disturb segmentation. `1. Open the door. 2. Walk inside.`
  stays one segment; emphasis, inline code, parentheses and decimals all pass
  through cleanly.
- There is no evidence Qwen3-TTS is disturbed by any of it.

Everything the model writes is forwarded verbatim. `<think>` is the sole
exception, and a categorical one.

Two things stream2sentence did handle that are now unowned — decide explicitly
rather than by omission: `cleanup_text_links` (URLs) and `cleanup_text_emojis`
were both on in the prototype config. Either port them here or write down that
they were dropped.

## Shape and threading

The stage runs on its own thread between two `BlockingQueue`s, the same pattern
`pipeline.py`'s `_pump_llm` / `_pump_segments` use:

- Pull pieces until the input queue closes (`pop()` returns `nullopt`).
- Close the output queue on *every* exit path — normal, cancelled, or
  exception. `QueueCloser` in `threading/BlockingQueue.hpp` already does this.
- On end of input, flush: the UTF-8 buffer, then the think-filter tail, then the
  sentence buffer, then coalesce's pending segments. All four, in order.
- Honour `push()`'s `bool` return. `false` means the consumer closed the queue,
  which is how cancellation reaches this stage — stop immediately and discard
  buffered text. This matters for barge-in: stale buffered text becoming audio
  after the user interrupted is the exact failure the queue return value exists
  to prevent.
- A `reset()` matching `ILlmBackend::createFreshContext`, clearing all four
  buffers and the `inThink` flag.

Keeping the segmenter pull-driven gives the pipeline its backpressure: it
consumes LLM tokens only as fast as TTS consumes segments.

## Testing

`py/tests/test_textprep.py` and `py/tests/test_segmentation.py` are the
specification. They assert the behaviour this project depends on rather than
the library's internals, so they port even though the implementation does not.
At minimum, pin:

**Think-stripping** — block removed; multiple blocks mid-text; unterminated block
dropped; result independent of chunk size; tags split across chunk boundaries; a
partial tag held rather than emitted; held partial released on flush; empty
input.

**Segmentation** — no text is lost; trailing fragment is flushed; a single short
input survives; multiple sentences yield multiple segments; short sentences merge
into one segment under the default config; `3.14` is not split; every segment is
stripped and non-empty.

**Coalescing** — merges until the threshold; flushes the tail; disabled at
`min_chars <= 0`; leaves already-long segments alone; merges the first segment
like any other; is lazy (does not consume the whole source to yield the first
result); handles an empty source.

Add UTF-8 cases, which have no Python precedent: a code point split across two
pieces, a 4-byte emoji split at each of its three interior boundaries, a stream
ending mid-sequence, and malformed input.

## Build

`cpp/CMakeLists.txt` lists sources explicitly rather than globbing, so new
`.cpp` files here need adding to the `add_executable(llmvoice_cpp …)` call
around line 149.
