#pragma once

// ReSharper disable CppUnusedIncludeDirective
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LLMVOICE_PCM_SAMPLE_RATE 24000
#define DEFAULT_QUEUE_CAPA 64

// LLMVOICE_BUILDING is defined only while compiling the llmvoice library itself.
#if defined(_WIN32)
#  ifdef LLMVOICE_BUILDING
#    define LLMVOICE_API __declspec(dllexport)
#  else
#    define LLMVOICE_API __declspec(dllimport)
#  endif
#else
#  define LLMVOICE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmvoiceHandle LlmvoiceHandle;

typedef struct {
    const char* llmModelPath;
    int llmContextSize;

    const char* ttsTalkerPath;
    const char* ttsCodecPath;
} LlmvoiceConfig;

/// @brief Initialize the llmvoice backend.
/// @return NULL on failure, see `llmvoiceLastError`.
LLMVOICE_API LlmvoiceHandle* llmvoiceCreate(const LlmvoiceConfig*);

/// @brief Free the llmvoice backend. NULL is a no-op.
LLMVOICE_API void llmvoiceDestroy(LlmvoiceHandle*);

/// @brief Warm up the llmvoice backend (load models and run a throwaway generation).
LLMVOICE_API int llmvoiceWarmup(const LlmvoiceHandle*);

/// @brief Create a fresh LLM context for generation.
/// Call before submitting to LLM or pipeline.
LLMVOICE_API int llmvoiceCreateLlmContext(const LlmvoiceHandle*, const char* systemPromptUtf8);

/// @brief Create a fresh TTS context for generation.
/// Call before submitting to TTS or pipeline.
LLMVOICE_API int llmvoiceCreateTtsContext(const LlmvoiceHandle*, int64_t seed, const char* instructUtf8);

/// @brief Configure how LLM output is split into segments, from the next submit on.
/// @param minSentenceLength Sentence ends before this many bytes are ignored (default 24).
/// @param coalesceMinChars Sentences are merged until a segment has this many bytes; 0 disables merging (default 60).
LLMVOICE_API void llmvoiceSetSegmenterConfig(LlmvoiceHandle*, size_t minSentenceLength, size_t coalesceMinChars);

/// @brief Submit a prompt to run through the whole pipeline (LLM -> segmenter -> TTS).
/// Both `pollText` and `pollPcm` must be polled: if DEFAULT_QUEUE_CAPA segments of text are unread, TTS stalls until `pollText` is called.
/// @return 0 on success, non-0 otherwise.
LLMVOICE_API int llmvoiceSubmitPipeline(LlmvoiceHandle*, const char* promptUtf8, bool noThink, int maxTokens);

/// @brief Submit a prompt to run only through the LLM and optionally the segmenter.
/// Only `pollText` will yield output.
/// @return 0 on success, non-0 otherwise.
LLMVOICE_API int llmvoiceSubmitLlm(LlmvoiceHandle*, const char* promptUtf8, bool segment, bool noThink, int maxTokens);

/// @brief Submit a text to run only through TTS.
/// Only `pollPcm` will yield output.
/// @return 0 on success, non-0 otherwise.
LLMVOICE_API int llmvoiceSubmitTts(LlmvoiceHandle*, const char* textUtf8);

/// @brief Cancel all running generation.
LLMVOICE_API void llmvoiceCancel(LlmvoiceHandle*);

/// @brief Poll generated output from the LLM stage (optionally segmented).
/// @return Number of bytes written.
LLMVOICE_API size_t llmvoicePollText(LlmvoiceHandle*, char* dstUtf8, size_t maxBytes);

/// @brief Poll generated output from the TTS stage.
/// Mono float32 PCM at LLMVOICE_PCM_SAMPLE_RATE, range -1.0..1.0 (0 = silence).
/// @return Number of frames written.
LLMVOICE_API size_t llmvoicePollPcm(LlmvoiceHandle*, float* dst, size_t maxFrames);

/// @brief Returns true if all generation completed and all output was polled.
/// Once true, `llmvoiceLastError` holds the reason if a stage failed, or "" if all succeeded.
/// A finished session is replaced by the next submit. Call from the thread that polls.
LLMVOICE_API bool llmvoiceIsDone(LlmvoiceHandle*);

/// @brief Human-readable reason for the most recent failure on this thread.
/// Valid until the next llmvoice call on this thread.
LLMVOICE_API const char* llmvoiceLastError(void);

#ifdef __cplusplus
}
#endif
