#pragma once

// ReSharper disable CppUnusedIncludeDirective
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define LLMVOICE_PCM_SAMPLE_RATE 24000

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
LlmvoiceHandle* llmvoiceCreate(const LlmvoiceConfig*);

/// @brief Free the llmvoice backend. NULL is a no-op.
void llmvoiceDestroy(LlmvoiceHandle*);

/// @brief Warm up the llmvoice backend (load models and run a throwaway generation).
int llmvoiceWarmup(const LlmvoiceHandle*);

/// @brief Create a fresh LLM context for generation.
/// Call before submitting to LLM or pipeline.
int llmvoiceCreateLlmContext(const LlmvoiceHandle*, const char* systemPromptUtf8);

/// @brief Create a fresh TTS context for generation.
/// Call before submitting to TTS or pipeline.
int llmvoiceCreateTtsContext(const LlmvoiceHandle*, int64_t seed, const char* instructUtf8);

/// @brief Submit a prompt to run through the whole pipeline (LLM -> segmenter -> TTS).
void llmvoiceSubmitPipeline(LlmvoiceHandle*, const char* promptUtf8, bool noThink);

/// @brief Submit a prompt to run only through the LLM and optionally the segmenter.
/// Only `pollText` will yield output.
/// @return 0 on success, non-0 otherwise.
int llmvoiceSubmitLlm(LlmvoiceHandle*, const char* promptUtf8, bool segment, bool noThink, int maxTokens);

/// @brief Submit a text to run only through TTS.
/// Only `pollPcm` will yield output.
/// @return 0 on success, non-0 otherwise.
int llmvoiceSubmitTts(LlmvoiceHandle*, const char* textUtf8, bool audio);

/// @brief Cancel all running generation.
void llmvoiceCancel(LlmvoiceHandle*);

/// @brief Poll generated output from the LLM stage (optionally segmented).
/// @return Number of bytes written.
size_t llmvoicePollText(LlmvoiceHandle*, char* dstUtf8, size_t maxBytes);

/// @brief Poll generated output from the TTS stage.
/// Mono float32 PCM at LLMVOICE_PCM_SAMPLE_RATE, range -1.0..1.0 (0 = silence).
/// @return Number of frames written.
size_t llmvoicePollPcm(LlmvoiceHandle*, float* dst, size_t maxFrames);

/// @brief Returns true if all generation completed and all output was polled.
/// Once true, `llmvoiceLastError` holds the reason if a stage failed, or "" if all succeeded.
/// A finished session is replaced by the next submit. Call from the thread that polls.
bool llmvoiceIsDone(LlmvoiceHandle*);

/// @brief Human-readable reason for the most recent failure on this thread.
/// Valid until the next llmvoice call on this thread.
const char* llmvoiceLastError(void);

#ifdef __cplusplus
}
#endif
