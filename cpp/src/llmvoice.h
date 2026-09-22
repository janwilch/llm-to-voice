#pragma once

typedef struct LlmvoiceHandle LlmvoiceHandle;

typedef struct {
    const char* llmModelPath;
    int llmContextSize;
    
    const char* ttsTalkerPath;
    const char* ttsCodecPath;
} LlmvoiceConfig;

/// @brief Initialize the llmvoice backend.
LlmvoiceHandle* llmvoiceCreate(const LlmvoiceConfig*);

/// @brief Free the llmvoice backend.
void llmvoiceDestroy(LlmvoiceHandle*);

/// @brief Warm up the llmvoice backend (load models and run a throwaway generation).
void llmvoiceWarmup(LlmvoiceHandle*);

/// @brief Create a fresh LLM context for generation.
/// Call before submitting to LLM or pipeline.
void llmvoiceCreateLlmContext(LlmvoiceHandle*, const char* systemPromptUtf8);

/// @brief Create a fresh TTS context for generation.
/// Call before submitting to TTS or pipeline.
void llmvoiceCreateTtsContext(LlmvoiceHandle*, const long seed, const char* instructUtf8);

/// @brief Submit a prompt to run through the whole pipeline (LLM -> segmenter -> TTS).
void llmvoiceSubmitPipeline(LlmvoiceHandle*, const char* promptUtf8, bool noThink = false);

/// @brief Submit a prompt to run only through the LLM and optionally the segmenter.
/// Only `pollText` will yield output.
void llmvoiceSubmitLlm(LlmvoiceHandle*, const char* promptUtf8, bool segment = true, bool noThink = false);

/// @brief Submit a text to run only through TTS.
/// Only `pollPcm` will yield output.
void llmvoiceSubmitTts(LlmvoiceHandle*, const char* textUtf8, bool audio = true);

/// @brief Cancel all running generation.
void llmvoiceCancel(LlmvoiceHandle*);

/// @brief Poll generated output from the LLM stage (optionally segmented).
int llmvoicePollText(LlmvoiceHandle*, char* dstUtf8, int maxBytes);

/// @brief Poll generated output from the TTS stage.
int llmvoicePollPcm(LlmvoiceHandle*, float dst, int maxFrames);

/// @brief Returns 1/true if all generation completed.
int llmvoiceIsDone(LlmvoiceHandle*);
