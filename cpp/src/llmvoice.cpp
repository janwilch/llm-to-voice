#include "llmvoice.h"
#include "llm/ILlmBackend.hpp"
#include "llm/Qwen3Backend.hpp"
#include "segmenter/Segmenter.hpp"
#include "threading/BlockingQueue.hpp"
#include "tts/ITtsBackend.hpp"
#include "tts/Qwen3TtsBackend.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


// ---------------------- "private" types ----------------------

 // ReSharper disable once CppUseInternalLinkage
struct PipelineSession {
    BlockingQueue<std::string> llmTokenQueue { 64 };
    BlockingQueue<std::string> segmentsToTtsQueue { 64 };
    BlockingQueue<std::string> segmentsOutQueue { 64 };
    BlockingQueue<std::vector<float>> pcmOutQueue { 64 };

    bool segment = false;
    bool audio = false;

    std::string llmTextRemainder;
    bool llmTextStarted = false;
    
    std::jthread llmThread;
    std::jthread segmentThread;
    std::jthread ttsThread;
};

// -------------------- ABI implementation ---------------------

 // ReSharper disable once CppUseInternalLinkage
 // ReSharper disable once CppClassNeverUsed
struct LlmvoiceHandle {
    std::unique_ptr<ILlmBackend> llmBackend;
    std::unique_ptr<ITtsBackend> ttsBackend;

    std::unique_ptr<PipelineSession> session;
};

LlmvoiceHandle* llmvoiceCreate(const LlmvoiceConfig* config) {
    auto* handle = new LlmvoiceHandle {
        .llmBackend = createQwen3Backend(config->llmModelPath, config->llmContextSize),
        .ttsBackend = createQwen3TtsBackend(config->ttsTalkerPath, config->ttsCodecPath)
    };

    return handle;
}

void llmvoiceDestroy(LlmvoiceHandle* handle) {
    llmvoiceCancel(handle);
    delete handle;
}

void llmvoiceWarmup(const LlmvoiceHandle* handle) {
    handle->llmBackend->warmup();
    handle->ttsBackend->warmup();
}

void llmvoiceCreateLlmContext(const LlmvoiceHandle* handle, const char *systemPromptUtf8) {
    handle->llmBackend->createFreshContext(systemPromptUtf8);
}

void llmvoiceCreateTtsContext(const LlmvoiceHandle* handle, const long seed, const char *instructUtf8) {
    handle->ttsBackend->createFreshContext(seed, instructUtf8);
}

void llmvoiceSubmitPipeline(LlmvoiceHandle* handle, const char *promptUtf8, bool noThink) {
    // TODO
}

 // ReSharper disable once CppUseInternalLinkage
void llmvoiceSubmitLlm(LlmvoiceHandle* handle, const char *promptUtf8, const bool segment, bool noThink, int maxTokens) {
    if (handle->session) {
        throw std::runtime_error("Cannot submit a new session, while another session is in progress.");
    }

    // copy the prompt from pointer
    std::string prompt = promptUtf8;

    handle->session = std::make_unique<PipelineSession>();
    handle->session->segment = segment;

    // immediately close all queues that we don't need
    handle->session->segmentsToTtsQueue.close();
    handle->session->pcmOutQueue.close();
    if (!segment) {
        handle->session->segmentsOutQueue.close();
    }
    
    PipelineSession* session = handle->session.get();
    
    if (segment) {
        session->segmentThread = std::jthread([session]() -> void {
            Segmenter segmenter;
            segmenter.segment(session->llmTokenQueue, session->segmentsOutQueue);
        });
    }

    handle->session->llmThread = std::jthread([prompt, handle, session, noThink, maxTokens]() -> void {
        handle->llmBackend->synthesizeToQueue(prompt, session->llmTokenQueue, noThink, maxTokens);
    });
}

void llmvoiceSubmitTts(LlmvoiceHandle* handle, const char *textUtf8, bool audio) {
    // TODO
}

void llmvoiceCancel(LlmvoiceHandle* handle) {
    handle->llmBackend->cancel();
    handle->ttsBackend->cancel();

    if (!handle->session) {
        return;
    }

    handle->session->llmTokenQueue.close();
    handle->session->segmentsToTtsQueue.close();
    handle->session->segmentsOutQueue.close();
    handle->session->pcmOutQueue.close();
    handle->session.reset();
}

unsigned long llmvoicePollText(const LlmvoiceHandle* handle, char *dstUtf8, const int maxBytes) {
    if (!handle->session) {
        throw std::runtime_error("No active session to poll");
    }
    
    PipelineSession& session = *handle->session;
    BlockingQueue<std::string>& queue = session.segment
        ? session.segmentsOutQueue
        : session.llmTokenQueue;

    size_t written = 0;

    // first handle any leftover from a previous poll
    if (!session.llmTextRemainder.empty()) {
        const size_t count = std::min<size_t>(session.llmTextRemainder.size(), maxBytes);
        std::memcpy(dstUtf8, session.llmTextRemainder.data(), count);
        session.llmTextRemainder.erase(0, count);
        written += count;
    }

    while (written < maxBytes) {
        std::optional<std::string> piece = queue.pop();
        if (!piece) {
            break;
        }

        std::string text = session.segment && session.llmTextStarted
            ? " " + piece.value()
            : std::move(piece.value());

        session.llmTextStarted = true;

        size_t remaining = maxBytes - written;
        const size_t count = std::min(remaining, text.size());
        std::memcpy(dstUtf8 + written, text.data(), count);
        written += count;

        if (count < text.size()) {
            // store remainder for next poll
            session.llmTextRemainder.assign(text, count);
            break;
        }
    }
    
    return written;
}

int llmvoicePollPcm(LlmvoiceHandle* handle, float dst, int maxFrames) {
    // TODO
}

int llmvoiceIsDone(const LlmvoiceHandle* handle) {
    return handle->session == nullptr || (
        handle->session->llmTextRemainder.empty() &&
        handle->session->llmTokenQueue.empty() && handle->session->llmTokenQueue.closed() &&
        handle->session->segmentsToTtsQueue.empty() && handle->session->segmentsToTtsQueue.closed() &&
        handle->session->segmentsOutQueue.empty() && handle->session->segmentsOutQueue.closed() &&
        handle->session->pcmOutQueue.empty() && handle->session->pcmOutQueue.closed()
    );
}
