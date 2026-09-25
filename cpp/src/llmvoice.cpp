#include "llmvoice.h"
#include "llm/ILlmBackend.hpp"
#include "llm/Qwen3Backend.hpp"
#include "segmenter/Segmenter.hpp"
#include "threading/BlockingQueue.hpp"
#include "threading/SpscRingBuffer.hpp"
#include "tts/ITtsBackend.hpp"
#include "tts/Qwen3TtsBackend.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

constexpr size_t PCM_BUFFER_SECONDS = 30;

// ---------------------- "private" types ----------------------

 // ReSharper disable once CppUseInternalLinkage
struct PipelineSession {
    BlockingQueue<std::string> llmTokenQueue { DEFAULT_QUEUE_CAPA };
    BlockingQueue<std::string> segmentsToTtsQueue { DEFAULT_QUEUE_CAPA };
    BlockingQueue<std::string> segmentsOutQueue { DEFAULT_QUEUE_CAPA };

    bool segment = false;

    std::string llmTextRemainder;
    bool llmTextStarted = false;

    SpscRingBuffer<float> pcmOut { LLMVOICE_PCM_SAMPLE_RATE * PCM_BUFFER_SECONDS };

    // the session is done once no worker is running anymore and all outputs are drained
    std::atomic<int> runningWorkers { 0 };

    // first failure of any worker thread, handed to the caller by llmvoiceIsDone
    std::mutex errorMutex;
    std::string error;

    void fail(std::string message) {
        std::lock_guard lock(errorMutex);
        if (error.empty()) {
            error = std::move(message);
        }
    }

    // declared last, so they are destroyed (= joined) first, while everything they use still exists
    std::jthread llmThread;
    std::jthread segmentThread;
    std::jthread ttsThread;
};

 // ReSharper disable once CppUseInternalLinkage
 // ReSharper disable once CppClassNeverUsed
struct LlmvoiceHandle {
    std::unique_ptr<ILlmBackend> llmBackend;
    std::unique_ptr<ITtsBackend> ttsBackend;

    SegmenterConfig segmenterConfig;

    std::unique_ptr<PipelineSession> session;
};

// ---------------------- "private" helpers ----------------------

// errno-style: set by a failing API call, read by llmvoiceLastError on the same thread
thread_local std::string lastError;

static void setError(std::string message) {
    lastError = std::move(message);
}

/// @brief Runs `body`, converting any exception into the thread-local error, since exceptions must not cross the C ABI.
/// @return `true` if `body` completed without throwing.
template <typename F>
static bool guarded(F&& body) noexcept {
    try {
        body();
        return true;
    }
    catch (const std::exception& ex) {
        setError(ex.what());
    }
    catch (...) {
        setError("unknown error");
    }
    return false;
}

/// @brief Starts a session worker thread. It counts as running until `work` returns; exceptions are recorded as the session error instead of terminating the process.
template <typename F>
static std::jthread startWorker(PipelineSession* session, const char* stage, F work) {
    // count *before* starting, so llmvoiceIsDone can never see 0 while the thread is still starting up
    session->runningWorkers.fetch_add(1);

    try {
        return std::jthread([session, stage, work = std::move(work)]() mutable -> void {
            try {
                work();
            }
            catch (const std::exception& ex) {
                session->fail(std::format("{} failed: {}", stage, ex.what()));
            }
            catch (...) {
                session->fail(std::format("{} failed: unknown error", stage));
            }

            // release: everything this worker wrote is visible to whoever sees the decrement
            session->runningWorkers.fetch_sub(1, std::memory_order_release);
        });
    }
    catch (...) {
        session->runningWorkers.fetch_sub(1);
        throw;
    }
}

static BlockingQueue<std::string>& textOutputQueue(PipelineSession& session) {
    return session.segment
        ? session.segmentsOutQueue
        : session.llmTokenQueue;
}

static bool isSessionDone(PipelineSession& session) {
    // workers first: once none is running, nothing can be added to the outputs anymore, so the empty checks are final
    return session.runningWorkers.load(std::memory_order_acquire) == 0
        && session.llmTextRemainder.empty()
        && textOutputQueue(session).empty()
        && session.pcmOut.available() == 0;
}

/// @brief Stops all workers of the current session and destroys it. Blocks until all worker threads have exited.
static void cancelSession(LlmvoiceHandle* handle) {
    handle->llmBackend->cancel();
    handle->ttsBackend->cancel();

    if (!handle->session) {
        return;
    }

    // unblock workers waiting on a queue; the backends' cancel flags stop the rest
    handle->session->llmTokenQueue.close();
    handle->session->segmentsToTtsQueue.close();
    handle->session->segmentsOutQueue.close();
    handle->session.reset();
}

/// @brief Replaces a finished session with a fresh one.
/// @throws std::runtime_error if a session is still in progress.
static PipelineSession* beginSession(LlmvoiceHandle* handle) {
    if (handle->session) {
        if (!isSessionDone(*handle->session)) {
            throw std::runtime_error("Cannot submit a new session while another session is in progress");
        }
        handle->session.reset();
    }

    // no worker is running here, so no synthesis can race the reset
    handle->llmBackend->resetCancel();
    handle->ttsBackend->resetCancel();

    handle->session = std::make_unique<PipelineSession>();
    return handle->session.get();
}

/// @brief Starts the segmenter worker, which turns the session's LLM tokens into segments pushed to `outQueue`.
static std::jthread startSegmenter(LlmvoiceHandle* handle, PipelineSession* session, BlockingQueue<std::string>& outQueue) {
    // copied, so a later llmvoiceSetSegmenterConfig does not race this session
    return startWorker(session, "segmentation", [handle, session, &outQueue, config = handle->segmenterConfig] {
        try {
            Segmenter segmenter;
            segmenter.segment(session->llmTokenQueue, outQueue, config);
        }
        catch (...) {
            // nobody drains the token queue anymore: stop the LLM instead of letting it block on a full queue
            handle->llmBackend->cancel();
            session->llmTokenQueue.close();
            throw;
        }
    });
}

// -------------------- ABI implementation ---------------------

LlmvoiceHandle* llmvoiceCreate(const LlmvoiceConfig* config) {
    LlmvoiceHandle* handle = nullptr;

    guarded([&] {
        handle = new LlmvoiceHandle {
            .llmBackend = createQwen3Backend(config->llmModelPath, config->llmContextSize),
            .ttsBackend = createQwen3TtsBackend(config->ttsTalkerPath, config->ttsCodecPath)
        };
    });

    return handle;
}

void llmvoiceDestroy(LlmvoiceHandle* handle) {
    if (handle == nullptr) {
        return;
    }

    guarded([&] { cancelSession(handle); });
    delete handle;
}

int llmvoiceWarmup(const LlmvoiceHandle* handle) {
    return guarded([&] {
        handle->llmBackend->warmup();
        handle->ttsBackend->warmup();
    }) ? 0 : 1;
}

int llmvoiceCreateLlmContext(const LlmvoiceHandle* handle, const char *systemPromptUtf8) {
    return guarded([&] {
        handle->llmBackend->createFreshContext(systemPromptUtf8);
    }) ? 0 : 1;
}

int llmvoiceCreateTtsContext(const LlmvoiceHandle* handle, const int64_t seed, const char *instructUtf8) {
    return guarded([&] {
        handle->ttsBackend->createFreshContext(seed, instructUtf8);
    }) ? 0 : 1;
}

void llmvoiceSetSegmenterConfig(LlmvoiceHandle* handle, const size_t minSentenceLength, const size_t coalesceMinChars) {
    handle->segmenterConfig = {
        .minSentenceLength = minSentenceLength,
        .coalesceMinChars = coalesceMinChars
    };
}

int llmvoiceSubmitPipeline(LlmvoiceHandle* handle, const char *promptUtf8, const bool noThink, const int maxTokens) {
    if (promptUtf8 == nullptr) {
        setError("prompt must not be null");
        return 1;
    }

    const bool ok = guarded([&] {
        PipelineSession* session = beginSession(handle);
        session->segment = true;

        try {
            session->llmThread = startWorker(session, "LLM generation", [handle, session, prompt = std::string(promptUtf8), noThink, maxTokens] {
                handle->llmBackend->synthesizeToQueue(prompt, session->llmTokenQueue, noThink, maxTokens);
            });

            session->segmentThread = startSegmenter(handle, session, session->segmentsToTtsQueue);

            session->ttsThread = startWorker(session, "TTS", [handle, session] {
                try {
                    while (std::optional<std::string> next = session->segmentsToTtsQueue.pop()) {
                        // speak and output the segment simultaneously (basically "closed captioning")
                        if (!session->segmentsOutQueue.push(next.value())) {
                            return; // closed = cancelled
                        }

                        handle->ttsBackend->synthesizeToBuffer(next.value(), session->pcmOut);
                    }
                }
                catch (...) {
                    // nobody drains the segments anymore: stop everything upstream instead of letting it block on full queues
                    handle->llmBackend->cancel();
                    session->llmTokenQueue.close();
                    session->segmentsToTtsQueue.close();
                    throw;
                }
            });
        }
        catch (...) {
            cancelSession(handle);
            throw;
        }
    });

    return ok ? 0 : 1;
}

 // ReSharper disable once CppUseInternalLinkage
int llmvoiceSubmitLlm(LlmvoiceHandle* handle, const char *promptUtf8, const bool segment, const bool noThink, const int maxTokens) {
    if (promptUtf8 == nullptr) {
        setError("prompt must not be null");
        return 1;
    }

    const bool ok = guarded([&] {
        PipelineSession* session = beginSession(handle);
        session->segment = segment;

        try {
            if (segment) {
                session->segmentThread = startSegmenter(handle, session, session->segmentsOutQueue);
            }

            session->llmThread = startWorker(session, "LLM generation", [handle, session, prompt = std::string(promptUtf8), noThink, maxTokens] {
                handle->llmBackend->synthesizeToQueue(prompt, session->llmTokenQueue, noThink, maxTokens);
            });
        }
        catch (...) {
            // a thread failed to start: don't leave the others running in a half-built session
            cancelSession(handle);
            throw;
        }
    });

    return ok ? 0 : 1;
}

int llmvoiceSubmitTts(LlmvoiceHandle* handle, const char *textUtf8) {
    if (textUtf8 == nullptr) {
        setError("text must not be null");
        return 1;
    }

    const bool ok = guarded([&] {
        PipelineSession* session = beginSession(handle);

        try {
            session->ttsThread = startWorker(session, "TTS", [handle, session, text = std::string(textUtf8)] {
                handle->ttsBackend->synthesizeToBuffer(text, session->pcmOut);
            });
        }
        catch (...) {
            cancelSession(handle);
            throw;
        }
    });

    return ok ? 0 : 1;
}

void llmvoiceCancel(LlmvoiceHandle* handle) {
    guarded([&] { cancelSession(handle); });
}

 // ReSharper disable once CppUseInternalLinkage
 // ReSharper disable once CppParameterMayBeConstPtrOrRef
size_t llmvoicePollText(LlmvoiceHandle* handle, char *dstUtf8, const size_t maxBytes) {
    if (!handle->session) {
        setError("No active session to poll");
        return 0;
    }

    PipelineSession& session = *handle->session;
    BlockingQueue<std::string>& queue = textOutputQueue(session);

    size_t written = 0;

    // first handle any leftover from a previous poll
    if (!session.llmTextRemainder.empty()) {
        const size_t count = std::min<size_t>(session.llmTextRemainder.size(), maxBytes);
        std::memcpy(dstUtf8, session.llmTextRemainder.data(), count);
        session.llmTextRemainder.erase(0, count);
        written += count;
    }

    std::string piece;
    while (written < maxBytes && queue.tryPop(piece)) {
        std::string text = session.segment && session.llmTextStarted
            ? " " + piece
            : std::move(piece);

        session.llmTextStarted = true;

        const size_t remaining = maxBytes - written;
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

 // ReSharper disable once CppUseInternalLinkage
 // ReSharper disable once CppParameterMayBeConstPtrOrRef
size_t llmvoicePollPcm(LlmvoiceHandle* handle, float* dst, const size_t maxFrames) {
    if (!handle->session) {
        setError("No active session to poll");
        return 0;
    }

    return handle->session->pcmOut.read(dst, maxFrames);
}

 // ReSharper disable once CppUseInternalLinkage
 // ReSharper disable once CppParameterMayBeConstPtrOrRef
bool llmvoiceIsDone(LlmvoiceHandle* handle) {
    PipelineSession* session = handle->session.get();
    if (session == nullptr) {
        return true;
    }

    if (!isSessionDone(*session)) {
        return false;
    }

    // hand the outcome to the caller: empty if every stage succeeded
    std::lock_guard lock(session->errorMutex);
    setError(session->error);
    return true;
}

const char* llmvoiceLastError(void) {
    return lastError.c_str();
}
