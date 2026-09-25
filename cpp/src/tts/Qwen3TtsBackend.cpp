#include "Qwen3TtsBackend.hpp"
#include "ITtsBackend.hpp"
#include "qwen.h"

#include "../threading/SpscRingBuffer.hpp"

#include <format>
#include <mutex>
#include <thread>

namespace {
class Qwen3TtsBackend : public ITtsBackend {
    int64_t _seed = 0;
    std::string _instruct;

    // One handle per loaded talker+codec GGUF pair. Aggregates talker LM weights, code predictor MTP head, optional speaker encoder, the 12Hz codec, the BPE tokenizer, and the GGML backend pair.
    qt_context* _context;
    std::mutex _mutex;
    std::atomic<bool> _cancelled = { false };

public:
    Qwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath) {
        qt_init_params initParams{};
        qt_init_default_params(&initParams);

        initParams.talker_path = talkerPath.c_str();
        initParams.codec_path = codecPath.c_str();

        _context = qt_init(&initParams);
        if (_context == nullptr) {
            throw std::runtime_error(std::format("qt_init failed: {}", qt_last_error()));
        }
    }

    ~Qwen3TtsBackend() override {
        qt_free(_context);
    }

    /// @copydoc ITtsBackend::createFreshContext
    void createFreshContext(const int64_t seed, const std::string& instruct) override {
        _seed = seed;
        _instruct = instruct;
    }

    /// @copydoc ITtsBackend::warmup
    void warmup() override {
        qt_tts_params params{};
        qt_tts_default_params(&params);
        params.text = "This is a short warmup sentence. It is long enough to produce several chunks of audio.";
        params.instruct = "A calm, neutral voice.";

        // non-null on_chunk selects the streaming pipeline, same as synthesizeToBuffer
        params.on_chunk = [](const float*, int, void*) -> bool { return true; };

        qt_audio out = {.samples = nullptr};
        const qt_status status = qt_synthesize(_context, &params, &out);
        qt_audio_free(&out);

        if (status != QT_STATUS_OK) {
            throw std::runtime_error(std::format("warmup qt_synthesize failed: {}", qt_last_error()));
        }
    }

    /// @copydoc ITtsBackend::synthesizeToBuffer
    void synthesizeToBuffer(const std::string& text, SpscRingBuffer<float>& buffer) override {
        qt_tts_params params{};
        qt_tts_default_params(&params);

        params.text = text.c_str();
        params.seed = _seed;
        params.instruct = _instruct.c_str();

        struct UserData {
            SpscRingBuffer<float>& b;
            std::atomic<bool>& cancelled;
        } workData {
            .b = buffer,
            .cancelled = _cancelled
        };

        params.on_chunk_user_data = &workData;
        params.on_chunk = [](const float* samples, const int n, void* userData) -> bool {
            const auto* work = static_cast<UserData*>(userData);

            size_t done = 0;
            while (done < static_cast<size_t>(n)) {
                if (work->cancelled.load()) {
                    return false; // stop synthesis
                }

                done += work->b.write(samples + done, n - done);
                if (done < static_cast<size_t>(n)) {
                    // TTS doesn't need to run in real-time - sleep while buffer is full
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }

            return true;
        };

        params.cancel_user_data = this;
        params.cancel = [](void* userData) -> bool {
            const auto self = static_cast<Qwen3TtsBackend*>(userData);
            std::lock_guard lock(self->_mutex);
            return self->_cancelled;
        };

        qt_audio out = {.samples = nullptr};
        const qt_status status = qt_synthesize(_context, &params, &out);
        qt_audio_free(&out);

        if (status != QT_STATUS_OK) {
            throw std::runtime_error(std::format("qt_synthesize failed: {}", qt_last_error()));
        }
    }

    /// @copydoc ITtsBackend::cancel
    void cancel() override {
        _cancelled.store(true);
    }

    /// @copydoc ITtsBackend::resetCancel
    void resetCancel() override {
        _cancelled.store(false);
    }
};
}

std::unique_ptr<ITtsBackend> createQwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath) {
    return std::make_unique<Qwen3TtsBackend>(talkerPath, codecPath);
}
