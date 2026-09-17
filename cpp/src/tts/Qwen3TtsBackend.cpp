#include "Qwen3TtsBackend.hpp"
#include "ITtsBackend.hpp"
#include "qwen.h"
#include <format>
#include <mutex>

class Qwen3TtsBackend : public ITtsBackend {
private:
    int64_t _seed;
    std::string _instruct;

    // One handle per loaded talker+codec GGUF pair. Aggregates talker LM weights, code predictor MTP head, optional speaker encoder, the 12Hz codec, the BPE tokenizer, and the GGML backend pair.
    qt_context* _context;
    std::mutex _mutex;
    bool _cancelled = false;

public:
    Qwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath) {
        struct qt_init_params initParams;
        qt_init_default_params(&initParams);
        
        initParams.talker_path = talkerPath.c_str();
        initParams.codec_path = codecPath.c_str();

        _context = qt_init(&initParams);
        if (_context == NULL) {
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
        qt_tts_params params;
        qt_tts_default_params(&params);
        params.text = "warmup";
        params.instruct = "default";

        qt_audio out = {0};
        qt_synthesize(_context, &params, &out);
        qt_audio_free(&out);
    }

    /// @copydoc ITtsBackend::synthesizeToQueue
    void synthesizeToQueue(const std::string& prompt, BlockingQueue<std::vector<float>>& queue) override {
        {
            std::lock_guard lock(_mutex);
            _cancelled = false;
        }

        // since there is no `finally` we use this tmp object's destroyer to clean up
        struct QueueCloser {
            BlockingQueue<std::vector<float>>& q;
            ~QueueCloser() {
                q.close();
            }
        } closer {queue};
            
        qt_tts_params params;
        qt_tts_default_params(&params);

        params.text = prompt.c_str();
        params.seed = _seed;
        params.instruct = _instruct.c_str();
        
        params.on_chunk_user_data = &queue;
        params.on_chunk = [](const float* samples, int n, void* userData) -> bool {
            BlockingQueue<std::vector<float>>* queuePtr = static_cast<BlockingQueue<std::vector<float>>*>(userData);
            // copy from first pointer target up to last (+n) pointer target into chunk
            queuePtr->push(std::vector<float>(samples, samples+n));
            return true;    // false cancels further synthesis
        };

        params.cancel_user_data = this;
        params.cancel = [](void* userData) -> bool {
            Qwen3TtsBackend* self = static_cast<Qwen3TtsBackend*>(userData);
            std::lock_guard lock(self->_mutex);
            return self->_cancelled;
        };

        qt_audio out = {0};
        qt_status status = qt_synthesize(_context, &params, &out);
        qt_audio_free(&out);

        if (status != QT_STATUS_OK) {
            throw std::runtime_error(std::format("qt_synthesize failed: {}", qt_last_error()));
        }
    }

    /// @copydoc ITtsBackend::cancel
    void cancel() override {
        std::lock_guard lock(_mutex);
        _cancelled = true;
    }
};

std::unique_ptr<ITtsBackend> createQwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath) {
    return std::make_unique<Qwen3TtsBackend>(talkerPath, codecPath);
}
