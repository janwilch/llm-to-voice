#include "../helpers/EnumToString.hpp"
#include "../helpers/FileTools.hpp"
#include "../threading/SpscRingBuffer.hpp"
#include "ITtsBackend.hpp"
#include "Qwen3TtsBackend.hpp"

#include "qwen.h"

// NOTE - this is an internal qwen-tts.cpp header; could have breaking changes between versions
#include "audio-io.h"

#include <filesystem>
#include <format>
#include <mutex>
#include <ostream>
#include <thread>

namespace {
class Qwen3TtsBackend : public ITtsBackend {
    GenerationType _mode;

    // for VoiceDesign mode
    int64_t _seed = 0;
    std::string _instruct;

    // for Base mode
    int32_t _refAudioSamples = 0;
    std::vector<float> _refAudio;
    std::string _transcript;

    // One handle per loaded talker+codec GGUF pair. Aggregates talker LM weights, code predictor MTP head, optional speaker encoder, the 12Hz codec, the BPE tokenizer, and the GGML backend pair.
    qt_context* _context;
    std::mutex _mutex;
    std::atomic<bool> _cancelled = { false };

    qt_tts_params getTtsParams(const char* text) {
        qt_tts_params params {};
        qt_tts_default_params(&params);

        params.text = text;

        if (_mode == GenerationType::VoiceDesign) {
            params.seed = _seed;
            params.instruct = _instruct.c_str();
        }
        else if (_mode == GenerationType::Base) {
            params.ref_n_samples = _refAudioSamples;
            params.ref_audio_24k = _refAudio.data();
            params.ref_text = _transcript.c_str();
        }
        else {
            throw std::runtime_error(std::format("unsupported mode {}", _mode));
        }

        return params;
    }

public:
    Qwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath, const GenerationType mode) {
        _mode = mode;

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

    /// @copydoc ITtsBackend::createVoiceDesignContext
    void createVoiceDesignContext(const int64_t seed, const std::string& instruct) override {
        if (_mode != GenerationType::VoiceDesign) {
            throw std::runtime_error("must set voice design mode during construction");
        }

        _seed = seed;
        _instruct = instruct;
    }

    /// @copydoc ITtsBackend::createBaseContext
    void createBaseContext(const std::string &wavPath, const std::string &transcriptPath) override {
        if (_mode != GenerationType::Base) {
            throw std::runtime_error("must set base mode during construction");
        }

        const std::string wavNotExists = std::format("sample WAV does not exist or is not readable: {}", wavPath);
        if (std::filesystem::exists(wavPath)) {
            _refAudioSamples = 0;
            if (float* raw = audio_read_mono(wavPath.c_str(), 24000, &_refAudioSamples)) {
                _refAudio.assign(raw, raw + _refAudioSamples);
                free(raw);
            }
            else {
                throw std::runtime_error(wavNotExists);
            }
        }
        else {
            throw std::runtime_error(wavNotExists);
        }

        if (std::filesystem::exists(transcriptPath)) {
            _transcript = readFileText(transcriptPath);
        }
        else {
            throw std::runtime_error(std::format("sample transcript does not exist or is not readable: {}", transcriptPath));
        }
    }

    /// @copydoc ITtsBackend::warmup
    void warmup() override {
        qt_tts_params params = getTtsParams("This is a short warmup sentence. It is long enough to produce several chunks of audio.");

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
        qt_tts_params params = getTtsParams(text.c_str());

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

std::unique_ptr<ITtsBackend> createQwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath, const GenerationType mode) {
    return std::make_unique<Qwen3TtsBackend>(talkerPath, codecPath, mode);
}
