#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <miniaudio/miniaudio.h>
#include <print>
#include <string_view>
#include <thread>

#include "llmvoice.h"

namespace {

/// @brief Runs on miniaudio's audio thread, which makes it the PCM ring buffer's one consumer.
/// Must not block: no locks, allocations or printing.
// ReSharper disable once CppParameterMayBeConstPtrOrRef - must match ma_device_data_proc
void playbackCallback(ma_device* device, void* output, const void* /*input*/, const ma_uint32 frameCount) {
    auto* handle = static_cast<LlmvoiceHandle*>(device->pUserData);
    auto* out = static_cast<float*>(output);

    // whatever TTS has not produced yet plays as silence
    const size_t got = llmvoicePollPcm(handle, out, frameCount);
    std::fill(out + got, out + frameCount, 0.0f);
}

} // namespace

static void prepLlm(const LlmvoiceHandle* handle) {
    if (llmvoiceCreateLlmContext(handle, "default") != 0) {
        throw std::runtime_error(std::format("llmvoiceCreateLlmContext failed: {}", llmvoiceLastError()));
    }
}

static void prepTts(const LlmvoiceHandle* handle) {
    if (llmvoiceCreateTtsContext(handle, 42, "A mature woman with a warm and kind voice. It has a slight crackle and sounds happy. She sounds like Helen Mirren.") != 0) {
        throw std::runtime_error(std::format("llmvoiceCreateTtsContext failed: {}", llmvoiceLastError()));
    }
}

int main(const int argc, char** argv) {
    CLI::App app("llmvoice - LLM chat to speech pipeline");
    app.require_subcommand(0);

    std::string prompt;
    app.add_option("--prompt,-p", prompt, "Prompt text")->required();

    bool noThink = false;
    app.add_flag("--no-think", noThink, "Suppress the model's reasoning block (faster first response, lower answer quality)");

    // llm-only subcommand
    CLI::App* llmOnly = app.add_subcommand("llm-only", "Run only the LLM stage");
    bool skipSegmenter = false;
    llmOnly->add_flag("--skip-segmenter", skipSegmenter, "Output raw text pieces without segmentation");

    // tts-only subcommand
    CLI::App* ttsOnly = app.add_subcommand("tts-only", "Run only the TTS stage");
    bool noAudio = false;
    ttsOnly->add_flag("--no-audio", noAudio, "Only print generated chunk info; no audio output");

    CLI11_PARSE(app, argc, argv);

    // -------------- set up & run the backend --------------
    const std::string llmModel = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf";
    const std::string talkerModel = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf";
    const std::string codecModel  = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-tokenizer-12hz-Q4_K_M.gguf";

    const LlmvoiceConfig config {
        .llmModelPath = llmModel.c_str(),
        .llmContextSize = 8192,
        .ttsTalkerPath = talkerModel.c_str(),
        .ttsCodecPath = codecModel.c_str()
    };

    LlmvoiceHandle* handle = llmvoiceCreate(&config);
    if (handle == nullptr) {
        throw std::runtime_error(std::format("llmvoiceCreate failed: {}", llmvoiceLastError()));
    }

    llmvoiceSetSegmenterConfig(handle, 24, 200);

    if (llmvoiceWarmup(handle) != 0) {
        throw std::runtime_error(std::format("llmvoiceWarmup failed: {}", llmvoiceLastError()));
    }

    std::println("{}", "fully warmed up");

    if (*llmOnly) {
        prepLlm(handle);
        if (llmvoiceSubmitLlm(handle, prompt.c_str(), !skipSegmenter, noThink, 1024) != 0) {
            throw std::runtime_error(std::format("llmvoiceSubmitLlm failed: {}", llmvoiceLastError()));
        }
    }
    else if (*ttsOnly) {
        prepTts(handle);
        if (llmvoiceSubmitTts(handle, prompt.c_str()) != 0) {
            throw std::runtime_error(std::format("llmvoiceSubmitTts failed: {}", llmvoiceLastError()));
        }
    }
    else {
        prepLlm(handle);
        prepTts(handle);
        if (llmvoiceSubmitPipeline(handle, prompt.c_str(), noThink, 1024) != 0) {
            throw std::runtime_error(std::format("llmvoiceSubmitPipeline failed: {}", llmvoiceLastError()));
        }
    }

    // -------------- (audio) output --------------
    // started only after submitting: before that there is no session to poll
    const bool playAudio = !*llmOnly && !noAudio;
    ma_device device {};
    if (playAudio) {
        ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
        deviceConfig.playback.format = ma_format_f32;
        deviceConfig.playback.channels = 1;
        deviceConfig.sampleRate = LLMVOICE_PCM_SAMPLE_RATE; // miniaudio resamples if the device runs at another rate
        deviceConfig.dataCallback = playbackCallback;
        deviceConfig.pUserData = handle;

        if (const ma_result result = ma_device_init(nullptr, &deviceConfig, &device); result != MA_SUCCESS) {
            throw std::runtime_error(std::format("ma_device_init failed: {}", ma_result_description(result)));
        }

        if (const ma_result result = ma_device_start(&device); result != MA_SUCCESS) {
            ma_device_uninit(&device);
            throw std::runtime_error(std::format("ma_device_start failed: {}", ma_result_description(result)));
        }
    }

    constexpr int32_t llmBufferSize = 64;
    std::string llmBuffer(llmBufferSize, '\0');

    constexpr size_t ttsBufferSize = 64;
    std::vector<float> ttsBuffer(ttsBufferSize);

    while (!llmvoiceIsDone(handle)) {
        const size_t written = llmvoicePollText(handle, llmBuffer.data(), llmBufferSize);
        // with audio on, the playback callback is the only one allowed to read PCM
        const size_t spoken = playAudio ? 0 : llmvoicePollPcm(handle, ttsBuffer.data(), ttsBufferSize);

        if (written == 0 && spoken == 0) {
            // polling is non-blocking: wait a bit instead of spinning while nothing is ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // pieces are fragments of one text (segments are already joined with a space), so no newline in between
        std::println("| {}", std::string_view(llmBuffer.data(), written));
        std::println("| ({} PCM values)", spoken);
        std::fflush(stdout);
    }
    std::println();

    if (playAudio) {
        // done means the callback has taken all PCM out of the ring buffer, but the device's
        // own buffer still holds up to (periods * period size) frames that have not been heard yet
        const auto& playback = device.playback;
        const auto tail = std::chrono::milliseconds(
            1000ull * playback.internalPeriods * playback.internalPeriodSizeInFrames / playback.internalSampleRate);
        std::this_thread::sleep_for(tail + std::chrono::milliseconds(50));

        // before llmvoiceDestroy: the callback must not poll a destroyed handle
        ma_device_uninit(&device);
    }

    llmvoiceDestroy(handle);

    if (const std::string error = llmvoiceLastError(); !error.empty()) {
        throw std::runtime_error(error);
    }

    return 0;
}
