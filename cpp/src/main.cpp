#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <miniaudio/miniaudio.h>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <thread>

#include "llmvoice.h"
#include "benchmark/Benchmark.hpp"
#include "helpers/FileTools.hpp"

namespace {

void prepLlm(const LlmvoiceHandle* handle) {
    if (llmvoiceCreateLlmContext(handle, "default") != 0) {
        throw std::runtime_error(std::format("llmvoiceCreateLlmContext failed: {}", llmvoiceLastError()));
    }
}

void prepTtsVoiceDesign(const LlmvoiceHandle* handle, const int64_t seed, const std::string &instruct) {
    if (llmvoiceCreateTtsVoiceDesignContext(handle, seed, instruct.c_str()) != 0) {
        throw std::runtime_error(std::format("llmvoiceCreateTtsVoiceDesignContext failed: {}", llmvoiceLastError()));
    }
}

void prepTtsVoiceClone(const LlmvoiceHandle* handle, const std::string &wavPath, const std::string &transcriptPath) {
    if (llmvoiceCreateTtsBaseContext(handle, wavPath.c_str(), transcriptPath.c_str()) != 0) {
        throw std::runtime_error(std::format("llmvoiceCreateTtsBaseContext failed: {}", llmvoiceLastError()));
    }
}

/// @brief What the audio thread needs; must outlive the device.
struct PlaybackContext {
    LlmvoiceHandle* handle;
    /// @brief Set on the first PCM the device receives (time to first audio).
    bench::Mark& firstPcm;
};

/// @brief Runs on miniaudio's audio thread, consuming the PCM ring buffer.
// ReSharper disable once CppParameterMayBeConstPtrOrRef
void playbackCallback(ma_device* device, void* output, const void* /*input*/, const ma_uint32 frameCount) {
    const auto* context = static_cast<PlaybackContext*>(device->pUserData);
    auto* out = static_cast<float*>(output);

    const size_t got = llmvoicePollPcm(context->handle, out, frameCount);
    if (got > 0) {
        context->firstPcm.hit();
    }
    std::fill(out + got, out + frameCount, 0.0f);
}

void initAudioDevice(PlaybackContext& context, ma_device& device) {
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format = ma_format_f32;
    deviceConfig.playback.channels = 1;
    deviceConfig.sampleRate = LLMVOICE_PCM_SAMPLE_RATE; // miniaudio resamples if the device runs at another rate
    deviceConfig.dataCallback = playbackCallback;
    deviceConfig.pUserData = &context;

    if (const ma_result result = ma_device_init(nullptr, &deviceConfig, &device); result != MA_SUCCESS) {
        throw std::runtime_error(std::format("ma_device_init failed: {}", ma_result_description(result)));
    }

    if (const ma_result result = ma_device_start(&device); result != MA_SUCCESS) {
        ma_device_uninit(&device);
        throw std::runtime_error(std::format("ma_device_start failed: {}", ma_result_description(result)));
    }
}

/// @brief There could still be some PCM chunks in the buffer, after generation ends.
void playbackAudioRemainder(ma_device& device) {
    const auto& playback = device.playback;
    const auto tail = std::chrono::milliseconds(
        1000ull * playback.internalPeriods * playback.internalPeriodSizeInFrames / playback.internalSampleRate);
    std::this_thread::sleep_for(tail + std::chrono::milliseconds(50));

    ma_device_uninit(&device);
}

struct MonoWavEncoderContext {
    std::filesystem::path path;
    ma_encoder encoder;
    size_t samplesWritten = 0;

    MonoWavEncoderContext(const std::filesystem::path &path, const ma_uint32 sampleRate) : path(path) {
        const ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16, 1, sampleRate);
#ifdef _WIN32
        const ma_result initResult = ma_encoder_init_file_w(path.c_str(), &config, &encoder);
#else
        const ma_result initResult = ma_encoder_init_file(path.c_str(), &config, &encoder);
#endif

        if (initResult != MA_SUCCESS) {
            throw std::runtime_error(std::format("cannot open {}: {}", path.string(), ma_result_description(initResult)));
        }
    }

    // prevent copies (i.e. double file handle and uninit)
    MonoWavEncoderContext(const MonoWavEncoderContext&) = delete;

    ~MonoWavEncoderContext() {
        // finalize the sizes specified in the header
        ma_encoder_uninit(&encoder);
    }

    void writePcm(const std::span<const float> pcm) {
        std::vector<ma_int16> s16(pcm.size());
        ma_pcm_f32_to_s16(s16.data(), pcm.data(), pcm.size(), ma_dither_mode_none);

        ma_uint64 framesWritten = 0;
        if (const ma_result writeResult = ma_encoder_write_pcm_frames(&encoder, s16.data(), s16.size(), &framesWritten);
            writeResult != MA_SUCCESS || framesWritten != s16.size()
            ) {
            throw std::runtime_error(std::format("writing {} failed: {}", path.string(), ma_result_description(writeResult)));
        }

        samplesWritten += pcm.size();
    }
};

} // namespace

int main(const int argc, char** argv) {
    CLI::App app("llmvoice - LLM chat to speech pipeline");
    app.require_subcommand(1);

    CLI::App* pipeline = app.add_subcommand("pipeline", "LLM->TTS with a cloned voice");
    CLI::App* llmOnly = app.add_subcommand("llm-only", "Run only the LLM stage");
    CLI::App* ttsVoiceDesign = app.add_subcommand("voice-design", "Render text to a spoken output in a wav file");

    std::string prompt;
    for (CLI::App* sub : {pipeline, llmOnly, ttsVoiceDesign}) {
        sub->add_option("--prompt,-p", prompt, "Prompt text")->required();
    }

    bool noThink = false;
    for (CLI::App* sub : {pipeline, llmOnly}) {
        sub->add_flag("--no-think", noThink, "Suppress the model's reasoning block (faster first response, lower answer quality)");
    }

    // voice-design writes a voice sample here, pipeline clones it from here by default
    const std::filesystem::path defaultVoiceDir = "out/voice";
    constexpr std::string_view sampleWavName = "sample.wav";
    constexpr std::string_view sampleTextName = "transcript.txt";

    std::string wavPath = (defaultVoiceDir / sampleWavName).generic_string();
    std::string transcriptPath = (defaultVoiceDir / sampleTextName).generic_string();
    pipeline->add_option("--sample-wav", wavPath, "Sample voice clip to clone")->capture_default_str();
    pipeline->add_option("--sample-text", transcriptPath, "Transcript of the sample voice clip")->capture_default_str();

    bool skipSegmenter = false;
    llmOnly->add_flag("--skip-segmenter", skipSegmenter, "Output raw text pieces without segmentation");

    std::string ttsInstruct;
    std::string ttsOutPath = defaultVoiceDir.string();
    ttsVoiceDesign->add_option("--instruct", ttsInstruct, "Describe the voice (mood and tone)")->required();
    ttsVoiceDesign->add_option("--out-path", ttsOutPath, "Folder where the spoken text is stored as wav and text transcript")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    // -------------- set up & run the backend --------------
    const std::string llmModel = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf";
    const std::string talkerModelVoiceDesign = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf";
    const std::string talkerModelBase = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-talker-1.7b-base-Q4_K_M.gguf";
    const std::string codecModel  = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-tokenizer-12hz-Q4_K_M.gguf";

    const LlmvoiceConfig config {
        .llmModelPath = llmModel.c_str(),
        .llmContextSize = 8192,
        .ttsTalkerPath = *ttsVoiceDesign
            ? talkerModelVoiceDesign.c_str()
            : talkerModelBase.c_str(),
        .ttsCodecPath = codecModel.c_str(),
        .ttsMode = *ttsVoiceDesign ? 1 : 0
    };

    LlmvoiceHandle* handle;

    {
        bench::Scoped measure("STAGE INIT");

        handle = llmvoiceCreate(&config);
        if (handle == nullptr) {
            throw std::runtime_error(std::format("llmvoiceCreate failed: {}", llmvoiceLastError()));
        }

        llmvoiceSetSegmenterConfig(handle, 24, 60);

        if (*llmOnly) {
            prepLlm(handle);
        }
        else if (*ttsVoiceDesign) {
            // intentionally roll a different seed every time to try out voice variations
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution dist(1, 100);
            prepTtsVoiceDesign(handle, dist(gen), ttsInstruct);
        }
        else {
            prepLlm(handle);
            prepTtsVoiceClone(handle, wavPath, transcriptPath);
        }
    }

    {
        bench::Scoped measure("STAGE WARMUP");

        if (llmvoiceWarmup(handle) != 0) {
            throw std::runtime_error(std::format("llmvoiceWarmup failed: {}", llmvoiceLastError()));
        }
    }

    // origin of the time-to-first-text / time-to-first-audio marks (work is handed to the pipeline)
    const bench::Snapshot submitStart = bench::snapshot();
    bench::Mark firstText;
    bench::Mark firstPcm;

    {
        bench::Scoped measure("STAGE SUBMIT");

        if (*llmOnly) {
            if (llmvoiceSubmitLlm(handle, prompt.c_str(), !skipSegmenter, noThink, 1024) != 0) {
                throw std::runtime_error(std::format("llmvoiceSubmitLlm failed: {}", llmvoiceLastError()));
            }
        }
        else if (*ttsVoiceDesign) {
            if (llmvoiceSubmitTts(handle, prompt.c_str()) != 0) {
                throw std::runtime_error(std::format("llmvoiceSubmitTts failed: {}", llmvoiceLastError()));
            }
        }
        else {
            if (llmvoiceSubmitPipeline(handle, prompt.c_str(), noThink, 1024) != 0) {
                throw std::runtime_error(std::format("llmvoiceSubmitPipeline failed: {}", llmvoiceLastError()));
            }
        }
    }

    // -------------- generate output --------------
    const bool playAudio = !*llmOnly && !*ttsVoiceDesign;
    ma_device device {};
    PlaybackContext playbackContext { .handle = handle, .firstPcm = firstPcm };
    if (playAudio) {
        // the pipeline is already running, so this is time the first-audio mark includes
        bench::Scoped measure("STAGE AUDIO INIT");
        initAudioDevice(playbackContext, device);
    }

    constexpr int32_t llmBufferSize = 64;
    std::string llmBuffer(llmBufferSize, '\0');

    constexpr int32_t ttsChunkSize = 64;
    std::vector<float> ttsChunkBuffer(ttsChunkSize);

    std::filesystem::path wavOut;
    std::filesystem::path transcriptOut;

    // ReSharper disable once CppTooWideScope - writes are *appended*
    std::optional<MonoWavEncoderContext> wavContext;
    if (*ttsVoiceDesign) {
        std::filesystem::path outDir = ttsOutPath;
        std::filesystem::create_directories(outDir);
        transcriptOut = outDir / sampleTextName;
        wavOut = outDir / sampleWavName;
        wavContext.emplace(wavOut, LLMVOICE_PCM_SAMPLE_RATE);
    }

    {
        bench::Scoped measure("STAGE GENERATION & PLAYBACK");

        while (!llmvoiceIsDone(handle)) {
            const size_t written = llmvoicePollText(handle, llmBuffer.data(), llmBufferSize);

            if (*ttsVoiceDesign && wavContext) {
                if (const size_t polled = llmvoicePollPcm(handle, ttsChunkBuffer.data(), ttsChunkSize)) {
                    firstPcm.hit();
                    wavContext->writePcm(std::span(ttsChunkBuffer.data(), polled));
                }
            }

            if (written == 0) {
                // polling is non-blocking: wait a bit instead of spinning while nothing is ready
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            firstText.hit();

            // pieces are fragments of one text (segments are already joined with a space), so no newline in between
            std::print("{}", std::string_view(llmBuffer.data(), written));
            std::fflush(stdout);
        }
        std::println();

        if (wavContext) {
            if (wavContext->samplesWritten == 0) {
                throw std::runtime_error(std::format("no PCM samples were written to {}", wavOut.string()));
            }

            writeFileText(transcriptOut.string(), prompt);
            std::println("wrote audio ({}) and transcript ({})", wavOut.string(), transcriptOut.string());
        }
    }

    // llmvoiceIsDone is true only once the PCM buffer is drained too, so this is the end of output, not of generation
    if (!*ttsVoiceDesign) {
        bench::printSince("time to first text", submitStart, firstText.when());
    }
    if (!*llmOnly) {
        bench::printSince("time to first audio", submitStart, firstPcm.when());
    }
    bench::printSince("all output polled", submitStart, std::chrono::steady_clock::now());

    if (playAudio) {
        playbackAudioRemainder(device);
    }

    llmvoiceDestroy(handle);

    if (const std::string error = llmvoiceLastError(); !error.empty()) {
        throw std::runtime_error(error);
    }

    return 0;
}
