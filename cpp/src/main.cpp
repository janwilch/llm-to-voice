#include <CLI/CLI.hpp>
#include <chrono>
#include <cstdio>
#include <print>
#include <string_view>
#include <thread>

#include "llmvoice.h"

int main(const int argc, char** argv) {
    CLI::App app("llmvoice - LLM chat to speech pipeline");
    app.require_subcommand(1);

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

    if (llmvoiceWarmup(handle) != 0) {
        throw std::runtime_error(std::format("llmvoiceWarmup failed: {}", llmvoiceLastError()));
    }

    if (*llmOnly) {
        if (llmvoiceCreateLlmContext(handle, "default") != 0) {
            throw std::runtime_error(std::format("llmvoiceCreateLlmContext failed: {}", llmvoiceLastError()));
        }

        if (llmvoiceSubmitLlm(handle, prompt.c_str(), !skipSegmenter, noThink, 1024) != 0) {
            throw std::runtime_error(std::format("llmvoiceSubmitLlm failed: {}", llmvoiceLastError()));
        }
    }
    else if (*ttsOnly) {
        // TODO
    }
    else {
        // TODO - full pipeline
    }

    std::string llmBuffer(64, '\0');
    
    while (!llmvoiceIsDone(handle)) {
        constexpr int32_t llmBufferSize = 64;
        const size_t written = llmvoicePollText(handle, llmBuffer.data(), llmBufferSize);
        // TODO - TTS polling

        if (written == 0) {
            // polling is non-blocking: wait a bit instead of spinning while nothing is ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // pieces are fragments of one text (segments are already joined with a space), so no newline in between
        std::print("{}", std::string_view(llmBuffer.data(), written));
        std::fflush(stdout);
    }
    std::println();

    llmvoiceDestroy(handle);

    const std::string error = llmvoiceLastError();
    if (!error.empty()) {
        throw std::runtime_error(error);
    }

    return 0;
}
