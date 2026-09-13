#include <CLI/CLI.hpp>
#include <print>
#include <thread>

#include "threading/BlockingQueue.hpp"
#include "tts/ITtsBackend.hpp"
#include "tts/Qwen3TtsBackend.hpp"
#include "tts/TtsBackendType.hpp"

int main(int argc, char** argv) {
    CLI::App app("llmvoice - LLM chat to speech pipeline");

    std::map<std::string, TtsBackendType> backendMap{
        {"qwen3-tts", TtsBackendType::Qwen3Tts},
        {"fake", TtsBackendType::Fake}
    };
    
    TtsBackendType ttsBackend = TtsBackendType::None;
    std::string voice = "default";
    std::string prompt;
    bool fakeLlm;
    bool noAudio;

    app.add_option("--tts-backend", ttsBackend, "set the TTS backend - 'fake' runs the pipeline w/o GPU")
        ->transform(CLI::CheckedTransformer(backendMap));

    app.add_option("--voice", voice, "select a voice or omit for default");
    app.add_option("-p,--prompt", prompt, "the prompt");
    
    app.add_flag("--fake-llm", fakeLlm, "canned LLM replies without llama");
    app.add_flag("--no-audio", noAudio, "run without opening audio device");

    CLI11_PARSE(app, argc, argv);
    
    std::print("llmvoice - tts={} voice={}, fake-llm={}, no-audio={}\nprompt={}\n", 
        ttsBackend, voice, fakeLlm, noAudio, prompt);

    // -------------- set up & run the TTS model --------------
    std::string talkerModel = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf";
    std::string codecModel  = std::string(QWEN_DEFAULT_MODELS_DIR) + "/Qwen3-TTS-GGUF/qwen-tokenizer-12hz-Q4_K_M.gguf";

    std::unique_ptr<ITtsBackend> backend = createQwen3TtsBackend(talkerModel, codecModel);
    size_t queueCapacity = 8;
    BlockingQueue<std::vector<float>> queue(queueCapacity);

    std::jthread producer([&backend, &prompt, &voice, &queue] {
        backend->synthesize_to_queue(prompt, voice, queue, -1);
    });

    int chunkNo = 0;
    while (std::optional<std::vector<float>> chunk = queue.pop()) {
        try {
            std::print("chunk {}: {} samples\n", chunkNo++, chunk->size());
        }
        catch (const std::exception& ex) {
            std::print(stderr, "synthesis failed: {}\n", ex.what());
        }
    }

    return 0;
}
