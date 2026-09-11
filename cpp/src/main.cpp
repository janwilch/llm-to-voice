#include <cstdio>
#include <CLI/CLI.hpp>
#include <print>

#include "model/tts_backend.hpp"

int main(int argc, char** argv) {
    CLI::App app("llmvoice - LLM chat to speech pipeline");

    std::map<std::string, TtsBackend> backend_map{
        {"qwen3_tts", TtsBackend::Qwen3Tts},
        {"fake", TtsBackend::Fake}
    };
    
    TtsBackend tts_backend;
    std::string voice = "default";
    std::string prompt;
    bool fake_llm;
    bool no_audio;

    app.add_option("--tts-backend", tts_backend, "set the TTS backend - 'fake' runs the pipeline w/o GPU")
        ->transform(CLI::CheckedTransformer(backend_map));

    app.add_option("--voice", voice, "select a voice or omit for default");
    app.add_option("-p,--prompt", prompt, "the prompt");
    
    app.add_flag("--fake-llm", fake_llm, "canned LLM replies without llama");
    app.add_flag("--no-audio", fake_llm, "run without opening audio device");
    
    std::print("llmvoice - tts={} voice={}, fake-llm={}, no-audio={}\nprompt={}\n", 
        tts_backend, voice, fake_llm, no_audio, prompt);
    
    return 0;
}
