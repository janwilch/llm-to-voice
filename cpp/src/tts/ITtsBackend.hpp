#pragma once

#include "../threading/SpscRingBuffer.hpp"

#include <string>

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;

    /// @brief Sets the speaker persona for a Qwen3-TTS-VoiceDesign generation.
    /// @param seed Seed for improved speaker consistency.
    /// @param instruct Define the speaker voice.
    virtual void createVoiceDesignContext(int64_t seed, const std::string& instruct) = 0;

    /// @brief Sets the speaker persona for a Qwen3-TTS-Base generation.
    /// @param wavPath Path to an example audio clip. The generation will clone this voice.
    /// @param transcriptPath Path to a transcript of the example.
    virtual void createBaseContext(const std::string &wavPath, const std::string &transcriptPath) = 0;

    /// @brief Runs a synthesis without returning anything.
    virtual void warmup() = 0;

    /// @brief Runs a streaming synthesis, writing audio chunks to the ring buffer.
    /// @param text The text to speak.
    /// @param buffer Receives generated synthesis chunks.
    virtual void synthesizeToBuffer(const std::string& text, SpscRingBuffer<float>& buffer) = 0;

    /// @brief Cancels a running synthesis. Stays in effect until `resetCancel`.
    virtual void cancel() = 0;

    /// @brief Clears a previous `cancel`. Call once per session, before any synthesis of that session starts.
    virtual void resetCancel() = 0;
};
