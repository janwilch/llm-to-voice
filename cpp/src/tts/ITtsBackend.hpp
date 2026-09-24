#pragma once

#include "../threading/SpscRingBuffer.hpp"

#include <string>

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;

    /// @brief Sets the speaker persona for the following generation.
    /// @param seed Seed for improved speaker consistency.
    /// @param instruct Define the speaker voice.
    virtual void createFreshContext(int64_t seed, const std::string& instruct) = 0;

    /// @brief Runs a synthesis without returning anything.
    virtual void warmup() = 0;

    /// @brief Runs a streaming synthesis, writing audio chunks to the ring buffer.
    /// @param text The text to speak.
    /// @param buffer Receives generated synthesis chunks.
    virtual void synthesizeToBuffer(const std::string& text, SpscRingBuffer<float>& buffer) = 0;

    /// @brief Cancels a running synthesis.
    virtual void cancel() = 0;
};
