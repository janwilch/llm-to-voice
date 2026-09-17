#pragma once

#include <string>
#include <vector>
#include "../threading/BlockingQueue.hpp"

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;

    /// @brief Sets the speaker persona for the following generation.
    /// @param seed Seed for improved speaker consistency.
    /// @param instruct Define the speaker voice.
    virtual void createFreshContext(const int64_t seed, const std::string& instruct) = 0;

    /// @brief Runs a synthesis without returning anything.
    virtual void warmup() = 0;

    /// @brief Runs a streaming synthesis, writing audio chunks to the queue.
    /// @param text The text to speak.
    /// @param queue Receives generated synthesis chunks. NOTE: Closed after synthesis!
    virtual void synthesizeToQueue(const std::string& text, BlockingQueue<std::vector<float>>& queue) = 0;

    /// @brief Cancels a running synthesis.
    virtual void cancel() = 0;
};
