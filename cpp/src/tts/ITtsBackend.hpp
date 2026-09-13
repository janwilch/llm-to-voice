#pragma once

#include <string>
#include <vector>
#include "../threading/BlockingQueue.hpp"

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;

    /// @brief Runs a synthesis without returning anything.
    virtual void warmup() = 0;

    /// @brief Runs a streaming synthesis, writing to the queue.
    /// @param text The text to speak.
    /// @param instruct Define the speaker voice.
    /// @param queue Receives generated synthesis chunks. NOTE: Closed after synthesis!
    /// @param seed Optionally pass a seed for improved speaker consistency.
    virtual void synthesize_to_queue(
        const std::string& text, 
        const std::string& instruct, 
        BlockingQueue<std::vector<float>>& queue,
        int64_t seed = -1) = 0;

    /// @brief Cancels a running synthesis.
    virtual void cancel() = 0;
};
