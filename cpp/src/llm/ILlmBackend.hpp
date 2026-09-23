#pragma once

#include <string>
#include "../threading/BlockingQueue.hpp"

class ILlmBackend {
public:
    virtual ~ILlmBackend() = default;

    /// @brief Runs a synthesis without returning anything.
    virtual void warmup() = 0;

    /// @brief Clears existing context & memory. Creates a new chat template from the system prompt.
    /// @param systemPrompt System prompt for chat template.
    virtual void createFreshContext(const std::string& systemPrompt) = 0;

    /// @brief Runs a streaming LLM call, writing response tokens to the queue.
    /// @param prompt The LLM prompt.
    /// @param queue Receives generated LLM tokens. NOTE: Closed after synthesis!
    /// @param noThink Suppresses the model's reasoning block.
    /// @param maxTokens The maximum number of generated response tokens.
    virtual void synthesizeToQueue(const std::string& prompt, BlockingQueue<std::string>& queue, bool noThink, int32_t maxTokens) = 0;
    
    /// @brief Cancels a running synthesis.
    virtual void cancel() = 0;
};
