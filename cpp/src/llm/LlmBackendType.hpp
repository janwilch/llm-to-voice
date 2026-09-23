#pragma once

#include <string>

enum class LlmBackendType {
    None,
    Qwen3,
    Fake
};

inline std::string to_string(const LlmBackendType backend) {
    switch (backend) {
        case LlmBackendType::None:
            return "none";

        case LlmBackendType::Qwen3:
            return "qwen3";

        case LlmBackendType::Fake:
            return "fake";
    }

    return "unknown";
}
