#pragma once

#include <string>

enum class TtsBackendType {
    None,
    Qwen3Tts,
    Fake
};

inline std::string to_string(TtsBackendType backend) {
    switch (backend) {
        case TtsBackendType::Qwen3Tts:
            return "qwen3_tts";
        
        case TtsBackendType::Fake:
            return "fake";
    }

    return "unknown";
}

template <>
struct std::formatter<TtsBackendType> : std::formatter<std::string> {
    auto format(TtsBackendType backend, auto& ctx) const {
        return std::formatter<std::string>::format(to_string(backend), ctx);
    }
};
