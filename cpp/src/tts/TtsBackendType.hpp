#pragma once

#include <string>

enum class TtsBackendType {
    None,
    Qwen3Tts,
    Fake
};

inline std::string to_string(TtsBackendType backend) {
    switch (backend) {
        case TtsBackendType::None:
            return "none";
        
        case TtsBackendType::Qwen3Tts:
            return "qwen3_tts";
        
        case TtsBackendType::Fake:
            return "fake";
    }

    return "unknown";
}
