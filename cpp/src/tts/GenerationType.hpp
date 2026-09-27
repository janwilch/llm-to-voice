#pragma once

#include <stdexcept>
#include <string>

enum class GenerationType {
    None,
    /// @brief Voice is described textually (`instruct`) and optionally by a seed.
    VoiceDesign,
    /// @brief Voice is cloned from a sample WAV and optionally a matching transcript.
    Base
};

inline std::string to_string(const GenerationType generationType) {
    switch (generationType) {
        case GenerationType::None:
            return "None";
        case GenerationType::VoiceDesign:
            return "VoiceDesign";
        case GenerationType::Base:
            return "Base";
        default:
            throw std::invalid_argument("GenerationType not implemented");
    }
}
