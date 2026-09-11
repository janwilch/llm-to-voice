#pragma once
#include <string>

enum class TtsBackend {
  Qwen3Tts,
  Fake
};

inline std::string to_string(TtsBackend backend) {
  switch (backend) {
    case TtsBackend::Qwen3Tts:  return "qwen3_tts";
    case TtsBackend::Fake:      return "fake";
  }

  return "unknown";
}

template <>
struct std::formatter<TtsBackend> : std::formatter<std::string> {
  auto format(TtsBackend backend, auto& ctx) const {
    return std::formatter<std::string>::format(to_string(backend), ctx);
  }
};
