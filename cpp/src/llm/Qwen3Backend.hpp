#pragma once

#include "ILlmBackend.hpp"

#include "llama.h"

#include <memory>
#include <string>

struct BackendLifetime {
    BackendLifetime() { llama_backend_init(); }
    ~BackendLifetime() { llama_backend_free(); }
};

std::unique_ptr<ILlmBackend> createQwen3Backend(const std::string& modelPath, int32_t contextSize);
