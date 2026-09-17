#pragma once

#include "ILlmBackend.hpp"
#include <memory>
#include <string>

std::unique_ptr<ILlmBackend> createQwen3Backend(const std::string& modelPath);
