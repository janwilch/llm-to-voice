#pragma once

#include "ITtsBackend.hpp"
#include <memory>
#include <string>

std::unique_ptr<ITtsBackend> createQwen3TtsBackend(std::string talkerPath, std::string codecPath);
