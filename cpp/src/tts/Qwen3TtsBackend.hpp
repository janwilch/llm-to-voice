#pragma once

#include "ITtsBackend.hpp"
#include <memory>
#include <string>

std::unique_ptr<ITtsBackend> createQwen3TtsBackend(const std::string& talkerPath, const std::string& codecPath);
