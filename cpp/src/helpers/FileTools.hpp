#pragma once

#include <format>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

inline std::string readFileText(const std::string &path) {
    const std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(std::format("cannot open file: {}", path));
    }

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

inline void writeFileText(const std::string &path, const std::string &text) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(std::format("cannot open file: {}", path));
    }
    file << text;
    if (!file) {
        throw std::runtime_error(std::format("cannot write to file: {}", path));
    }
}
