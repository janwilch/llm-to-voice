#include <concepts>
#include <format>
#include <string>

template <typename T>
concept HasToString = requires (T value) {
    { to_string(value) } -> std::convertible_to<std::string>;
};

template<HasToString T>
struct std::formatter<T> : std::formatter<std::string> {
    auto format(T value, auto& ctx) const {
        return std::formatter<std::string>::format(to_string(value), ctx);
    }
};
