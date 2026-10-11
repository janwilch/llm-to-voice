#pragma once

#include <cctype>
#include <iterator>
#include <regex>
#include <string>
#include <string_view>

// A tag is `{{name}}` with a name of letters, digits and underscores. The LLM emits it for whoever polls the text (e.g. to trigger an animation); it is never spoken.
// The name cannot hold a sentence delimiter, so the segmenter never splits a tag across segments.

/// @brief Removes every tag from `text`, along with the whitespace gap it leaves behind.
inline std::string stripTags(const std::string_view text) {
    static const std::regex tag { R"(\{\{[A-Za-z0-9_]+\}\})" };
    static constexpr std::string_view whitespace = " \t\r\n";
    static constexpr std::string_view closing = ".!?,;:";

    std::string stripped;
    std::regex_replace(std::back_inserter(stripped), text.begin(), text.end(), tag, "");

    // a removed tag leaves its surroundings behind: "happy {{smile}}." -> "happy ."
    std::string tidy;
    for (const char c : stripped) {
        const bool afterGap = !tidy.empty() && whitespace.contains(tidy.back());

        if (whitespace.contains(c)) {
            if (tidy.empty() || afterGap) {
                continue;
            }
        }
        else if (afterGap && closing.contains(c)) {
            tidy.pop_back();
        }

        tidy += c;
    }

    if (!tidy.empty() && whitespace.contains(tidy.back())) {
        tidy.pop_back();
    }

    return tidy;
}

/// @brief `text` without the tags (and whitespace) at its end. Empty if `text` is nothing but tags.
inline std::string_view withoutTrailingTags(const std::string_view text) {
    static const std::regex trailingTags { R"((?:\s*\{\{[A-Za-z0-9_]+\}\})+\s*$)" };

    std::match_results<std::string_view::const_iterator> match;
    if (!std::regex_search(text.begin(), text.end(), match, trailingTags)) {
        return text;
    }

    return text.substr(0, static_cast<size_t>(match.position(0)));
}

/// @brief Whether `text` holds anything besides whitespace and ASCII punctuation, i.e. anything a TTS could speak.
inline bool hasSpeakableText(const std::string_view text) {
    for (const unsigned char c : text) {
        if (!std::isspace(c) && !std::ispunct(c)) {
            return true;
        }
    }

    return false;
}
