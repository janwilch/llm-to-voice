#pragma once

#include "../threading/BlockingQueue.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdio>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>

struct SegmenterConfig {
    size_t minSentenceLength = 24;
    size_t coalesceMinChars = 60;
};

/// @brief Turns a stream of raw LLM token pieces into speakable segments: reassembles UTF-8, drops emoji and `<think>` blocks, splits on sentence ends, and merges the result up to `coalesceMinChars`.
class Segmenter {
    /// @brief Raw bytes from llama, possibly ending mid-character.
    std::string _bytes;
    /// @brief Complete UTF-8 characters, not yet think-filtered.
    std::string _utf8;
    /// @brief Think-filtered text, not yet split into sentences.
    std::string _text;
    /// @brief Sentences held back until `coalesceMinChars` is met.
    std::string _pending;
    bool _inThink = false;
    /// @brief How far into `_text` the sentence scan has already rejected every candidate.
    size_t _scanned = 0;
    size_t _badByteRun = 0;

    void reset() {
        _bytes.clear();
        _utf8.clear();
        _text.clear();
        _pending.clear();
        _inThink = false;
        _scanned = 0;
        _badByteRun = 0;
    }

    /// @brief Malformed bytes are dropped, never forwarded. Reported a run at a time: a stretch of garbage would otherwise be one line per byte.
    void reportBadBytes() {
        if (_badByteRun == 0) {
            return;
        }

        std::println(stderr, "segmenter: dropped {} malformed UTF-8 byte(s)", _badByteRun);
        _badByteRun = 0;
    }

    /// @brief The number of leading ones in a UTF8 byte signals how many subsequent bytes are needed for the full character (i.e. `0xxxxxxx` = 1 byte, `110xxxxx` = 2 bytes, `1110xxxx` = 3, `11110xxx` = 4).
    /// @return 0 if `lead` cannot start a character at all.
    static size_t utf8LeadLength(const unsigned char lead) {
        const int32_t ones = std::countl_one(lead);
        if (ones == 0) {
            return 1;
        }

        if (ones >= 2 && ones <= 4) {
            // 0xF5..0xF7 look like 4-byte leads but encode code points past U+10FFFF
            return lead >= 0xF5 ? 0 : static_cast<size_t>(ones);
        }

        // this is either a continuation or invalid
        return 0;
    }

    /// @brief Emoji and their invisible glue characters, which TTS cannot speak. A byte-range check, not the full Unicode `Extended_Pictographic` property.
    static bool isEmoji(const std::string_view c) {
        if (c.size() == 4) {
            return c.starts_with("\xF0\x9F"); // U+1F000..1FFFF: nearly all emoji, incl. skin tones and flags
        }

        if (c.size() == 3) {
            const auto second = static_cast<unsigned char>(c[1]);
            return (c[0] == '\xE2' && second >= 0x98 && second <= 0x9E) // U+2600..27BF: ☀ ❤ ✅ ✨
                || c == "\xEF\xB8\x8F"  // U+FE0F variation selector
                || c == "\xE2\x80\x8D"  // U+200D zero-width joiner
                || c == "\xE2\x83\xA3"; // U+20E3 keycap
        }

        return false;
    }

    /// @brief Pieces from llama are raw UTF8 *bytes*. A single piece is not always a valid UTF8 character because some take multiple bytes. This method checks received bytes and accumulates them only as valid UTF8 characters.
    void accumulateUtf8Pieces() {
        size_t pos = 0;

        while (pos < _bytes.size()) {
            const unsigned char lead = _bytes[pos];
            const size_t expectedBytes = utf8LeadLength(lead);

            if (expectedBytes == 0) {
                ++pos;
                ++_badByteRun;
                continue;
            }

            if (_bytes.size() - pos < expectedBytes) {
                break; // incomplete character, wait for more bytes
            }

            // every byte after a lead byte must be a continuation (10xxxxxx)
            bool complete = true;
            for (size_t i = 1; i < expectedBytes; ++i) {
                if ((static_cast<unsigned char>(_bytes[pos + i]) & 0xC0) != 0x80) {
                    complete = false;
                    break;
                }
            }

            if (!complete) {
                // drop the lead and resynchronise at the next byte
                ++pos;
                ++_badByteRun;
                continue;
            }

            reportBadBytes();
            if (!isEmoji(std::string_view(_bytes).substr(pos, expectedBytes))) {
                _utf8.append(_bytes, pos, expectedBytes);
            }

            pos += expectedBytes;
        }

        _bytes.erase(0, pos);
    }

    /// @brief The current generation string may *end* like "... <thi", i.e. the "nk>" has not been generated yet. This returns the length of the longest suffix of `text` that is still a proper prefix of `marker`, which is the tail that cannot be emitted yet.
    static size_t heldPrefixLength(const std::string_view text, const std::string_view marker) {
        for (size_t size = std::min(text.size(), marker.size() - 1); size > 0; --size) {
            if (marker.starts_with(text.substr(text.size() - size))) {
                return size;
            }
        }

        return 0;
    }

    /// @brief Qwen3 emits <think>...</think>, which should not make it into the final output.
    void removeThinking() {
        static constexpr std::string_view thinkStart = "<think>";
        static constexpr std::string_view thinkEnd = "</think>";

        while (!_utf8.empty()) {
            std::string_view marker = _inThink ? thinkEnd : thinkStart;
            const size_t markerIdx = _utf8.find(marker);

            if (markerIdx == std::string::npos) {
                // hold back what could still become the marker
                const size_t held = heldPrefixLength(_utf8, marker);
                if (!_inThink) {
                    _text.append(_utf8, 0, _utf8.size() - held);
                }

                _utf8.erase(0, _utf8.size() - held);
                return;
            }

            if (!_inThink) {
                _text.append(_utf8, 0, markerIdx);
            }

            _utf8.erase(0, markerIdx + marker.size());
            _inThink = !_inThink;
        }
    }

    /// @brief Checks (case-insensitively) whether `abbreviation` is the text ending at `_text[candidate]`
    [[nodiscard]] bool matchesAbbreviationAt(const size_t candidate, std::string_view abbreviation) const {
        if (candidate + 1 < abbreviation.size()) {
            return false; // not enough preceding characters to fit the abbreviation
        }

        const size_t start = candidate + 1 - abbreviation.size();
        if (start > 0 && std::isalnum(static_cast<unsigned char>(_text[start - 1]))) {
            return false; // mid-word, so it's the tail of something longer
        }

        std::string_view slice = std::string_view(_text).substr(start, abbreviation.size());
        return std::ranges::equal(slice, abbreviation, [](const unsigned char a, const unsigned char b) {
            return std::toupper(a) == std::toupper(b);
        });
    }

    /// @brief Checks some hard-coded abbreviations that take a period without ending a sentence.
    [[nodiscard]] bool isAbbreviation(const size_t candidate) const {
        // single letters are covered by `isSingleLetterInitial`
        static constexpr std::array abbreviations {
            // titles
            "MR.", "MRS.", "MS.", "MX.", "DR.", "PROF.", "SR.", "JR.", "ST.", "REV.",
            "CAPT.", "GEN.", "LT.", "COL.", "SGT.", "MAJ.", "GOV.", "PRES.", "SEN.", "REP.",
            "ADM.", "CMDR.", "CPL.", "PVT.", "HON.", "FR.", "MSGR.", "ATTY.", "SUPT.",
            // latin / academic
            "E.G.", "I.E.", "ETC.", "VS.", "CF.", "VIZ.", "AL.", "N.B.", "A.M.", "P.M.",
            "NO.", "VOL.", "PP.", "PG.", "FIG.", "FIGS.", "REF.", "APPROX.", "EST.",
            "DEPT.", "GOVT.", "UNIV.", "ASSOC.", "INST.", "ORG.", "ASSN.",
            "ED.", "EDS.", "TRANS.", "VER.", "APPX.", "ART.", "CH.", "SEC.", "SECS.",
            // business
            "CO.", "CORP.", "INC.", "LTD.", "LLC.", "PLC.", "BROS.", "MFG.", "DIV.",
            // units
            "IN.", "FT.", "YD.", "MI.", "CM.", "MM.", "KM.", "KG.", "LB.", "LBS.",
            "OZ.", "MG.", "ML.", "HR.", "HRS.", "MIN.", "MINS.", "SQ.",
            "FL.", "TBSP.", "TSP.",
            // months
            "JAN.", "FEB.", "MAR.", "APR.", "JUN.", "JUL.", "AUG.", "SEP.", "SEPT.", "OCT.", "NOV.", "DEC.",
            // days
            "MON.", "TUE.", "TUES.", "WED.", "THU.", "THUR.", "THURS.", "FRI.", "SAT.", "SUN.",
            // addresses
            "AVE.", "BLVD.", "RD.", "LN.", "CT.", "PL.", "APT.", "BLDG.", "RM.", "HWY.", "PKWY.",
        };

        return std::ranges::any_of(abbreviations, [&](const std::string_view abbreviation) {
            return matchesAbbreviationAt(candidate, abbreviation);
        });
    }

    /// @brief Checks whether `candidate` is a period directly preceded by a single, isolated capital letter (an initial, e.g. "J." in "J. Wilch")
    [[nodiscard]] bool isSingleLetterInitial(const size_t candidate) const {
        if (candidate < 1) {
            return false;
        }

        if (!std::isupper(static_cast<unsigned char>(_text[candidate - 1]))) {
            return false;
        }

        if (candidate >= 2 && std::isalpha(static_cast<unsigned char>(_text[candidate - 2]))) {
            return false;
        }

        return true;
    }

    /// @brief Checks whether `candidate` is the period of a markdown list marker ("1. Open the door")
    [[nodiscard]] bool isListMarker(const size_t candidate) const {
        size_t start = candidate;
        while (start > 0 && std::isdigit(static_cast<unsigned char>(_text[start - 1]))) {
            --start;
        }

        if (start == candidate) {
            return false; // no digits before the period
        }

        while (start > 0 && (_text[start - 1] == ' ' || _text[start - 1] == '\t')) {
            --start;
        }

        return start == 0 || _text[start - 1] == '\n';
    }

    /// @brief Tries finding the proper ends of sentences (`.`, `!`, `?`, `…`, `\n` etc.)
    /// @return The index of the sentence's *last* byte, or npos if the buffer does not hold a complete sentence yet.
    size_t findSentenceEnd(const SegmenterConfig& config) {
        // the last byte of "…" (U+2026 = E2 80 A6) is in the set because find_first_of works on bytes and cannot see the character
        static constexpr std::string_view delimiters = ".!?\n\xA6";
        static constexpr std::string_view whitespace = " \n\t\r";

        size_t candidate = _scanned;

        while ((candidate = _text.find_first_of(delimiters, candidate)) != std::string::npos) {
            // rejections below are final: _text only ever grows at the back, so indices never shift
            _scanned = candidate;

            // if minSentenceLength is not met, continue
            if (candidate < config.minSentenceLength) {
                ++candidate;
                continue;
            }

            if (_text[candidate] == '\xA6' && (candidate < 2 || _text.compare(candidate - 2, 2, "\xE2\x80") != 0)) {
                ++candidate;
                continue; // some other character that happens to end in this byte
            }

            // a line break is definitely a sentence end
            if (_text[candidate] == '\n') {
                return candidate;
            }

            // if the segment ends here, it's not enough to decide
            if (candidate + 1 >= _text.size()) {
                return std::string::npos;
            }

            // if the next char is not whitespace, it isn't a sentence end
            if (!whitespace.contains(_text[candidate + 1])) {
                ++candidate;
                continue;
            }

            if (_text[candidate] == '.' && (isSingleLetterInitial(candidate) || isListMarker(candidate) || isAbbreviation(candidate))) {
                ++candidate;
                continue;
            }

            // sentence end found!
            return candidate;
        }

        _scanned = _text.size();
        return std::string::npos;
    }

    /// @brief Trims ASCII whitespace from both ends. The result may be empty.
    static std::string_view strip(const std::string_view text) {
        static constexpr std::string_view whitespace = " \t\r\n\f\v";

        const size_t first = text.find_first_not_of(whitespace);
        if (first == std::string_view::npos) {
            return {};
        }

        return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
    }

    /// @brief Whether `text` already closes with punctuation.
    static bool endsWithPunctuation(const std::string_view text) {
        // "…", "–", "—" are multi-byte, so they cannot be checked as a single character
        for (const std::string_view suffix : { "\xE2\x80\xA6", "\xE2\x80\x93", "\xE2\x80\x94" }) {
            if (text.ends_with(suffix)) {
                return true;
            }
        }

        // sentence-final, plus the connectors a line break leaves dangling
        static constexpr std::string_view punctuation = ".!?:,;-";
        return !text.empty() && punctuation.contains(text.back());
    }

    /// @brief Pushes sentences at least `coalesceMinChars` long.
    /// @return `false` if the consumer closed the queue.
    bool pushCoalesced(const std::string_view sentence, BlockingQueue<std::string>& segmentsOutQueue, const SegmenterConfig& config) {
        if (sentence.empty()) {
            return true;
        }

        if (config.coalesceMinChars == 0) {
            return segmentsOutQueue.push(std::string(sentence));
        }

        if (!_pending.empty()) {
            _pending += ' ';
        }

        _pending += sentence;

        // yield as soon as the threshold is met
        if (_pending.size() < config.coalesceMinChars) {
            return true;
        }

        const bool accepted = segmentsOutQueue.push(std::move(_pending));
        _pending.clear();
        return accepted;
    }

    /// @brief Drains every complete sentence currently in `_text`.
    /// @return `false` if the consumer closed the queue.
    bool emitSentences(BlockingQueue<std::string>& segmentsOutQueue, const SegmenterConfig& config) {
        size_t sentenceEnd;

        while ((sentenceEnd = findSentenceEnd(config)) != std::string::npos) {
            std::string sentence = _text.substr(0, sentenceEnd + 1);
            _text.erase(0, sentenceEnd + 1);
            _scanned = 0;

            std::string speakable(strip(sentence));
            if (speakable.empty()) {
                continue;
            }

            // Make sure sentences always end with punctuation
            if (!endsWithPunctuation(speakable)) {
                speakable += '.';
            }

            if (!pushCoalesced(speakable, segmentsOutQueue, config)) {
                return false;
            }
        }

        return true;
    }

    /// @brief End of input: release every buffer.
    void flush(BlockingQueue<std::string>& segmentsOutQueue, const SegmenterConfig& config) {
        _badByteRun += _bytes.size();
        _bytes.clear();
        reportBadBytes();

        if (!_inThink) {
            _text += _utf8;
        }

        _utf8.clear();

        if (!emitSentences(segmentsOutQueue, config)) {
            return;
        }

        // what is left is the last segment (not considering sentence end checks)
        if (
            const std::string_view tail = strip(_text);
            !tail.empty()
            ) {
            if (!_pending.empty()) {
                _pending += ' ';
            }

            _pending += tail;

            if (!endsWithPunctuation(_pending)) {
                _pending += '.';
            }
        }

        _text.clear();

        if (!_pending.empty()) {
            segmentsOutQueue.push(std::move(_pending));
            _pending.clear();
        }
    }

public:
    /// @brief Pulls token pieces until the input queue closes.
    void segment(BlockingQueue<std::string>& tokensInQueue, BlockingQueue<std::string>& segmentsOutQueue, const SegmenterConfig config = {}) {
        // the consumer blocks on pop(), so the output queue has to close on *every* exit path
        QueueCloser closer { segmentsOutQueue };
        reset();

        // pop() returns nullopt only once queue is closed *and* empty
        while (std::optional<std::string> piece = tokensInQueue.pop()) {
            _bytes += piece.value();
            accumulateUtf8Pieces();
            removeThinking();

            if (!emitSentences(segmentsOutQueue, config)) {
                // the consumer closed the queue -> do nothing more
                return;
            }
        }

        flush(segmentsOutQueue, config);
    }
};
