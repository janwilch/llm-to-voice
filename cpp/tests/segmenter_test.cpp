// Pins the segmenter's behaviour: UTF-8 reassembly, <think> filtering, sentence
// splitting and coalescing.

#include "../src/segmenter/Segmenter.hpp"
#include "../src/threading/BlockingQueue.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <chrono>
#include <concepts>
#include <string>
#include <thread>
#include <vector>

using Segments = std::vector<std::string>;

/// @brief Runs the segmenter over `pieces` to completion. The input queue is closed up front, so pop() never blocks and the stage can run inline.
static Segments run(const Segments& pieces, const SegmenterConfig config = {}) {
    BlockingQueue<std::string> in { 8192 };
    BlockingQueue<std::string> out { 8192 };

    for (const std::string& piece : pieces) {
        in.push(piece);
    }

    in.close();

    Segmenter segmenter;
    segmenter.segment(in, out, config);

    Segments segments;
    while (std::optional<std::string> segment = out.pop()) {
        segments.push_back(std::move(segment.value()));
    }

    return segments;
}

/// @brief Feeds `text` one byte at a time, which is the worst case every buffer in the stage has to survive.
static Segments runByBytes(const std::string& text, const SegmenterConfig config = {}) {
    Segments pieces;
    for (char c : text) {
        pieces.emplace_back(1, c);
    }

    return run(pieces, config);
}

/// @brief Collapses runs of whitespace so "no text is lost" can be checked without asserting where the splitter put its spaces.
static std::string collapse(const std::string_view text) {
    std::string out;
    bool inSpace = true; // leading whitespace is dropped

    for (const char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            inSpace = true;
            continue;
        }

        if (inSpace && !out.empty()) {
            out += ' ';
        }

        inSpace = false;
        out += c;
    }

    return out;
}

/// @brief The spoken text as one string, so completeness can be compared against the input regardless of where the splitter drew its boundaries.
static std::string collapse(const Segments& segments) {
    std::string joined;
    for (const std::string& segment : segments) {
        if (!joined.empty()) {
            joined += ' ';
        }

        joined += segment;
    }

    return collapse(std::string_view(joined));
}

/// @brief Polls until `condition` holds or the budget runs out.
/// @return Whether it held.
static bool waitFor(std::predicate auto condition, const int attempts = 200) {
    for (int attempt = 0; attempt < attempts; ++attempt) {
        if (condition()) {
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

// Splitting only: no minimum length, no coalescing, so each sentence is visible.
static constexpr SegmenterConfig splitOnly { .minSentenceLength = 0, .coalesceMinChars = 0 };
static constexpr SegmenterConfig coalescing { .minSentenceLength = 0, .coalesceMinChars = 60 };

TEST_CASE("sentence splitting", "[segmenter][split]") {
    auto [name, pieces, expected] = GENERATE(table<std::string_view, Segments, Segments>({
        {"multiple sentences yield multiple segments",
         {"Hello there. General Kenobi. "},
         {"Hello there.", "General Kenobi."}},

        {"trailing fragment is flushed, and punctuated",
         {"Hello there. And a trailing fragment"},
         {"Hello there.", "And a trailing fragment."}},

        {"a truncated stream does not leave the final segment trailing off",
         {"Cut off mid sentence and then we"},
         {"Cut off mid sentence and then we."}},

        {"a final segment that already has punctuation keeps it",
         {"Ends on a connector,"},
         {"Ends on a connector,"}},

        {"a single short input survives", {"Yes."}, {"Yes."}},

        {"whitespace-only input yields no segment", {"   "}, {}},

        {"empty input yields no segment", {}, {}},

        {"3.14 is not split",
         {"Pi is 3.14 exactly. Yes. "},
         {"Pi is 3.14 exactly.", "Yes."}},

        {"abbreviations do not end a sentence",
         {"Ask Dr. Smith about it. Then go. "},
         {"Ask Dr. Smith about it.", "Then go."}},

        {"single-letter initials do not end a sentence",
         {"Ask J. K. Rowling about it. Then go. "},
         {"Ask J. K. Rowling about it.", "Then go."}},

        // the regression that made the splitter stop splitting: without a word boundary,
        // "left." matches the abbreviation "FT." and "jumped." matches "ED."
        {"a word merely ending in an abbreviation still ends a sentence",
         {"He had already left. Then it ran off. "},
         {"He had already left.", "Then it ran off."}},

        {"\"jumped.\" is not the abbreviation \"ed.\"",
         {"The fox had jumped. Then it ran off. "},
         {"The fox had jumped.", "Then it ran off."}},

        {"a sentence ending in a number still ends a sentence",
         {"The answer is 42. Then we left. "},
         {"The answer is 42.", "Then we left."}},

        {"a list marker at the start of a line is not a sentence end",
         {"1. Open the door. "},
         {"1. Open the door."}},

        // the line break itself is trimmed with the surrounding whitespace, so it has to be
        // replaced or the segment reaches the TTS with nothing to speak it as a sentence
        {"a line break ends a sentence, and becomes a period",
         {"First line\nSecond line\n"},
         {"First line.", "Second line."}},

        {"a line already ending in punctuation does not get a second one",
         {"First line.\nSecond line?\nThird line…\n"},
         {"First line.", "Second line?", "Third line…"}},

        {"a line ending in a connector keeps the punctuation the model wrote",
         {"You will need:\nA hammer,\nand a nail\n"},
         {"You will need:", "A hammer,", "and a nail."}},

        {"an ellipsis ends a sentence",
         {"Wait for it… Then go. "},
         {"Wait for it…", "Then go."}},

        {"? and ! end a sentence",
         {"Really?! No way! "},
         {"Really?!", "No way!"}},

        {"every segment is stripped", {"  Hello there.  "}, {"Hello there."}},
    }));

    INFO(name);
    CAPTURE(pieces);
    CHECK(run(pieces, splitOnly) == expected);
}

TEST_CASE("coalesced lines do not run together", "[segmenter][split]") {
    CHECK_THAT(run({"First line\nSecond line\n"}, coalescing),
               Catch::Matchers::Equals(Segments{"First line. Second line."}));
}

TEST_CASE("sentence ends below the minimum length are ignored", "[segmenter][split]") {
    CHECK_THAT(run({"Yes. No. Maybe so, and here is a longer clause. "},
                   { .minSentenceLength = 24, .coalesceMinChars = 0 }),
               Catch::Matchers::Equals(Segments{"Yes. No. Maybe so, and here is a longer clause."}));
}

TEST_CASE("think filtering", "[segmenter][think]") {
    auto [name, pieces, expected] = GENERATE(table<std::string_view, Segments, Segments>({
        {"a think block is removed",
         {"<think>reasoning</think>Hello there. "},
         {"Hello there."}},

        {"multiple think blocks are removed",
         {"One. <think>first</think>Two. <think>second</think>Three. "},
         {"One.", "Two.", "Three."}},

        {"an unterminated think block is dropped, not spoken",
         {"<think>", "I should ", "say hi"},
         {}},

        {"text before an unterminated block survives it",
         {"Hello there. <think>reasoning"},
         {"Hello there."}},

        {"tags split across pieces are still matched",
         {"<thi", "nk>reasoning</thi", "nk>Hello there. "},
         {"Hello there."}},

        {"a held partial tag is released on flush",
         {"Hello there friend<thi"},
         {"Hello there friend<thi."}},

        {"a bare '<' is not mistaken for a tag",
         {"5 < 10 and 3 > 1, so there. "},
         {"5 < 10 and 3 > 1, so there."}},

        {"text after the closing tag is processed in the same call",
         {"<think>reasoning</think>Hello there. And more. "},
         {"Hello there.", "And more."}},
    }));

    INFO(name);
    CAPTURE(pieces);
    CHECK(run(pieces, splitOnly) == expected);
}

TEST_CASE("the result is independent of chunk size", "[segmenter][think]") {
    const std::string text = "Before. <think>hidden</think> After the block. ";
    CHECK(runByBytes(text, splitOnly) == run({text}, splitOnly));
}

TEST_CASE("a 4-byte character split across pieces is reassembled", "[segmenter][utf8]") {
    // U+1F600, four bytes: split at each of its three interior boundaries
    const std::string emoji = "\xF0\x9F\x98\x80";
    const size_t cut = GENERATE(range(1uz, 4uz));
    CAPTURE(cut);

    CHECK(run({ "hi " + emoji.substr(0, cut), emoji.substr(cut) + " there. " }, splitOnly)
          == Segments{"hi " + emoji + " there."});
}

TEST_CASE("UTF-8 reassembly", "[segmenter][utf8]") {
    auto [name, pieces, expected] = GENERATE(table<std::string_view, Segments, Segments>({
        {"2-byte characters split across pieces are reassembled",
         {"Gr\xC3", "\xBC\xC3\x9F e. "},
         {"Gr\xC3\xBC\xC3\x9F e."}},

        {"a stream ending mid-sequence drops the incomplete tail",
         {"Hello there\xC3"},
         {"Hello there."}},

        {"an invalid byte is dropped", {"He\xFFllo there. "}, {"Hello there."}},

        {"a lead byte without a continuation is dropped", {"He\xC3llo there. "}, {"Hello there."}},

        {"a stray continuation byte is dropped", {"He\x80llo there. "}, {"Hello there."}},
    }));

    INFO(name);
    CAPTURE(pieces);
    CHECK(run(pieces, splitOnly) == expected);
}

TEST_CASE("coalescing", "[segmenter][coalesce]") {
    SECTION("short segments are merged until the threshold is met") {
        // six short sentences, each well under the threshold
        const std::string shorts = "One two. Three four. Five six. Seven eight. Nine ten. Eleven twelve. ";
        Segments merged = run({shorts}, coalescing);

        CHECK(merged.size() < 6);
        REQUIRE_FALSE(merged.empty());
        CHECK(merged.front().size() >= 60);
        CHECK(collapse(merged) == collapse(shorts));
    }

    SECTION("an already-long segment is left alone and the tail is flushed") {
        const std::string longSentence = "This one sentence is already comfortably past the coalescing threshold on its own. ";
        CHECK_THAT(run({longSentence + "And a short tail. "}, coalescing),
                   Catch::Matchers::Equals(Segments{collapse(longSentence), "And a short tail."}));
    }

    SECTION("coalescing is disabled at 0") {
        CHECK_THAT(run({"One two. Three four. "}, splitOnly),
                   Catch::Matchers::Equals(Segments{"One two.", "Three four."}));
    }

    SECTION("the first segment is coalesced like any other") {
        // a short opener is exactly the one VoiceDesign renders as somebody else
        Segments opener = run({"Hi. Now here is a much longer second sentence to merge with. "}, coalescing);

        REQUIRE_FALSE(opener.empty());
        CHECK(opener.front().starts_with("Hi. Now"));
    }
}

/// @brief Coalescing must yield as soon as the threshold is met, not at end of input, or it would cost time-to-first-audio on every segment rather than only on short ones.
TEST_CASE("coalescing yields before the input is exhausted", "[segmenter][threading][slow]") {
    BlockingQueue<std::string> in { 64 };
    BlockingQueue<std::string> out { 64 };

    Segmenter segmenter;
    std::jthread worker([&] { segmenter.segment(in, out); });
    // declared after the worker so it runs first, unblocking pop() before the join
    QueueCloser closer { in };

    in.push("This first sentence is comfortably longer than the coalescing threshold. ");

    CHECK(waitFor([&] { return !out.empty(); }));
}

/// @brief push() returning false is how barge-in reaches this stage: buffered text is stale and must not become audio after the user interrupted.
TEST_CASE("a closed output queue stops the stage before it drains its input", "[segmenter][threading][slow]") {
    BlockingQueue<std::string> in { 64 };
    BlockingQueue<std::string> out { 64 };

    for (int i = 0; i < 50; ++i) {
        in.push("This is one more sentence in a long stream of them. ");
    }

    out.close();

    Segmenter segmenter;
    std::jthread worker([&] { segmenter.segment(in, out, splitOnly); });
    // declared after the worker so it runs first, unblocking pop() before the join
    QueueCloser closer { in };

    // it must stop without waiting for the input queue to close
    CHECK_FALSE(waitFor([&] { return in.empty(); }));
}

/// @brief The property the completeness tests exist to guard.
TEST_CASE("no text is lost", "[segmenter][property]") {
    const std::string answer = "Hello there. Ask Dr. Smith about the 3.14 result, please. "
                               "It was 42. Then everyone left.";
    const std::string stream = "<think>let me consider this</think>" + answer;

    SECTION("one piece") {
        CHECK(collapse(run({stream})) == collapse(answer));
    }

    SECTION("one byte at a time") {
        CHECK(collapse(runByBytes(stream)) == collapse(answer));
    }

    SECTION("no coalescing") {
        CHECK(collapse(run({stream}, splitOnly)) == collapse(answer));
    }
}
