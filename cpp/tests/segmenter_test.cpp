// Pins the segmenter's behaviour: UTF-8 reassembly, <think> filtering, sentence
// splitting and coalescing.

#include "../src/segmenter/Segmenter.hpp"
#include "../src/threading/BlockingQueue.hpp"

#include <chrono>
#include <print>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

static void check(bool ok, std::string_view name) {
    if (!ok) {
        ++failures;
    }

    std::println("{} {}", ok ? "ok  " : "FAIL", name);
}

static void checkEqual(const std::vector<std::string>& got, const std::vector<std::string>& want, std::string_view name) {
    if (got == want) {
        check(true, name);
        return;
    }

    check(false, name);
    std::println("       got : {}", got);
    std::println("       want: {}", want);
}

/// @brief Runs the segmenter over `pieces` to completion. The input queue is closed up front, so pop() never blocks and the stage can run inline.
static std::vector<std::string> run(const std::vector<std::string>& pieces, SegmenterConfig config = {}) {
    BlockingQueue<std::string> in { 8192 };
    BlockingQueue<std::string> out { 8192 };

    for (const std::string& piece : pieces) {
        in.push(piece);
    }

    in.close();

    Segmenter segmenter;
    segmenter.segment(in, out, config);

    std::vector<std::string> segments;
    while (std::optional<std::string> segment = out.pop()) {
        segments.push_back(std::move(segment.value()));
    }

    return segments;
}

/// @brief Feeds `text` one byte at a time, which is the worst case every buffer in the stage has to survive.
static std::vector<std::string> runByBytes(const std::string& text, SegmenterConfig config = {}) {
    std::vector<std::string> pieces;
    for (char c : text) {
        pieces.emplace_back(1, c);
    }

    return run(pieces, config);
}

static std::string join(const std::vector<std::string>& segments) {
    std::string joined;
    for (const std::string& segment : segments) {
        if (!joined.empty()) {
            joined += ' ';
        }

        joined += segment;
    }

    return joined;
}

/// @brief Collapses runs of whitespace so "no text is lost" can be checked without asserting where the splitter put its spaces.
static std::string collapse(std::string_view text) {
    std::string out;
    bool inSpace = true; // leading whitespace is dropped

    for (char c : text) {
        bool space = c == ' ' || c == '\n' || c == '\t' || c == '\r';
        if (space) {
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

// Splitting only: no minimum length, no coalescing, so each sentence is visible.
static constexpr SegmenterConfig splitOnly { .minSentenceLength = 0, .coalesceMinChars = 0 };

static void testSentenceSplitting() {
    checkEqual(run({"Hello there. General Kenobi. "}, splitOnly),
               {"Hello there.", "General Kenobi."},
               "multiple sentences yield multiple segments");

    checkEqual(run({"Hello there. And a trailing fragment"}, splitOnly),
               {"Hello there.", "And a trailing fragment."},
               "trailing fragment is flushed, and punctuated");

    checkEqual(run({"Cut off mid sentence and then we"}, splitOnly),
               {"Cut off mid sentence and then we."},
               "a truncated stream does not leave the final segment trailing off");

    checkEqual(run({"Ends on a connector,"}, splitOnly),
               {"Ends on a connector,"},
               "a final segment that already has punctuation keeps it");

    checkEqual(run({"Yes."}, splitOnly), {"Yes."}, "a single short input survives");

    checkEqual(run({"   "}, splitOnly), {}, "whitespace-only input yields no segment");

    checkEqual(run({}, splitOnly), {}, "empty input yields no segment");

    checkEqual(run({"Pi is 3.14 exactly. Yes. "}, splitOnly),
               {"Pi is 3.14 exactly.", "Yes."},
               "3.14 is not split");

    checkEqual(run({"Ask Dr. Smith about it. Then go. "}, splitOnly),
               {"Ask Dr. Smith about it.", "Then go."},
               "abbreviations do not end a sentence");

    checkEqual(run({"Ask J. K. Rowling about it. Then go. "}, splitOnly),
               {"Ask J. K. Rowling about it.", "Then go."},
               "single-letter initials do not end a sentence");

    // the regression that made the splitter stop splitting: without a word boundary,
    // "left." matches the abbreviation "FT." and "jumped." matches "ED."
    checkEqual(run({"He had already left. Then it ran off. "}, splitOnly),
               {"He had already left.", "Then it ran off."},
               "a word merely ending in an abbreviation still ends a sentence");

    checkEqual(run({"The fox had jumped. Then it ran off. "}, splitOnly),
               {"The fox had jumped.", "Then it ran off."},
               "\"jumped.\" is not the abbreviation \"ed.\"");

    checkEqual(run({"The answer is 42. Then we left. "}, splitOnly),
               {"The answer is 42.", "Then we left."},
               "a sentence ending in a number still ends a sentence");

    checkEqual(run({"1. Open the door. "}, splitOnly),
               {"1. Open the door."},
               "a list marker at the start of a line is not a sentence end");

    // the line break itself is trimmed with the surrounding whitespace, so it has to be replaced or
    // the segment reaches the TTS with nothing to speak it as a sentence
    checkEqual(run({"First line\nSecond line\n"}, splitOnly),
               {"First line.", "Second line."},
               "a line break ends a sentence, and becomes a period");

    checkEqual(run({"First line.\nSecond line?\nThird line…\n"}, splitOnly),
               {"First line.", "Second line?", "Third line…"},
               "a line already ending in punctuation does not get a second one");

    checkEqual(run({"You will need:\nA hammer,\nand a nail\n"}, splitOnly),
               {"You will need:", "A hammer,", "and a nail."},
               "a line ending in a connector keeps the punctuation the model wrote");

    checkEqual(run({"First line\nSecond line\n"}, { .minSentenceLength = 0, .coalesceMinChars = 60 }),
               {"First line. Second line."},
               "coalesced lines do not run together");

    checkEqual(run({"Wait for it… Then go. "}, splitOnly),
               {"Wait for it…", "Then go."},
               "an ellipsis ends a sentence");

    checkEqual(run({"Really?! No way! "}, splitOnly),
               {"Really?!", "No way!"},
               "? and ! end a sentence");

    checkEqual(run({"  Hello there.  "}, splitOnly),
               {"Hello there."},
               "every segment is stripped");

    // minimumSentenceLength: a sentence end below the floor is ignored
    checkEqual(run({"Yes. No. Maybe so, and here is a longer clause. "},
                   { .minSentenceLength = 24, .coalesceMinChars = 0 }),
               {"Yes. No. Maybe so, and here is a longer clause."},
               "sentence ends below the minimum length are ignored");
}

static void testThinkFiltering() {
    checkEqual(run({"<think>reasoning</think>Hello there. "}, splitOnly),
               {"Hello there."},
               "a think block is removed");

    checkEqual(run({"One. <think>first</think>Two. <think>second</think>Three. "}, splitOnly),
               {"One.", "Two.", "Three."},
               "multiple think blocks are removed");

    checkEqual(run({"<think>", "I should ", "say hi"}, splitOnly),
               {},
               "an unterminated think block is dropped, not spoken");

    checkEqual(run({"Hello there. <think>reasoning"}, splitOnly),
               {"Hello there."},
               "text before an unterminated block survives it");

    checkEqual(run({"<thi", "nk>reasoning</thi", "nk>Hello there. "}, splitOnly),
               {"Hello there."},
               "tags split across pieces are still matched");

    checkEqual(run({"Hello there friend<thi"}, splitOnly),
               {"Hello there friend<thi."},
               "a held partial tag is released on flush");

    checkEqual(run({"5 < 10 and 3 > 1, so there. "}, splitOnly),
               {"5 < 10 and 3 > 1, so there."},
               "a bare '<' is not mistaken for a tag");

    checkEqual(run({"<think>reasoning</think>Hello there. And more. "}, splitOnly),
               {"Hello there.", "And more."},
               "text after the closing tag is processed in the same call");

    const std::string text = "Before. <think>hidden</think> After the block. ";
    checkEqual(runByBytes(text, splitOnly), run({text}, splitOnly),
               "the result is independent of chunk size");
}

static void testUtf8() {
    // U+1F600, four bytes: split at each of its three interior boundaries
    const std::string emoji = "\xF0\x9F\x98\x80";
    for (size_t cut = 1; cut < 4; ++cut) {
        std::vector<std::string> pieces { "hi " + emoji.substr(0, cut), emoji.substr(cut) + " there. " };
        checkEqual(run(pieces, splitOnly), {"hi " + emoji + " there."},
                   std::format("a 4-byte character split at byte {} is reassembled", cut));
    }

    checkEqual(run({"Gr\xC3", "\xBC\xC3\x9F e. "}, splitOnly),
               {"Gr\xC3\xBC\xC3\x9F e."},
               "2-byte characters split across pieces are reassembled");

    checkEqual(run({"Hello there\xC3"}, splitOnly),
               {"Hello there."},
               "a stream ending mid-sequence drops the incomplete tail");

    checkEqual(run({"He\xFFllo there. "}, splitOnly),
               {"Hello there."},
               "an invalid byte is dropped");

    checkEqual(run({"He\xC3llo there. "}, splitOnly),
               {"Hello there."},
               "a lead byte without a continuation is dropped");

    checkEqual(run({"He\x80llo there. "}, splitOnly),
               {"Hello there."},
               "a stray continuation byte is dropped");
}

static void testCoalescing() {
    const SegmenterConfig coalescing { .minSentenceLength = 0, .coalesceMinChars = 60 };

    // six short sentences, each well under the threshold
    const std::vector<std::string> shorts { "One two. Three four. Five six. Seven eight. Nine ten. Eleven twelve. " };
    std::vector<std::string> merged = run(shorts, coalescing);
    check(merged.size() < 6, "short segments are merged");
    check(!merged.empty() && merged.front().size() >= 60, "merging continues until the threshold is met");
    check(collapse(join(merged)) == collapse(shorts.front()), "coalescing loses no text");

    const std::string longSentence = "This one sentence is already comfortably past the coalescing threshold on its own. ";
    checkEqual(run({longSentence + "And a short tail. "}, coalescing),
               {collapse(longSentence), "And a short tail."},
               "an already-long segment is left alone and the tail is flushed");

    checkEqual(run({"One two. Three four. "}, { .minSentenceLength = 0, .coalesceMinChars = 0 }),
               {"One two.", "Three four."},
               "coalescing is disabled at 0");

    // the first segment is coalesced like any other: a short opener is exactly the one
    // VoiceDesign renders as somebody else
    std::vector<std::string> opener = run({"Hi. Now here is a much longer second sentence to merge with. "}, coalescing);
    check(!opener.empty() && opener.front().starts_with("Hi. Now"),
          "the first segment is coalesced like any other");
}

/// @brief Coalescing must yield as soon as the threshold is met, not at end of input, or it would cost time-to-first-audio on every segment rather than only on short ones.
static void testCoalesceIsLazy() {
    BlockingQueue<std::string> in { 64 };
    BlockingQueue<std::string> out { 64 };

    Segmenter segmenter;
    std::jthread worker([&] { segmenter.segment(in, out); });

    in.push("This first sentence is comfortably longer than the coalescing threshold. ");

    bool yielded = false;
    for (int i = 0; i < 200 && !yielded; ++i) {
        yielded = !out.empty();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    in.close();
    check(yielded, "coalescing yields before the input is exhausted");
}

/// @brief push() returning false is how barge-in reaches this stage: buffered text is stale and must not become audio after the user interrupted.
static void testConsumerCancellation() {
    BlockingQueue<std::string> in { 64 };
    BlockingQueue<std::string> out { 64 };

    for (int i = 0; i < 50; ++i) {
        in.push("This is one more sentence in a long stream of them. ");
    }

    out.close();

    Segmenter segmenter;
    std::jthread worker([&] { segmenter.segment(in, out, { .minSentenceLength = 0, .coalesceMinChars = 0 }); });

    // it must return without waiting for the input queue to close
    for (int i = 0; i < 200 && !in.empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    check(!in.empty(), "a closed output queue stops the stage before it drains its input");
    in.close();
}

/// @brief The property the completeness tests exist to guard.
static void testNoTextIsLost() {
    const std::string answer = "Hello there. Ask Dr. Smith about the 3.14 result, please. "
                               "It was 42. Then everyone left.";
    const std::string stream = "<think>let me consider this</think>" + answer;

    check(collapse(join(run({stream}))) == collapse(answer), "no text is lost (one piece)");
    check(collapse(join(runByBytes(stream))) == collapse(answer), "no text is lost (one byte at a time)");
    check(collapse(join(run({stream}, splitOnly))) == collapse(answer), "no text is lost (no coalescing)");
}

int main() {
    testSentenceSplitting();
    testThinkFiltering();
    testUtf8();
    testCoalescing();
    testCoalesceIsLazy();
    testConsumerCancellation();
    testNoTextIsLost();

    std::println("\n{}", failures == 0 ? "all segmenter checks passed" : std::format("{} segmenter check(s) FAILED", failures));
    return failures == 0 ? 0 : 1;
}
