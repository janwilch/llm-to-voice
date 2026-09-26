// Checks that the benchmark helpers actually measure: a sleep shows up as wall
// time but not CPU time, a busy loop as both, and an allocation as RSS.

#include "../src/benchmark/Benchmark.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("sleeping costs wall time but almost no CPU", "[bench]") {
    const bench::Snapshot before = bench::snapshot();
    std::this_thread::sleep_for(50ms);
    const bench::Snapshot after = bench::snapshot();

    CHECK(bench::elapsedMs(before, after) >= 50.0);
    CHECK(after.cpuSeconds - before.cpuSeconds < 0.025);
}

TEST_CASE("a busy loop costs CPU time and resident memory", "[bench]") {
    const bench::Snapshot before = bench::snapshot();

    std::vector<char> block(64u * 1024 * 1024, 1); // written, so the pages are really resident
    volatile uint64_t sink = 0;
    const auto until = std::chrono::steady_clock::now() + 50ms;
    while (std::chrono::steady_clock::now() < until) {
        sink = sink + block[sink % block.size()];
    }

    const bench::Snapshot after = bench::snapshot();
    bench::printDelta("busy loop", before, after);

    CHECK(after.cpuSeconds - before.cpuSeconds >= 0.04);
#if defined(__linux__) || defined(_WIN32)
    CHECK(after.rssBytes >= before.rssBytes + 60u * 1024 * 1024);
    CHECK(after.peakRssBytes >= after.rssBytes);
#endif
}

TEST_CASE("Scoped prints on destruction", "[bench]") {
    bench::Scoped timer("scoped");
    std::this_thread::sleep_for(1ms);
}

TEST_CASE("CPU share is left out when the stage is too short to divide by", "[bench]") {
    CHECK(bench::detail::cpuShare(0.0, 0.0).empty());
    CHECK(bench::detail::cpuShare(0.4, 0.4).empty());
    CHECK(bench::detail::cpuShare(100.0, 46.0) == " (46%)");
    CHECK(bench::detail::cpuShare(100.0, 400.0) == " (400%)");
}

TEST_CASE("Mark keeps the first hit only", "[bench]") {
    bench::Mark mark;
    CHECK_FALSE(mark.when().has_value());

    mark.hit();
    const auto first = mark.when();
    REQUIRE(first.has_value());

    std::this_thread::sleep_for(5ms);
    mark.hit();
    CHECK(mark.when() == first);
}

TEST_CASE("printSince handles a mark that was never hit", "[bench]") {
    const bench::Snapshot origin = bench::snapshot();
    bench::Mark never;
    bench::printSince("never", origin, never.when());
    bench::printSince("now", origin, std::chrono::steady_clock::now());
}
