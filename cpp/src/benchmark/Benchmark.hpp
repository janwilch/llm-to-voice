#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#if defined(__linux__)
#include <dlfcn.h>
#include <sys/resource.h>
#include <time.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace bench {

/// @brief (Device-wide) GPU state snapshot.
struct GpuSample {
    uint64_t usedBytes = 0;
    uint64_t totalBytes = 0;
    /// @brief Share of the last driver sample window (~1/6 s to 1 s) in which a kernel was running.
    std::optional<uint32_t> utilPercent;
};

/// @brief Everything measured at one point in time.
struct Snapshot {
    std::chrono::steady_clock::time_point wall;
    /// @brief User + system CPU time of the whole process, summed over all threads.
    double cpuSeconds = 0.0;
    /// @brief Resident set size right now. 0 where unsupported.
    uint64_t rssBytes = 0;
    /// @brief Peak resident set size since process start. It never shrinks, so a delta only shows growth past the previous peak.
    uint64_t peakRssBytes = 0;
    std::optional<GpuSample> gpu;
};

namespace detail {

#if defined(__linux__)

inline double processCpuSeconds() {
    timespec ts {};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

/// @brief Reads a `Key:   1234 kB` line out of /proc/self/status, in bytes.
inline uint64_t procStatusKb(const std::string_view key) {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.starts_with(key) && line.size() > key.size() && line[key.size()] == ':') {
            return std::strtoull(line.c_str() + key.size() + 1, nullptr, 10) * 1024;
        }
    }
    return 0;
}

inline void* openLibrary() {
    return dlopen("libnvidia-ml.so.1", RTLD_NOW);
}

inline void* librarySymbol(void* lib, const char* name) {
    return dlsym(lib, name);
}

#elif defined(_WIN32)

inline double processCpuSeconds() {
    FILETIME creation {}, exit {}, kernel {}, user {};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        return 0.0;
    }
    // FILETIME counts 100 ns ticks
    const auto ticks = [](const FILETIME& ft) {
        return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    return static_cast<double>(ticks(kernel) + ticks(user)) * 1e-7;
}

/// @brief Working set now and at its peak, in bytes: what VmRSS and VmHWM are on Linux.
inline std::pair<uint64_t, uint64_t> processMemory() {
    PROCESS_MEMORY_COUNTERS counters {};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) {
        return { 0, 0 };
    }
    return { counters.WorkingSetSize, counters.PeakWorkingSetSize };
}

/// @brief nvml.dll ships with the driver in System32; older drivers only put it into NVSMI.
inline void* openLibrary() {
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (lib == nullptr) {
        lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    }
    return reinterpret_cast<void*>(lib);
}

inline void* librarySymbol(void* lib, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
}

#endif

#if defined(__linux__) || defined(_WIN32)

/// @brief NVIDIA driver library, opened at runtime (doesn't need nvml.h or the CUDA toolkit; runs without NVIDIA GPU).
class Nvml {
    // Layouts of nvmlMemory_t and nvmlUtilization_t from nvml.h
    struct Memory {
        unsigned long long total;
        unsigned long long free;
        unsigned long long used;
    };
    struct Utilization {
        unsigned int gpu;
        unsigned int memory;
    };

    using Device = void*;
    using InitFn = int (*)();
    using GetHandleFn = int (*)(unsigned int, Device*);
    using GetMemoryFn = int (*)(Device, Memory*);
    using GetUtilizationFn = int (*)(Device, Utilization*);

    void* _lib = nullptr;
    Device _device = nullptr;
    GetMemoryFn _getMemory = nullptr;
    GetUtilizationFn _getUtilization = nullptr;

public:
    /// @brief Binds to GPU 0 (NVML numbers only NVIDIA devices, so iGPU is not captured)
    Nvml() {
        _lib = openLibrary();
        if (_lib == nullptr) {
            return;
        }

        const auto init = reinterpret_cast<InitFn>(librarySymbol(_lib, "nvmlInit_v2"));
        const auto getHandle = reinterpret_cast<GetHandleFn>(librarySymbol(_lib, "nvmlDeviceGetHandleByIndex_v2"));
        _getMemory = reinterpret_cast<GetMemoryFn>(librarySymbol(_lib, "nvmlDeviceGetMemoryInfo"));
        _getUtilization = reinterpret_cast<GetUtilizationFn>(librarySymbol(_lib, "nvmlDeviceGetUtilizationRates"));

        // 0 == NVML_SUCCESS
        if (init == nullptr || getHandle == nullptr || _getMemory == nullptr || init() != 0 || getHandle(0, &_device) != 0) {
            _getMemory = nullptr;
        }
    }

    // NVML is left initialised
    Nvml(const Nvml&) = delete;
    Nvml& operator=(const Nvml&) = delete;

    [[nodiscard]] bool available() const { return _getMemory != nullptr; }

    [[nodiscard]] std::optional<GpuSample> sample() const {
        Memory mem {};
        if (!available() || _getMemory(_device, &mem) != 0) {
            return std::nullopt;
        }

        GpuSample out { .usedBytes = mem.used, .totalBytes = mem.total };
        Utilization util {};
        if (_getUtilization != nullptr && _getUtilization(_device, &util) == 0) {
            out.utilPercent = util.gpu;
        }
        return out;
    }
};

#endif

#if defined(__linux__)

inline std::optional<uint64_t> readU64(const std::filesystem::path& path) {
    std::ifstream in(path);
    uint64_t value = 0;
    if (in >> value) {
        return value;
    }
    return std::nullopt;
}

/// @brief NOTE - UNTESTED AMD GPU support
inline std::optional<GpuSample> sampleAmdSysfs() {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const fs::directory_entry& card : fs::directory_iterator("/sys/class/drm", ec)) {
        const fs::path device = card.path() / "device";
        std::ifstream vendorFile(device / "vendor");
        std::string vendor;
        if (!(vendorFile >> vendor) || vendor != "0x1002") {
            continue;
        }

        const auto used = readU64(device / "mem_info_vram_used");
        const auto total = readU64(device / "mem_info_vram_total");
        if (!used || !total) {
            continue;
        }

        GpuSample out { .usedBytes = *used, .totalBytes = *total };
        if (const auto busy = readU64(device / "gpu_busy_percent")) {
            out.utilPercent = static_cast<uint32_t>(*busy);
        }
        return out;
    }
    return std::nullopt;
}

/// @brief NVIDIA through NVML if present, otherwise AMD through sysfs.
inline std::optional<GpuSample> sampleGpu() {
    static const Nvml nvml;
    if (nvml.available()) {
        return nvml.sample();
    }
    return sampleAmdSysfs();
}

#elif defined(_WIN32)

/// @brief NVIDIA through NVML only.
inline std::optional<GpuSample> sampleGpu() {
    static const Nvml nvml;
    return nvml.sample();
}

#endif

inline double toMiB(const uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline double signedMiB(const uint64_t after, const uint64_t before) {
    return after >= before ? toMiB(after - before) : -toMiB(before - after);
}

/// @brief ` (46%)`, or nothing when the stage is too short for CPU time / wall time to mean anything (both are clock-granularity noise below ~1 ms).
inline std::string cpuShare(const double wallMs, const double cpuMs) {
    constexpr double MIN_WALL_MS = 1.0;
    if (wallMs < MIN_WALL_MS) {
        return "";
    }
    return std::format(" ({:.0f}%)", cpuMs / wallMs * 100.0);
}

} // namespace detail

/// @brief Records wall clock, process CPU time, RSS and (where a supported GPU exists) GPU memory and utilisation.
inline Snapshot snapshot() {
    Snapshot s;
    s.wall = std::chrono::steady_clock::now();
#if defined(__linux__)
    s.cpuSeconds = detail::processCpuSeconds();
    s.rssBytes = detail::procStatusKb("VmRSS");
    s.peakRssBytes = detail::procStatusKb("VmHWM");
    s.gpu = detail::sampleGpu();
#elif defined(_WIN32)
    s.cpuSeconds = detail::processCpuSeconds();
    std::tie(s.rssBytes, s.peakRssBytes) = detail::processMemory();
    s.gpu = detail::sampleGpu();
#endif
    return s;
}

inline double elapsedMs(const Snapshot& before, const Snapshot& after) {
    return std::chrono::duration<double, std::milli>(after.wall - before.wall).count();
}

/// @brief The first time something happens, recorded from any thread. Only the first `hit()` counts.
class Mark {
    static constexpr int64_t UNSET = -1;
    std::atomic<int64_t> _ns { UNSET };

public:
    void hit() {
        if (_ns.load(std::memory_order_relaxed) != UNSET) {
            return;
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        int64_t expected = UNSET;
        _ns.compare_exchange_strong(expected, std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(), std::memory_order_relaxed);
    }

    /// @return Empty if `hit()` was never called.
    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point> when() const {
        const int64_t ns = _ns.load(std::memory_order_relaxed);
        if (ns == UNSET) {
            return std::nullopt;
        }
        return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
    }
};

/// @brief Prints how long after `origin` something happened, e.g. time to first audio.
inline void printSince(const std::string_view label, const Snapshot& origin, const std::optional<std::chrono::steady_clock::time_point> when) {
    if (!when) {
        std::println(stderr, "[bench] {}: not reached", label);
        return;
    }
    std::println(stderr, "[bench] {}: +{:.1f} ms", label, std::chrono::duration<double, std::milli>(*when - origin.wall).count());
}

/// @brief Prints what happened between two snapshots, on stderr so it does not mix with the CLI's own output.
/// GPU utilisation is the driver's reading at the end, i.e. the last fraction of a second only: it is identical for stages ending within the same window and says nothing about a long stage as a whole.
inline void printDelta(const std::string_view label, const Snapshot& before, const Snapshot& after) {
    const double wallMs = elapsedMs(before, after);
    const double cpuMs = (after.cpuSeconds - before.cpuSeconds) * 1000.0;

    std::println(stderr, "[bench] {}: wall {:.1f} ms | cpu {:.1f} ms{} | rss {:.1f} MiB ({:+.1f} MiB), peak {:.1f} MiB",
        label,
        wallMs,
        cpuMs,
        detail::cpuShare(wallMs, cpuMs),
        detail::toMiB(after.rssBytes),
        detail::signedMiB(after.rssBytes, before.rssBytes),
        detail::toMiB(after.peakRssBytes));

    if (after.gpu && before.gpu) {
        std::print(stderr, "[bench] {}: gpu mem {:.0f}/{:.0f} MiB ({:+.0f} MiB)",
            label,
            detail::toMiB(after.gpu->usedBytes),
            detail::toMiB(after.gpu->totalBytes),
            detail::signedMiB(after.gpu->usedBytes, before.gpu->usedBytes));

        if (after.gpu->utilPercent) {
            std::print(stderr, " | gpu util {}%", *after.gpu->utilPercent);
        }
        std::println(stderr);
    }
}

/// @brief Measures the scope it lives in and prints the delta when it ends.
/// @code
/// {
///     bench::Scoped timer("llm prompt eval");
///     runLlm();
/// } // prints here
/// @endcode
/// GPU work is asynchronous: make sure the scope ends only after the result has been read back or the backend synchronised, or the time is just kernel launch.
class Scoped {
    std::string _label;
    Snapshot _start;

public:
    explicit Scoped(std::string label) : _label(std::move(label)), _start(snapshot()) {}

    Scoped(const Scoped&) = delete;
    Scoped& operator=(const Scoped&) = delete;

    ~Scoped() { printDelta(_label, _start, snapshot()); }
};

} // namespace bench
