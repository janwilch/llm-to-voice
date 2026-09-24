#pragma once

#include <algorithm>
#include <atomic>
#include <new>
#include <vector>

/// @brief A lock-free ring buffer for *exactly* one producer and one consumer.
template <typename T>
class SpscRingBuffer {
    std::vector<T> _buffer;
    size_t _mask;
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _writePos {0};
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _readPos {0};

public:
    explicit SpscRingBuffer(const size_t capacity) :
        _buffer(std::bit_ceil(capacity)), // must round up to a power of 2
        _mask(_buffer.size() - 1) {}

    /// @brief Producer: Request `count` items in.
    /// @return The number of items actually written.
    size_t write(const T* src, const size_t count) {
        const size_t w = _writePos.load(std::memory_order_relaxed);
        const size_t r = _readPos.load(std::memory_order_acquire);

        size_t filledCount = w - r;
        const size_t freeCount = _buffer.size() - filledCount;
        const size_t n = std::min(count, freeCount);

        const size_t start = w & _mask;
        const size_t first = std::min(n, _buffer.size() - start);

        // since we are "rolling over" the buffer, copy in 2 chunks
        std::copy_n(src, first, _buffer.data() + start);
        std::copy_n(src + first, n - first, _buffer.data());

        _writePos.store(w + n, std::memory_order_release);
        return n;
    }

    /// @brief Consumer: Request `count` items out.
    /// @return The number of items actually read.
    size_t read(T* dst, const size_t count) {
        const size_t r = _readPos.load(std::memory_order_relaxed);
        const size_t w = _writePos.load(std::memory_order_acquire);

        const size_t filledCount = w - r;
        const size_t n = std::min(count, filledCount);

        const size_t start = r & _mask;
        const size_t first = std::min(n, _buffer.size() - start);

        // since we are "rolling over" the buffer, copy in 2 chunks
        std::copy_n(_buffer.data() + start, first, dst);
        std::copy_n(_buffer.data(), n - first, dst + first);

        _readPos.store(r + n, std::memory_order_release);
        return n;
    }

    /// @brief Consumer: number of items ready to read.
    size_t available() const {
        return _writePos.load(std::memory_order_acquire) - _readPos.load(std::memory_order_relaxed);
    }

    /// @brief Consumer: discard everything currently buffered.
    void clear() {
        _readPos.store(_writePos.load(std::memory_order_acquire), std::memory_order_release);
    }
};
