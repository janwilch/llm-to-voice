#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

/// @brief Thread-safe blocking queue wrapper.
/// @tparam T 
template <typename T>
class BlockingQueue {
    std::deque<T> _items;
    size_t _capacity;
    mutable std::mutex _mutex; // mutable to allow the const empty() and closed methods
    std::condition_variable _notEmpty;
    std::condition_variable _notFull;
    bool _closed = false;

public:
    explicit BlockingQueue(const size_t capacity) : _capacity(capacity) {}

    /// @brief Add to queue, if not full or closed.
    /// @param item 
    /// @return `false` if the queue is closed.
    bool push(T item) {
        std::unique_lock lock(_mutex);
        
        // equivalent to: `while(!predicate) wait(lock);`
        // while waiting, *other threads can acquire the _mutex*
        _notFull.wait(
            lock,
            [&] { return _items.size() < _capacity || _closed; }
        );

        if (_closed) {
            return false;
        }

        // std::move *moves* the item memory position into the queue, without creating another copy
        // after this, the `item` variable is unspecified
        _items.push_back(std::move(item));

        lock.unlock();
        _notEmpty.notify_one();
        return true;
    }

    /// @brief Remove from queue if not empty or closed. BLOCKING.
    /// @return An item.
    std::optional<T> pop() {
        std::unique_lock lock(_mutex);

        _notEmpty.wait(
            lock,
            [&] { return !_items.empty() || _closed; }
        );

        if (_items.empty()) {
            return std::nullopt;
        }

        T item = std::move(_items.front());
        _items.pop_front();

        lock.unlock();
        _notFull.notify_one();
        return item;
    }

    /// @brief Remove from the current queue if not empty or closed. NON-BLOCKING.
    /// @return True if an item was removed.
    bool tryPop(T& item) {
        std::lock_guard lock(_mutex);

        if (_items.empty()) {
            return false;
        }

        item = std::move(_items.front());
        _items.pop_front();

        _notFull.notify_one();
        return true;
    }

    /// @brief Close the queue, unlocking all waiting threads.
    void close() {
        {
            std::lock_guard lock(_mutex);
            _closed = true;
        }

        _notEmpty.notify_all();
        _notFull.notify_all();
    }

    bool empty() const {
        std::lock_guard lock(_mutex);
        return _items.empty();
    }

    bool closed() const {
        std::lock_guard lock(_mutex);
        return _closed;
    }
};

template <typename T>
struct QueueCloser {
    BlockingQueue<T>& q;
    ~QueueCloser() {
        q.close();
    }
};
