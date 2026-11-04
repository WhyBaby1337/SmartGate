#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace smartgate {

// Single-producer/single-consumer queue for fixed-size audio frames.
// The audio callback is the only producer; a dedicated inference worker is
// the only consumer. push() never waits and drops the newest frame when full.
template <std::size_t FrameSize, std::size_t Capacity>
class VadFrameQueue {
    static_assert(FrameSize > 0, "FrameSize must be non-zero");
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "The real-time queue requires lock-free 32-bit atomics");

public:
    using Frame = std::array<float, FrameSize>;

    bool push(const float* samples) noexcept {
        if (samples == nullptr)
            return false;

        const auto write = writeIndex_.load(std::memory_order_relaxed);
        const auto read = readIndex_.load(std::memory_order_acquire);
        if (static_cast<std::uint32_t>(write - read) >= Capacity)
            return false;

        auto& destination = frames_[write & indexMask];
        for (std::size_t i = 0; i < FrameSize; ++i)
            destination[i] = samples[i];

        writeIndex_.store(write + 1, std::memory_order_release);
        return true;
    }

    bool pop(Frame& destination) noexcept {
        const auto read = readIndex_.load(std::memory_order_relaxed);
        const auto write = writeIndex_.load(std::memory_order_acquire);
        if (read == write)
            return false;

        destination = frames_[read & indexMask];
        readIndex_.store(read + 1, std::memory_order_release);
        return true;
    }

    // Call only while producer and consumer are stopped.
    void reset() noexcept {
        readIndex_.store(0, std::memory_order_relaxed);
        writeIndex_.store(0, std::memory_order_relaxed);
    }

    static constexpr std::size_t frameSize = FrameSize;
    static constexpr std::size_t capacity = Capacity;

private:
    static constexpr std::uint32_t indexMask =
        static_cast<std::uint32_t>(Capacity - 1);

    alignas(64) std::array<Frame, Capacity> frames_{};
    alignas(64) std::atomic<std::uint32_t> writeIndex_{0};
    alignas(64) std::atomic<std::uint32_t> readIndex_{0};
};

} // namespace smartgate
