#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace smartgate {

// Single-producer/single-consumer queue for fixed-size audio frames.
// The audio callback is the only producer; a dedicated inference worker is
// the only consumer. push() never waits and drops the newest frame when full.
// Slot metadata is published with the same release/acquire handoff as samples.
template <std::size_t FrameSize, std::size_t Capacity>
class VadFrameQueue {
    static_assert(FrameSize > 0, "FrameSize must be non-zero");
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of two");
    static_assert(Capacity <= (std::uint64_t{1} << 31),
                  "Capacity must fit the unambiguous uint32_t ring-index range");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "The real-time queue requires lock-free 32-bit atomics");

public:
    using Frame = std::array<float, FrameSize>;

    bool push(const float* samples, std::uint64_t generation = 0) noexcept {
        if (samples == nullptr)
            return false;

        const auto write = writeIndex_.load(std::memory_order_relaxed);
        const auto read = readIndex_.load(std::memory_order_acquire);
        if (static_cast<std::uint32_t>(write - read) >= Capacity)
            return false;

        const auto slot = write & indexMask;
        for (std::size_t i = 0; i < FrameSize; ++i)
            frames_[slot][i] = samples[i];
        generations_[slot] = generation;
        writeIndex_.store(write + 1, std::memory_order_release);
        return true;
    }

    bool pop(Frame& destination) noexcept {
        std::uint64_t ignoredGeneration = 0;
        return pop(destination, ignoredGeneration);
    }

    bool pop(Frame& destination, std::uint64_t& generation) noexcept {
        const auto read = readIndex_.load(std::memory_order_relaxed);
        const auto write = writeIndex_.load(std::memory_order_acquire);
        if (read == write)
            return false;

        const auto slot = read & indexMask;
        destination = frames_[slot];
        generation = generations_[slot];
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
    std::array<std::uint64_t, Capacity> generations_{};
    alignas(64) std::atomic<std::uint32_t> writeIndex_{0};
    alignas(64) std::atomic<std::uint32_t> readIndex_{0};
};

} // namespace smartgate
