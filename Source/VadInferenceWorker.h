#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

namespace smartgate {

// Owns the non-real-time consumer thread for a VAD frame queue.
// The audio callback remains the sole queue producer; this worker is the sole
// consumer. Lifecycle methods must only be called from a non-audio thread.
template <typename Queue>
class VadInferenceWorker {
public:
    using Frame = typename Queue::Frame;
    using InferenceFunction = float (*)(const Frame&, std::uint64_t, void*) noexcept;

    explicit VadInferenceWorker(Queue& queue) noexcept : queue_(queue) {
        static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                      "VAD generation checks require lock-free 64-bit atomics");
        static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                      "VAD result snapshots require lock-free 32-bit atomics");
        static_assert(std::atomic<float>::is_always_lock_free,
                      "VAD probability publication requires lock-free float atomics");
    }

    ~VadInferenceWorker() { stop(); }

    VadInferenceWorker(const VadInferenceWorker&) = delete;
    VadInferenceWorker& operator=(const VadInferenceWorker&) = delete;

    // Call while the audio producer is stopped. The queue is cleared before
    // the consumer starts, so frames from an earlier run cannot leak through.
    bool start(InferenceFunction inference, void* context,
               std::uint64_t generation) noexcept {
        if (inference == nullptr || thread_.joinable())
            return false;

        queue_.reset();
        inference_ = inference;
        context_ = context;
        activeGeneration_.store(generation, std::memory_order_release);
        resultSequence_.store(0, std::memory_order_relaxed);
        publishedGeneration_.store(~std::uint64_t{0}, std::memory_order_relaxed);
        publishedProbability_.store(0.0f, std::memory_order_relaxed);
        publishedSourceSamplePosition_.store(0, std::memory_order_relaxed);
        publishedResultCount_.store(0, std::memory_order_relaxed);
        stopRequested_.store(false, std::memory_order_release);

        try {
            thread_ = std::thread(&VadInferenceWorker::run, this);
        } catch (...) {
            inference_ = nullptr;
            context_ = nullptr;
            return false;
        }

        running_.store(true, std::memory_order_release);
        return true;
    }

    // Safe for the audio thread: one lock-free store, no wakeup or allocation.
    void setGeneration(std::uint64_t generation) noexcept {
        activeGeneration_.store(generation, std::memory_order_release);
    }

    // Returns zero until a result for the requested active generation exists.
    float latestProbability(std::uint64_t generation,
                            std::uint64_t currentSamplePosition = 0,
                            std::uint64_t maxAgeSamples = ~std::uint64_t{0}) const noexcept {
        const auto sequenceBefore = resultSequence_.load(std::memory_order_seq_cst);
        if ((sequenceBefore & 1u) != 0u)
            return 0.0f;

        const auto publishedGeneration =
            publishedGeneration_.load(std::memory_order_seq_cst);
        const auto probability = publishedProbability_.load(std::memory_order_seq_cst);
        const auto sourceSamplePosition =
            publishedSourceSamplePosition_.load(std::memory_order_seq_cst);
        const auto sequenceAfter = resultSequence_.load(std::memory_order_seq_cst);

        if (sequenceBefore != sequenceAfter || (sequenceAfter & 1u) != 0u
            || publishedGeneration != generation)
            return 0.0f;
        if (maxAgeSamples != ~std::uint64_t{0}
            && (currentSamplePosition < sourceSamplePosition
                || currentSamplePosition - sourceSamplePosition > maxAgeSamples))
            return 0.0f;
        return probability;
    }

    bool isRunning() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    // Monotonic diagnostic counter; incremented only after a complete result
    // snapshot has been published. Useful to observe worker progress off-thread.
    std::uint64_t publishedResultCount() const noexcept {
        return publishedResultCount_.load(std::memory_order_acquire);
    }

    // Stop/join only from a non-audio thread. In-flight inference may finish,
    // but its result is discarded once shutdown has been requested.
    void stop() noexcept {
        stopRequested_.store(true, std::memory_order_release);
        if (thread_.joinable())
            thread_.join();
        running_.store(false, std::memory_order_release);
        inference_ = nullptr;
        context_ = nullptr;
    }

private:
    void run() noexcept {
        Frame frame{};
        while (!stopRequested_.load(std::memory_order_acquire)) {
            std::uint64_t frameGeneration = 0;
            std::uint64_t sourceSamplePosition = 0;
            if (!queue_.pop(frame, frameGeneration, sourceSamplePosition)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Inference may be slower than real time on a busy machine. Do
            // not spend several future frames evaluating stale audio: drain the
            // SPSC queue and retain only its newest complete frame. The producer
            // never waits or changes its indices, and inference still runs once
            // per worker iteration at most.
            Frame newestFrame{};
            std::uint64_t newestGeneration = frameGeneration;
            std::uint64_t newestSourceSamplePosition = sourceSamplePosition;
            while (queue_.pop(newestFrame, newestGeneration, newestSourceSamplePosition)) {
                frame = newestFrame;
                frameGeneration = newestGeneration;
                sourceSamplePosition = newestSourceSamplePosition;
            }

            const auto generation = activeGeneration_.load(std::memory_order_acquire);
            if (frameGeneration != generation)
                continue;

            const float rawProbability = inference_(frame, frameGeneration, context_);
            // Recheck after inference: AI may have been disabled/re-enabled
            // while the model was evaluating this frame.
            if (stopRequested_.load(std::memory_order_acquire)
                || activeGeneration_.load(std::memory_order_acquire) != frameGeneration)
                continue;

            const float probability = std::isfinite(rawProbability)
                ? std::max(0.0f, std::min(1.0f, rawProbability)) : 0.0f;
            // Publish the generation/probability pair as one bounded snapshot.
            // The audio reader never spins: an in-progress or changing snapshot
            // simply produces a neutral result for this block.
            resultSequence_.fetch_add(1, std::memory_order_seq_cst);
            publishedProbability_.store(probability, std::memory_order_seq_cst);
            publishedGeneration_.store(frameGeneration, std::memory_order_seq_cst);
            publishedSourceSamplePosition_.store(sourceSamplePosition, std::memory_order_seq_cst);
            resultSequence_.fetch_add(1, std::memory_order_seq_cst);
            publishedResultCount_.fetch_add(1, std::memory_order_release);
        }
        running_.store(false, std::memory_order_release);
    }

    Queue& queue_;
    InferenceFunction inference_ = nullptr;
    void* context_ = nullptr;
    std::thread thread_;
    std::atomic<bool> stopRequested_{true};
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> activeGeneration_{0};
    std::atomic<std::uint32_t> resultSequence_{0};
    std::atomic<std::uint64_t> publishedGeneration_{~std::uint64_t{0}};
    std::atomic<float> publishedProbability_{0.0f};
    std::atomic<std::uint64_t> publishedSourceSamplePosition_{0};
    std::atomic<std::uint64_t> publishedResultCount_{0};
};

} // namespace smartgate
