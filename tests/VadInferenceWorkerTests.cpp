#include "VadFrameQueue.h"
#include "VadInferenceWorker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace {
void requireWorker(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct InferenceProbe {
    std::atomic<int> calls{0};
    std::atomic<std::uint64_t> lastGeneration{0};
};

struct BlockingProbe {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    std::atomic<int> calls{0};
};

float blockingInference(const std::array<float, 8>& frame, std::uint64_t,
                       void* opaque) noexcept {
    auto& probe = *static_cast<BlockingProbe*>(opaque);
    probe.entered.store(true, std::memory_order_release);
    while (!probe.release.load(std::memory_order_acquire))
        std::this_thread::yield();
    probe.calls.fetch_add(1, std::memory_order_release);
    return frame[0];
}

float fakeInference(const std::array<float, 8>& frame, std::uint64_t generation,
                    void* opaque) noexcept {
    auto& probe = *static_cast<InferenceProbe*>(opaque);
    probe.lastGeneration.store(generation, std::memory_order_relaxed);
    probe.calls.fetch_add(1, std::memory_order_release);
    return frame[0];
}

bool waitForCalls(const InferenceProbe& probe, int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (probe.calls.load(std::memory_order_acquire) >= expected)
            return true;
        std::this_thread::yield();
    }
    return probe.calls.load(std::memory_order_acquire) >= expected;
}
}

void runVadInferenceWorkerTests() {
    using Queue = smartgate::VadFrameQueue<8, 8>;
    Queue queue;
    InferenceProbe probe;
    smartgate::VadInferenceWorker<Queue> worker(queue);

    requireWorker(!worker.start(nullptr, &probe, 10),
                  "Worker must reject a null inference callback");
    requireWorker(worker.start(&fakeInference, &probe, 10),
                  "Worker should start with a valid inference callback");
    requireWorker(worker.isRunning(), "Worker should report its running state");

    std::array<float, 8> staleFrame{};
    staleFrame.fill(0.95f);
    std::array<float, 8> currentFrame{};
    currentFrame.fill(0.65f);
    requireWorker(queue.push(staleFrame.data(), 9), "Queue should accept a stale test frame");
    requireWorker(queue.push(currentFrame.data(), 10), "Queue should accept the current frame");
    requireWorker(waitForCalls(probe, 1), "Worker should process the current-generation frame");
    requireWorker(probe.lastGeneration.load(std::memory_order_relaxed) == 10,
                  "Worker must discard queued frames from an earlier generation");
    const auto firstDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.latestProbability(10) != 0.65f
           && std::chrono::steady_clock::now() < firstDeadline)
        std::this_thread::yield();
    requireWorker(worker.latestProbability(10) == 0.65f,
                  "Worker should publish the inference probability for its generation");

    worker.setGeneration(11);
    requireWorker(worker.latestProbability(11) == 0.0f,
                  "A new generation must not reuse a previous generation's result");
    requireWorker(queue.push(currentFrame.data(), 10),
                  "Queue should accept another stale frame for the generation test");
    std::array<float, 8> nextFrame{};
    nextFrame.fill(0.25f);
    requireWorker(queue.push(nextFrame.data(), 11),
                  "Queue should accept the new-generation frame");
    requireWorker(waitForCalls(probe, 2),
                  "Worker should skip stale data and process the new generation");
    requireWorker(probe.lastGeneration.load(std::memory_order_relaxed) == 11,
                  "Inference callback must receive the frame's active generation");
    const auto secondDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.latestProbability(11) != 0.25f
           && std::chrono::steady_clock::now() < secondDeadline)
        std::this_thread::yield();
    requireWorker(worker.latestProbability(11) == 0.25f,
                  "Worker should publish only the current generation's result");

    worker.stop();
    requireWorker(!worker.isRunning(), "Worker should stop and join cleanly");
    requireWorker(!worker.start(nullptr, &probe, 12),
                  "Stopped worker must still reject a null callback");
    requireWorker(worker.start(&fakeInference, &probe, 12),
                  "Worker should be restartable after a completed stop");
    worker.stop();
    requireWorker(!worker.isRunning(), "Restarted worker should also stop cleanly");

    // Force an inference to overlap an AI generation change. Its result must
    // not be published after the audio thread has moved to the new generation.
    Queue raceQueue;
    BlockingProbe blockingProbe;
    smartgate::VadInferenceWorker<Queue> raceWorker(raceQueue);
    requireWorker(raceWorker.start(&blockingInference, &blockingProbe, 20),
                  "Worker should start for the in-flight generation test");
    std::array<float, 8> inFlightFrame{};
    inFlightFrame.fill(0.8f);
    requireWorker(raceQueue.push(inFlightFrame.data(), 20),
                  "In-flight test frame should be queued");
    const auto enteredDeadline = std::chrono::steady_clock::now()
        + std::chrono::seconds(2);
    while (!blockingProbe.entered.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < enteredDeadline)
        std::this_thread::yield();
    requireWorker(blockingProbe.entered.load(std::memory_order_acquire),
                  "Inference callback should begin before generation changes");

    raceWorker.setGeneration(21);
    blockingProbe.release.store(true, std::memory_order_release);
    const auto completedDeadline = std::chrono::steady_clock::now()
        + std::chrono::seconds(2);
    while (blockingProbe.calls.load(std::memory_order_acquire) == 0
           && std::chrono::steady_clock::now() < completedDeadline)
        std::this_thread::yield();
    requireWorker(blockingProbe.calls.load(std::memory_order_acquire) == 1,
                  "In-flight inference callback should complete");
    requireWorker(raceWorker.latestProbability(20) == 0.0f
                  && raceWorker.latestProbability(21) == 0.0f,
                  "Inference completed for an old generation must not be published");

    std::array<float, 8> freshFrame{};
    freshFrame.fill(0.4f);
    requireWorker(raceQueue.push(freshFrame.data(), 21),
                  "Fresh generation frame should be queued after stale inference");
    const auto freshDeadline = std::chrono::steady_clock::now()
        + std::chrono::seconds(2);
    while (raceWorker.latestProbability(21) != 0.4f
           && std::chrono::steady_clock::now() < freshDeadline)
        std::this_thread::yield();
    requireWorker(raceWorker.latestProbability(21) == 0.4f,
                  "Worker should resume publishing after the new generation arrives");
    raceWorker.stop();
}
