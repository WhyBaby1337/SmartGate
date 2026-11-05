#include "VadFrameQueue.h"
#include "VadInferenceWorker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <limits>

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
    std::atomic<float> lastInput{0.0f};
};

float blockingInference(const std::array<float, 8>& frame, std::uint64_t,
                       std::uint64_t, void* opaque) noexcept {
    auto& probe = *static_cast<BlockingProbe*>(opaque);
    probe.lastInput.store(frame[0], std::memory_order_relaxed);
    probe.entered.store(true, std::memory_order_release);
    while (!probe.release.load(std::memory_order_acquire))
        std::this_thread::yield();
    probe.calls.fetch_add(1, std::memory_order_release);
    return frame[0];
}

float fakeInference(const std::array<float, 8>& frame, std::uint64_t generation,
                    std::uint64_t, void* opaque) noexcept {
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
    requireWorker(queue.push(currentFrame.data(), 10, 900),
                  "Queue should accept the current frame with its source position");
    requireWorker(waitForCalls(probe, 1), "Worker should process the current-generation frame");
    requireWorker(probe.lastGeneration.load(std::memory_order_relaxed) == 10,
                  "Worker must discard queued frames from an earlier generation");
    const auto firstDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.latestProbability(10) != 0.65f
           && std::chrono::steady_clock::now() < firstDeadline)
        std::this_thread::yield();
    requireWorker(worker.latestProbability(10) == 0.65f,
                  "Worker should publish the inference probability for its generation");
    requireWorker(worker.latestProbability(10, 1000, 200) == 0.65f,
                  "A result inside the configured sample-age budget should remain usable");
    requireWorker(worker.latestProbability(10, 1201, 200) == 0.0f,
                  "A result older than the configured sample-age budget must be neutral");

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

    const auto resultsBeforeClamp = worker.publishedResultCount();
    std::array<float, 8> aboveRangeFrame{};
    aboveRangeFrame.fill(2.0f);
    requireWorker(queue.push(aboveRangeFrame.data(), 12),
                  "Queue should accept a frame used to test probability clamping");
    requireWorker(waitForCalls(probe, 3),
                  "Restarted worker should process the clamping test frame");
    const auto clampDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.publishedResultCount() == resultsBeforeClamp
           && std::chrono::steady_clock::now() < clampDeadline)
        std::this_thread::yield();
    requireWorker(worker.publishedResultCount() > resultsBeforeClamp,
                  "Worker should finish publishing the clamping test result");
    requireWorker(worker.latestProbability(12) == 1.0f,
                  "Worker must clamp probabilities above one");

    const auto resultsBeforeNonFinite = worker.publishedResultCount();
    std::array<float, 8> nonFiniteFrame{};
    nonFiniteFrame.fill(std::numeric_limits<float>::quiet_NaN());
    requireWorker(queue.push(nonFiniteFrame.data(), 12),
                  "Queue should accept a frame used to test non-finite probabilities");
    requireWorker(waitForCalls(probe, 4),
                  "Worker should process the non-finite probability test frame");
    const auto finiteDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.publishedResultCount() == resultsBeforeNonFinite
           && std::chrono::steady_clock::now() < finiteDeadline)
        std::this_thread::yield();
    requireWorker(worker.publishedResultCount() > resultsBeforeNonFinite,
                  "Worker should finish publishing the non-finite test result");
    requireWorker(worker.latestProbability(12) == 0.0f,
                  "Worker must convert non-finite probabilities to zero");

    worker.stop();
    requireWorker(!worker.isRunning(), "Restarted worker should also stop cleanly");

    // If inference stalls while new frames arrive, the worker must resume
    // from the newest queued audio rather than replaying a stale backlog.
    {
        Queue freshQueue;
        BlockingProbe freshProbe;
        smartgate::VadInferenceWorker<Queue> freshWorker(freshQueue);
        requireWorker(freshWorker.start(&blockingInference, &freshProbe, 15),
                      "Worker should start for the stale-backlog test");
        std::array<float, 8> firstFrame{};
        firstFrame.fill(0.1f);
        requireWorker(freshQueue.push(firstFrame.data(), 15, 100),
                      "Initial frame should enter the freshness test queue");
        const auto blockedDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (!freshProbe.entered.load(std::memory_order_acquire)
               && std::chrono::steady_clock::now() < blockedDeadline)
            std::this_thread::yield();
        const bool firstInferenceEntered =
            freshProbe.entered.load(std::memory_order_acquire);
        if (!firstInferenceEntered)
            freshProbe.release.store(true, std::memory_order_release);
        requireWorker(firstInferenceEntered,
                      "First inference should block before backlog frames are queued");

        for (const float value : {0.2f, 0.3f, 0.4f}) {
            std::array<float, 8> queuedFrame{};
            queuedFrame.fill(value);
            requireWorker(freshQueue.push(queuedFrame.data(), 15,
                                           static_cast<std::uint64_t>(value * 1000.0f)),
                          "Backlog frame should be accepted while inference is busy");
        }
        freshProbe.release.store(true, std::memory_order_release);
        const auto newestDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (freshProbe.calls.load(std::memory_order_acquire) < 2
               && std::chrono::steady_clock::now() < newestDeadline)
            std::this_thread::yield();
        requireWorker(freshProbe.calls.load(std::memory_order_acquire) >= 2,
                      "Worker should process another frame after the blocked inference");
        requireWorker(freshProbe.lastInput.load(std::memory_order_relaxed) == 0.4f,
                      "Worker should skip stale backlog and infer the newest queued frame");
        const auto freshResultDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (freshWorker.latestProbability(15) != 0.4f
               && std::chrono::steady_clock::now() < freshResultDeadline)
            std::this_thread::yield();
        requireWorker(freshWorker.latestProbability(15) == 0.4f,
                      "Published VAD result should correspond to the newest queued frame");
        freshWorker.stop();
    }

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
    const bool inferenceEntered = blockingProbe.entered.load(std::memory_order_acquire);
    if (!inferenceEntered)
        blockingProbe.release.store(true, std::memory_order_release);
    requireWorker(inferenceEntered,
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
