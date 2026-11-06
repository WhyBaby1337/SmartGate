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
    std::atomic<std::uint64_t> lastSourceSamplePosition{0};
};

struct BlockingProbe {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
    std::atomic<int> calls{0};
    std::atomic<float> lastInput{0.0f};
    std::atomic<std::uint64_t> lastSourceSamplePosition{0};
};

float blockingInference(const std::array<float, 8>& frame, std::uint64_t,
                       std::uint64_t sourceSamplePosition, void* opaque) noexcept {
    auto& probe = *static_cast<BlockingProbe*>(opaque);
    probe.lastSourceSamplePosition.store(sourceSamplePosition, std::memory_order_relaxed);
    probe.lastInput.store(frame[0], std::memory_order_relaxed);
    probe.entered.store(true, std::memory_order_release);
    while (!probe.release.load(std::memory_order_acquire))
        std::this_thread::yield();
    probe.calls.fetch_add(1, std::memory_order_release);
    return frame[0];
}

float fakeInference(const std::array<float, 8>& frame, std::uint64_t generation,
                    std::uint64_t sourceSamplePosition, void* opaque) noexcept {
    auto& probe = *static_cast<InferenceProbe*>(opaque);
    probe.lastSourceSamplePosition.store(sourceSamplePosition, std::memory_order_relaxed);
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
    requireWorker(probe.lastSourceSamplePosition.load(std::memory_order_relaxed) == 900,
                  "Worker must pass the source audio position to its inference callback");
    requireWorker(worker.latestProbability(10, 1100, 200) == 0.65f,
                  "A result exactly at the configured sample-age boundary should remain usable");
    requireWorker(worker.latestProbability(10, 1201, 200) == 0.0f,
                  "A result older than the configured sample-age budget must be neutral");
    requireWorker(worker.latestProbability(10, 899, 200) == 0.0f,
                  "A result timestamped in the future must be rejected as inconsistent");

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
    // Reuse the same generation across a lifecycle restart. A previous
    // run's snapshot must not become visible before fresh inference completes.
    requireWorker(worker.start(&fakeInference, &probe, 11),
                  "Worker should be restartable after a completed stop");
    requireWorker(worker.latestProbability(11) == 0.0f,
                  "Restart must clear a previously published result even when the generation is reused");
    requireWorker(worker.publishedResultCount() == 0,
                  "Published-result diagnostics must restart with a clean worker lifecycle");

    const auto resultsBeforeClamp = worker.publishedResultCount();
    std::array<float, 8> aboveRangeFrame{};
    aboveRangeFrame.fill(2.0f);
    requireWorker(queue.push(aboveRangeFrame.data(), 11),
                  "Queue should accept a frame used to test probability clamping");
    requireWorker(waitForCalls(probe, 3),
                  "Restarted worker should process the clamping test frame");
    const auto clampDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.publishedResultCount() == resultsBeforeClamp
           && std::chrono::steady_clock::now() < clampDeadline)
        std::this_thread::yield();
    requireWorker(worker.publishedResultCount() > resultsBeforeClamp,
                  "Worker should finish publishing the clamping test result");
    requireWorker(worker.latestProbability(11) == 1.0f,
                  "Worker must clamp probabilities above one");

    const auto resultsBeforeNonFinite = worker.publishedResultCount();
    std::array<float, 8> nonFiniteFrame{};
    nonFiniteFrame.fill(std::numeric_limits<float>::quiet_NaN());
    requireWorker(queue.push(nonFiniteFrame.data(), 11),
                  "Queue should accept a frame used to test non-finite probabilities");
    requireWorker(waitForCalls(probe, 4),
                  "Worker should process the non-finite probability test frame");
    const auto finiteDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.publishedResultCount() == resultsBeforeNonFinite
           && std::chrono::steady_clock::now() < finiteDeadline)
        std::this_thread::yield();
    requireWorker(worker.publishedResultCount() > resultsBeforeNonFinite,
                  "Worker should finish publishing the non-finite test result");
    requireWorker(worker.latestProbability(11) == 0.0f,
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
        requireWorker(freshProbe.lastSourceSamplePosition.load(std::memory_order_relaxed) == 400,
                      "Newest-frame inference must retain its source audio position");
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

    // Shutdown while inference is in flight must join safely and suppress
    // the result produced by the callback that was already running.
    {
        Queue shutdownQueue;
        BlockingProbe shutdownProbe;
        smartgate::VadInferenceWorker<Queue> shutdownWorker(shutdownQueue);
        requireWorker(shutdownWorker.start(&blockingInference, &shutdownProbe, 25),
                      "Worker should start for the in-flight shutdown test");
        std::array<float, 8> shutdownFrame{};
        shutdownFrame.fill(0.7f);
        requireWorker(shutdownQueue.push(shutdownFrame.data(), 25),
                      "Shutdown test frame should be queued");
        const auto shutdownEnteredDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (!shutdownProbe.entered.load(std::memory_order_acquire)
               && std::chrono::steady_clock::now() < shutdownEnteredDeadline)
            std::this_thread::yield();
        const bool shutdownInferenceEntered =
            shutdownProbe.entered.load(std::memory_order_acquire);
        if (!shutdownInferenceEntered)
            shutdownProbe.release.store(true, std::memory_order_release);
        requireWorker(shutdownInferenceEntered,
                      "Inference should be in flight before shutdown begins");

        std::thread releaseInference([&shutdownProbe] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            shutdownProbe.release.store(true, std::memory_order_release);
        });
        shutdownWorker.stop();
        releaseInference.join();
        requireWorker(!shutdownWorker.isRunning(),
                      "Shutdown must join the in-flight worker");
        requireWorker(shutdownWorker.latestProbability(25) == 0.0f,
                      "An inference result completing during shutdown must not be published");
        requireWorker(shutdownWorker.publishedResultCount() == 0,
                      "Shutdown must not increment the published-result counter");
    }

    // Restart must clear frames left in the SPSC queue after a worker stops.
    // This models a shutdown during a busy audio period followed by prepareToPlay.
    {
        Queue restartQueue;
        BlockingProbe restartProbe;
        smartgate::VadInferenceWorker<Queue> restartWorker(restartQueue);
        requireWorker(restartWorker.start(&blockingInference, &restartProbe, 26),
                      "Worker should start for the queued-frame restart test");
        std::array<float, 8> activeFrame{};
        activeFrame.fill(0.3f);
        requireWorker(restartQueue.push(activeFrame.data(), 26),
                      "Active inference frame should be queued before restart");
        const auto activeDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (!restartProbe.entered.load(std::memory_order_acquire)
               && std::chrono::steady_clock::now() < activeDeadline)
            std::this_thread::yield();
        const bool activeEntered = restartProbe.entered.load(std::memory_order_acquire);
        if (!activeEntered)
            restartProbe.release.store(true, std::memory_order_release);
        requireWorker(activeEntered,
                      "Worker inference should be active before queuing restart backlog");

        std::array<float, 8> staleQueuedFrame{};
        staleQueuedFrame.fill(0.9f);
        requireWorker(restartQueue.push(staleQueuedFrame.data(), 26, 999),
                      "A stale frame should remain queued behind the active inference");
        std::thread releaseRestartInference([&restartProbe] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            restartProbe.release.store(true, std::memory_order_release);
        });
        restartWorker.stop();
        releaseRestartInference.join();
        requireWorker(!restartWorker.isRunning(),
                      "Original worker should be fully stopped before restart");

        InferenceProbe freshRestartProbe;
        requireWorker(restartWorker.start(&fakeInference, &freshRestartProbe, 26),
                      "Worker should restart with the same generation");
        requireWorker(restartWorker.latestProbability(26) == 0.0f
                      && restartWorker.publishedResultCount() == 0,
                      "Restart must clear prior results and diagnostics");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        requireWorker(freshRestartProbe.calls.load(std::memory_order_acquire) == 0,
                      "Restart must discard frames queued before the previous worker stopped");

        std::array<float, 8> freshRestartFrame{};
        freshRestartFrame.fill(0.45f);
        requireWorker(restartQueue.push(freshRestartFrame.data(), 26, 1200),
                      "Fresh post-restart frame should be accepted");
        requireWorker(waitForCalls(freshRestartProbe, 1),
                      "Restarted worker should process fresh frames normally");
        const auto restartResultDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (restartWorker.latestProbability(26) != 0.45f
               && std::chrono::steady_clock::now() < restartResultDeadline)
            std::this_thread::yield();
        requireWorker(restartWorker.latestProbability(26) == 0.45f,
                      "Restarted worker should publish only fresh post-restart inference");
        restartWorker.stop();
    }

    // A real-time stop request must be non-blocking while inference is in flight.
    // The owner joins later from its non-audio lifecycle path.
    {
        Queue requestQueue;
        BlockingProbe requestProbe;
        smartgate::VadInferenceWorker<Queue> requestWorker(requestQueue);
        requireWorker(requestWorker.start(&blockingInference, &requestProbe, 27),
                      "Worker should start for the real-time stop-request test");
        std::array<float, 8> requestFrame{};
        requestFrame.fill(0.6f);
        requireWorker(requestQueue.push(requestFrame.data(), 27),
                      "Stop-request test frame should be queued");
        const auto requestEnteredDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (!requestProbe.entered.load(std::memory_order_acquire)
               && std::chrono::steady_clock::now() < requestEnteredDeadline)
            std::this_thread::yield();
        const bool requestInferenceEntered =
            requestProbe.entered.load(std::memory_order_acquire);
        if (!requestInferenceEntered)
            requestProbe.release.store(true, std::memory_order_release);
        requireWorker(requestInferenceEntered,
                      "Inference should be in flight before the stop request");

        requestWorker.requestStop();
        requestProbe.release.store(true, std::memory_order_release);
        const auto requestStoppedDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (requestWorker.isRunning()
               && std::chrono::steady_clock::now() < requestStoppedDeadline)
            std::this_thread::yield();
        requireWorker(!requestWorker.isRunning(),
                      "Real-time stop request should let the worker exit after inference returns");
        requireWorker(requestWorker.latestProbability(27) == 0.0f
                      && requestWorker.publishedResultCount() == 0,
                      "A result completing after a stop request must not be published");
        requestWorker.stop();
    }

    // Sustained producer/worker traffic: the producer must never wait on
    // inference, and after a burst the worker must converge to the newest
    // accepted frame instead of publishing an old backlog indefinitely.
    {
        Queue sustainedQueue;
        InferenceProbe sustainedProbe;
        smartgate::VadInferenceWorker<Queue> sustainedWorker(sustainedQueue);
        requireWorker(sustainedWorker.start(&fakeInference, &sustainedProbe, 30),
                      "Worker should start for sustained traffic stress");

        constexpr int burstFrames = 5000;
        for (int sequence = 0; sequence < burstFrames; ++sequence) {
            std::array<float, 8> frame{};
            frame.fill(0.01f + static_cast<float>(sequence % 800) * 0.001f);
            while (!sustainedQueue.push(frame.data(), 30,
                                        static_cast<std::uint64_t>(sequence) * 512u))
                std::this_thread::yield();
        }

        std::array<float, 8> finalFrame{};
        finalFrame.fill(0.9375f);
        constexpr std::uint64_t finalSourcePosition =
            static_cast<std::uint64_t>(burstFrames) * 512u;
        const auto finalDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(3);
        while (!sustainedQueue.push(finalFrame.data(), 30, finalSourcePosition)
               && std::chrono::steady_clock::now() < finalDeadline)
            std::this_thread::yield();
        while (sustainedWorker.latestProbability(30) != 0.9375f
               && std::chrono::steady_clock::now() < finalDeadline)
            std::this_thread::yield();
        requireWorker(sustainedWorker.latestProbability(30) == 0.9375f,
                      "Worker must eventually publish the final frame after sustained traffic");
        requireWorker(sustainedProbe.lastSourceSamplePosition.load(
                          std::memory_order_relaxed) == finalSourcePosition,
                      "Sustained traffic must preserve the newest frame source position");
        requireWorker(sustainedProbe.calls.load(std::memory_order_acquire) > 0,
                      "Sustained traffic must execute inference callbacks");
        sustainedWorker.stop();
    }

    // An idle worker must also observe the lock-free stop request and exit
    // without requiring another audio frame to wake it.
    {
        Queue idleQueue;
        InferenceProbe idleProbe;
        smartgate::VadInferenceWorker<Queue> idleWorker(idleQueue);
        requireWorker(idleWorker.start(&fakeInference, &idleProbe, 31),
                      "Worker should start for idle stop-request coverage");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        idleWorker.requestStop();
        const auto idleStopDeadline = std::chrono::steady_clock::now()
            + std::chrono::seconds(2);
        while (idleWorker.isRunning()
               && std::chrono::steady_clock::now() < idleStopDeadline)
            std::this_thread::yield();
        requireWorker(!idleWorker.isRunning(),
                      "Idle worker must exit after a stop request without new frames");
        requireWorker(idleProbe.calls.load(std::memory_order_acquire) == 0,
                      "Idle stop-request test must not invoke inference without a frame");
        idleWorker.stop();
    }
}
