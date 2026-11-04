#include "VadFrameQueue.h"

#include <array>
#include <atomic>
#include <stdexcept>
#include <thread>

namespace {
void requireQueue(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runVadFrameQueueTests() {
    using Queue = smartgate::VadFrameQueue<4, 4>;
    Queue queue;
    Queue::Frame output{};

    requireQueue(!queue.pop(output), "An empty VAD queue must not produce a frame");
    requireQueue(!queue.push(nullptr), "A null frame must be rejected");

    for (int frame = 0; frame < 4; ++frame) {
        std::array<float, 4> input{
            static_cast<float>(frame), static_cast<float>(frame + 1),
            static_cast<float>(frame + 2), static_cast<float>(frame + 3)};
        requireQueue(queue.push(input.data()), "Queue should accept frames while capacity remains");
    }

    const std::array<float, 4> overflow{9.0f, 9.0f, 9.0f, 9.0f};
    requireQueue(!queue.push(overflow.data()),
                 "A full real-time queue must drop the newest frame without waiting");

    for (int frame = 0; frame < 4; ++frame) {
        requireQueue(queue.pop(output), "Queued frame should be available");
        requireQueue(output[0] == static_cast<float>(frame)
                     && output[3] == static_cast<float>(frame + 3),
                     "VAD frames must be delivered in FIFO order without corruption");
    }
    requireQueue(!queue.pop(output), "Queue should be empty after consuming all frames");

    const std::array<float, 4> wrapped{20.0f, 21.0f, 22.0f, 23.0f};
    requireQueue(queue.push(wrapped.data()), "Queue should accept frames after index wrap");
    requireQueue(queue.pop(output) && output == wrapped,
                 "Queue should preserve frames after ring-buffer wraparound");

    // Exercise the SPSC contract with producer and consumer running at the
    // same time. The producer retries on full; the queue itself never waits.
    {
        using ConcurrentQueue = smartgate::VadFrameQueue<8, 64>;
        ConcurrentQueue concurrentQueue;
        constexpr int frameCount = 20000;
        std::atomic<bool> producerDone{false};
        bool fifoIntact = true;

        std::thread producer([&] {
            std::array<float, 8> input{};
            for (int sequence = 0; sequence < frameCount; ++sequence) {
                input.fill(static_cast<float>(sequence));
                while (!concurrentQueue.push(input.data()))
                    std::this_thread::yield();
            }
            producerDone.store(true, std::memory_order_release);
        });

        std::thread consumer([&] {
            ConcurrentQueue::Frame received{};
            int expected = 0;
            while (expected < frameCount) {
                if (concurrentQueue.pop(received)) {
                    for (const float sample : received) {
                        if (sample != static_cast<float>(expected))
                            fifoIntact = false;
                    }
                    ++expected;
                } else if (producerDone.load(std::memory_order_acquire)) {
                    // The producer can only finish after every frame was
                    // accepted; keep draining until the expected count lands.
                    std::this_thread::yield();
                }
            }
        });

        producer.join();
        consumer.join();
        requireQueue(fifoIntact,
                     "Concurrent SPSC producer/consumer must preserve FIFO frames");
        requireQueue(!concurrentQueue.pop(output),
                     "Concurrent SPSC stress test must drain the queue");
    }

    queue.reset();
    requireQueue(!queue.pop(output), "Reset should empty the queue while threads are stopped");
}
