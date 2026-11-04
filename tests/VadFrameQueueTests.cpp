#include "VadFrameQueue.h"

#include <array>
#include <stdexcept>

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

    queue.reset();
    requireQueue(!queue.pop(output), "Reset should empty the queue while threads are stopped");
}
