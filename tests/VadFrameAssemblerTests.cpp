#include "VadFrameAssembler.h"
#include "VadFrameQueue.h"

#include <cmath>
#include <stdexcept>

namespace {
void requireAssembler(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runVadFrameAssemblerTests() {
    using Queue = smartgate::VadFrameQueue<512, 4>;
    Queue queue;
    smartgate::VadFrameAssembler<Queue> assembler(queue);
    requireAssembler(assembler.prepare(48000.0), "Assembler should prepare for 48 kHz input");

    for (int i = 0; i < 4800; ++i)
        (void) assembler.processSample(std::sin(static_cast<float>(i) * 0.01f));

    requireAssembler(assembler.pendingSamples() == 63,
                     "Assembler should retain the partial frame across host blocks");
    requireAssembler(assembler.droppedFrames() == 0,
                     "Assembler should not drop frames while queue capacity remains");

    Queue::Frame frame{};
    requireAssembler(queue.pop(frame), "First complete model frame should be queued");
    for (const float sample : frame)
        requireAssembler(std::isfinite(sample), "Assembled model samples must be finite");
    requireAssembler(queue.pop(frame) && queue.pop(frame),
                     "All three complete frames should be available in FIFO order");
    requireAssembler(!queue.pop(frame), "Partial samples must not be queued as a full frame");

    // A full queue drops complete frames without blocking the audio producer.
    using SmallQueue = smartgate::VadFrameQueue<512, 2>;
    SmallQueue smallQueue;
    smartgate::VadFrameAssembler<SmallQueue> overflowAssembler(smallQueue);
    requireAssembler(overflowAssembler.prepare(48000.0), "Overflow assembler should prepare");
    for (int i = 0; i < 4800; ++i)
        (void) overflowAssembler.processSample(0.25f);
    requireAssembler(overflowAssembler.droppedFrames() == 1,
                     "A full queue should count and drop the newest completed frame");

    // A failed re-prepare must disable the assembler, not leave the previous
    // sample-rate configuration active and continue queuing stale-rate frames.
    requireAssembler(!overflowAssembler.prepare(8000.0),
                     "Assembler must reject unsupported upsampling");
    for (int i = 0; i < 4800; ++i)
        (void) overflowAssembler.processSample(0.5f);
    requireAssembler(overflowAssembler.pendingSamples() == 0,
                     "Unprepared assembler must not accumulate partial frames");
    requireAssembler(overflowAssembler.droppedFrames() == 0,
                     "Unprepared assembler must not attempt to queue frames");
}
