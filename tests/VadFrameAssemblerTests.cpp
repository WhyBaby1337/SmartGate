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
    requireAssembler(assembler.prepare(48000.0, 16000.0, 42),
                     "Assembler should prepare for 48 kHz input");

    for (int i = 0; i < 4800; ++i)
        (void) assembler.processSample(std::sin(static_cast<float>(i) * 0.01f));

    requireAssembler(assembler.pendingSamples() == 64,
                     "Assembler should retain the partial frame across host blocks");
    requireAssembler(assembler.droppedFrames() == 0,
                     "Assembler should not drop frames while queue capacity remains");

    Queue::Frame frame{};
    std::uint64_t generation = 0;
    requireAssembler(queue.pop(frame, generation), "First complete model frame should be queued");
    requireAssembler(generation == 42, "Assembled frame must retain its configured generation");
    for (const float sample : frame)
        requireAssembler(std::isfinite(sample), "Assembled model samples must be finite");
    requireAssembler(queue.pop(frame, generation) && queue.pop(frame, generation),
                     "All three complete frames should be available in FIFO order");
    requireAssembler(!queue.pop(frame), "Partial samples must not be queued as a full frame");

    // Exercise the full assembler at the model's native rate: exactly one
    // model frame of host input must immediately become one queued frame.
    using NativeRateQueue = smartgate::VadFrameQueue<512, 2>;
    NativeRateQueue nativeRateQueue;
    smartgate::VadFrameAssembler<NativeRateQueue> nativeRateAssembler(nativeRateQueue);
    requireAssembler(nativeRateAssembler.prepare(16000.0, 16000.0, 77),
                     "Assembler should support native 16 kHz input");
    for (int i = 0; i < 512; ++i)
        (void) nativeRateAssembler.processSample(static_cast<float>(i) / 512.0f,
                                                 static_cast<std::uint64_t>(i));
    NativeRateQueue::Frame nativeRateFrame{};
    std::uint64_t nativeGeneration = 0;
    std::uint64_t nativeSourcePosition = 0;
    requireAssembler(nativeRateQueue.pop(nativeRateFrame, nativeGeneration,
                                          nativeSourcePosition),
                     "Exactly 512 native-rate samples must produce a complete VAD frame");
    requireAssembler(nativeGeneration == 77 && nativeSourcePosition == 511,
                     "Native-rate VAD frame must retain generation and final source position");
    requireAssembler(nativeRateFrame.front() == 0.0f
                     && nativeRateFrame.back() == 511.0f / 512.0f,
                     "Native-rate VAD frame must preserve its first and final samples");
    requireAssembler(!nativeRateQueue.pop(nativeRateFrame),
                     "Native-rate input must produce exactly one frame, not duplicates");

    // Exercise the complete assembler at 44.1 kHz, where the 16000/44100
    // ratio is non-integer. Frame timestamps must remain monotonic and close
    // to the expected host-sample distance so VAD freshness/discontinuity
    // checks do not mistake normal fractional resampling cadence for a gap.
    using FractionalRateQueue = smartgate::VadFrameQueue<512, 64>;
    FractionalRateQueue fractionalRateQueue;
    smartgate::VadFrameAssembler<FractionalRateQueue> fractionalRateAssembler(
        fractionalRateQueue);
    requireAssembler(fractionalRateAssembler.prepare(44100.0, 16000.0, 88),
                     "Assembler should support fractional 44.1 kHz conversion");
    for (int i = 0; i < 44100; ++i)
        (void) fractionalRateAssembler.processSample(
            static_cast<float>(i % 97) / 97.0f, static_cast<std::uint64_t>(i));
    std::uint64_t previousSourcePosition = 0;
    std::uint64_t sourcePosition = 0;
    std::size_t fractionalFrames = 0;
    while (fractionalRateQueue.pop(frame, generation, sourcePosition)) {
        requireAssembler(generation == 88,
                         "Fractional-rate frames must retain their generation");
        if (fractionalFrames != 0) {
            const auto sourceDelta = sourcePosition - previousSourcePosition;
            requireAssembler(sourceDelta >= 1411 && sourceDelta <= 1412,
                             "44.1 kHz frame timestamps must follow the fractional 512-sample cadence");
        }
        previousSourcePosition = sourcePosition;
        ++fractionalFrames;
    }
    requireAssembler(fractionalFrames == 31,
                     "One second at 44.1 kHz must assemble 31 complete 512-sample VAD frames");
    requireAssembler(fractionalRateAssembler.pendingSamples() == 128,
                     "Fractional-rate assembly must retain only the expected partial frame");

    // Changing generations must discard a partial frame and resampler history.
    for (int i = 0; i < 300; ++i)
        (void) assembler.processSample(0.1f);
    requireAssembler(assembler.pendingSamples() > 0,
                     "Test setup should leave a partial frame before generation change");
    assembler.setGeneration(43);
    requireAssembler(assembler.pendingSamples() == 0,
                     "Generation change must discard the old partial frame");
    for (int i = 0; i < 1600; ++i)
        (void) assembler.processSample(0.2f);
    requireAssembler(queue.pop(frame, generation) && generation == 43,
                     "Frames after a generation change must carry the new generation");
    requireAssembler(std::abs(frame.front() - 0.2f) < 1.0e-4f
                     && std::abs(frame.back() - 0.2f) < 1.0e-4f,
                     "A generation change must clear old FIR history and preserve the new stream level");

    // A full queue drops complete frames without blocking the audio producer.
    using SmallQueue = smartgate::VadFrameQueue<512, 2>;
    SmallQueue smallQueue;
    smartgate::VadFrameAssembler<SmallQueue> overflowAssembler(smallQueue);
    requireAssembler(overflowAssembler.prepare(48000.0), "Overflow assembler should prepare");
    for (int i = 0; i < 4800; ++i)
        (void) overflowAssembler.processSample(0.25f);
    requireAssembler(overflowAssembler.droppedFrames() == 1,
                     "A full queue should count and drop the newest completed frame");

    requireAssembler(!overflowAssembler.prepare(8000.0),
                     "Assembler must reject unsupported upsampling");
    for (int i = 0; i < 4800; ++i)
        (void) overflowAssembler.processSample(0.5f);
    requireAssembler(overflowAssembler.pendingSamples() == 0,
                     "Unprepared assembler must not accumulate partial frames");
    requireAssembler(overflowAssembler.droppedFrames() == 0,
                     "Unprepared assembler must not attempt to queue frames");
}
