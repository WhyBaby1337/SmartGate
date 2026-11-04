#pragma once

#include "VadResampler.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace smartgate {

// Converts the host-rate mono stream into fixed model frames. The caller owns
// the SPSC queue; this object is used only by its audio-thread producer.
template <typename Queue>
class VadFrameAssembler {
public:
    static constexpr std::size_t frameSize = Queue::frameSize;

    explicit VadFrameAssembler(Queue& queue) noexcept : queue_(queue) {}

    bool prepare(double inputRate, double outputRate = 16000.0,
                 std::uint64_t generation = 0) noexcept {
        generation_ = generation;
        frame_.fill(0.0f);
        framePosition_ = 0;
        droppedFrames_ = 0;
        return resampler_.prepare(inputRate, outputRate);
    }

    // Returns true only when a complete frame was successfully queued.
    bool processSample(float input) noexcept {
        float converted = 0.0f;
        if (!resampler_.processSample(input, converted))
            return false;

        frame_[framePosition_++] = converted;
        if (framePosition_ != frameSize)
            return false;

        const bool queued = queue_.push(frame_.data(), generation_);
        if (!queued)
            ++droppedFrames_;
        framePosition_ = 0;
        return queued;
    }

    // Called on the audio thread at an AI enable/disable boundary. A partial
    // frame and resampler history are discarded so no frame straddles sessions.
    void setGeneration(std::uint64_t generation) noexcept {
        generation_ = generation;
        frame_.fill(0.0f);
        framePosition_ = 0;
        resampler_.reset();
    }

    std::uint64_t droppedFrames() const noexcept { return droppedFrames_; }
    std::size_t pendingSamples() const noexcept { return framePosition_; }

    // Call only while the producer is stopped. Does not reset the shared queue.
    void reset() noexcept {
        resampler_.reset();
        frame_.fill(0.0f);
        framePosition_ = 0;
        droppedFrames_ = 0;
    }

private:
    Queue& queue_;
    VadResampler resampler_;
    std::array<float, frameSize> frame_{};
    std::size_t framePosition_ = 0;
    std::uint64_t droppedFrames_ = 0;
    std::uint64_t generation_ = 0;
};

} // namespace smartgate
