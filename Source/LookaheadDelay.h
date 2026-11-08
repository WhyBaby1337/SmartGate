#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <vector>

namespace smartgate {

// Fixed-latency stereo delay used as a lookahead window. All memory is
// allocated in prepare(); process() is bounded, allocation-free and lock-free.
class LookaheadDelay {
public:
    bool prepare(double sampleRate, double milliseconds = 30.0) {
        const double safeRate = std::isfinite(sampleRate) && sampleRate > 0.0
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        const double safeMs = std::isfinite(milliseconds)
            ? std::clamp(milliseconds, 0.0, 30.0) : 30.0;
        delaySamples_ = static_cast<std::size_t>(std::llround(safeRate * safeMs * 0.001));
        ring_.assign(delaySamples_ + 1, {0.0f, 0.0f});
        writePosition_ = 0;
        return !ring_.empty();
    }

    void reset() noexcept {
        std::fill(ring_.begin(), ring_.end(), std::array<float, 2>{0.0f, 0.0f});
        writePosition_ = 0;
    }

    void process(float left, float right, float& delayedLeft, float& delayedRight) noexcept {
        if (delaySamples_ == 0 || ring_.empty()) {
            delayedLeft = left;
            delayedRight = right;
            return;
        }
        ring_[writePosition_] = {left, right};
        auto readPosition = writePosition_ + 1;
        if (readPosition == ring_.size())
            readPosition = 0;
        delayedLeft = ring_[readPosition][0];
        delayedRight = ring_[readPosition][1];
        ++writePosition_;
        if (writePosition_ == ring_.size())
            writePosition_ = 0;
    }

    std::size_t latencySamples() const noexcept { return delaySamples_; }

private:
    std::vector<std::array<float, 2>> ring_;
    std::size_t delaySamples_ = 0;
    std::size_t writePosition_ = 0;
};

} // namespace smartgate
