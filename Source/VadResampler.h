#pragma once

#include <cmath>

namespace smartgate {

// Streaming mono sample-rate converter for the VAD input path.
// Linear interpolation is deliberately lightweight; state is retained across
// host blocks, so arbitrary block boundaries do not change the output clock.
// Supports downsampling only (inputRate >= outputRate), which covers common
// audio-device rates feeding a 16 kHz VAD model.
class VadResampler {
public:
    bool prepare(double inputRate, double outputRate = 16000.0) noexcept {
        if (!std::isfinite(inputRate) || !std::isfinite(outputRate)
            || inputRate < outputRate || outputRate <= 0.0)
            return false;

        inputRate_ = inputRate;
        outputRate_ = outputRate;
        reset();
        return true;
    }

    void reset() noexcept {
        phase_ = 0.0;
        previous_ = 0.0f;
        hasPrevious_ = false;
    }

    // Returns true when one output sample is ready. No allocation or locking.
    bool processSample(float input, float& output) noexcept {
        const float current = std::isfinite(input) ? input : 0.0f;
        if (!hasPrevious_) {
            previous_ = current;
            hasPrevious_ = true;
            return false;
        }

        phase_ += outputRate_;
        bool produced = false;
        if (phase_ >= inputRate_) {
            phase_ -= inputRate_;
            const double fraction = 1.0 - phase_ / outputRate_;
            output = previous_ + (current - previous_)
                * static_cast<float>(fraction);
            produced = true;
        }

        previous_ = current;
        return produced;
    }

    double inputRate() const noexcept { return inputRate_; }
    double outputRate() const noexcept { return outputRate_; }

private:
    double inputRate_ = 0.0;
    double outputRate_ = 16000.0;
    double phase_ = 0.0;
    float previous_ = 0.0f;
    bool hasPrevious_ = false;
};

} // namespace smartgate
