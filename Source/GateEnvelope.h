#pragma once
#include <algorithm>
#include <cmath>

namespace smartgate {
class GateEnvelope {
public:
    void prepare(double sampleRate) noexcept { sampleRate_ = std::max(1.0, sampleRate); reset(); }
    void reset() noexcept { gain_ = 1.0f; }
    void setTimes(float attackMs, float releaseMs) noexcept {
        attackMs_ = std::max(0.1f, attackMs);
        releaseMs_ = std::max(1.0f, releaseMs);
    }
    void setThresholdDb(float thresholdDb) noexcept { thresholdLinear_ = std::pow(10.0f, thresholdDb / 20.0f); }
    float process(float inputLevel) noexcept {
        const float target = inputLevel >= thresholdLinear_ ? 1.0f : 0.0f;
        const float timeMs = target > gain_ ? attackMs_ : releaseMs_;
        const float coefficient = std::exp(-1.0f / (0.001f * timeMs * static_cast<float>(sampleRate_)));
        gain_ = target + coefficient * (gain_ - target);
        return gain_;
    }
    float getGain() const noexcept { return gain_; }
private:
    double sampleRate_ = 44100.0;
    float attackMs_ = 5.0f, releaseMs_ = 120.0f;
    float thresholdLinear_ = 0.01f, gain_ = 1.0f;
};
}
