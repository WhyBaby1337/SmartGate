#pragma once

#include <algorithm>
#include <cmath>

namespace smartgate {

class GateEnvelope {
public:
    void prepare(double sampleRate) noexcept {
        sampleRate_ = std::isfinite(sampleRate) ? std::max(1.0, sampleRate) : 44100.0;
        updateCoefficients();
        reset();
    }

    void reset() noexcept { gain_ = 1.0f; }

    void setTimes(float attackMs, float releaseMs) noexcept {
        attackMs_ = std::isfinite(attackMs) ? std::clamp(attackMs, 0.1f, 500.0f) : 5.0f;
        releaseMs_ = std::isfinite(releaseMs) ? std::clamp(releaseMs, 1.0f, 2000.0f) : 120.0f;
        updateCoefficients();
    }

    void setThresholdDb(float thresholdDb) noexcept {
        const float safeDb = std::isfinite(thresholdDb)
            ? std::clamp(thresholdDb, -100.0f, 0.0f)
            : -36.0f;
        thresholdLinear_ = std::pow(10.0f, safeDb / 20.0f);
    }

    float process(float inputLevel) noexcept {
        const float safeLevel = std::isfinite(inputLevel) ? std::max(0.0f, inputLevel) : 0.0f;
        const float target = safeLevel >= thresholdLinear_ ? 1.0f : 0.0f;
        const float coefficient = target > gain_ ? attackCoefficient_ : releaseCoefficient_;
        gain_ = target + coefficient * (gain_ - target);
        gain_ = std::clamp(gain_, 0.0f, 1.0f);
        return gain_;
    }

    float getGain() const noexcept { return gain_; }

private:
    void updateCoefficients() noexcept {
        const auto rate = static_cast<float>(sampleRate_);
        attackCoefficient_ = std::exp(-1.0f / (0.001f * attackMs_ * rate));
        releaseCoefficient_ = std::exp(-1.0f / (0.001f * releaseMs_ * rate));
    }

    double sampleRate_ = 44100.0;
    float attackMs_ = 5.0f;
    float releaseMs_ = 120.0f;
    float thresholdLinear_ = 0.0158489f;
    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float gain_ = 1.0f;
};

} // namespace smartgate
