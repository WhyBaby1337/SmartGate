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

    void reset() noexcept {
        detectorLevel_ = 0.0f;
        gain_ = 1.0f;
    }

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

        // Smooth the rectified detector signal so waveform zero-crossings do not
        // repeatedly toggle the gate around the threshold.
        const float detectorCoefficient = safeLevel > detectorLevel_
            ? detectorAttackCoefficient_
            : detectorReleaseCoefficient_;
        detectorLevel_ = safeLevel + detectorCoefficient * (detectorLevel_ - safeLevel);

        const float target = detectorLevel_ >= thresholdLinear_ ? 1.0f : 0.0f;
        const float gainCoefficient = target > gain_ ? attackCoefficient_ : releaseCoefficient_;
        gain_ = target + gainCoefficient * (gain_ - target);
        gain_ = std::clamp(gain_, 0.0f, 1.0f);
        return gain_;
    }

    float getGain() const noexcept { return gain_; }
    float getDetectorLevel() const noexcept { return detectorLevel_; }

private:
    static float coefficientFor(float timeMs, float sampleRate) noexcept {
        return std::exp(-1.0f / (0.001f * timeMs * sampleRate));
    }

    void updateCoefficients() noexcept {
        const auto rate = static_cast<float>(sampleRate_);
        attackCoefficient_ = coefficientFor(attackMs_, rate);
        releaseCoefficient_ = coefficientFor(releaseMs_, rate);
        detectorAttackCoefficient_ = coefficientFor(0.1f, rate);
        detectorReleaseCoefficient_ = coefficientFor(20.0f, rate);
    }

    double sampleRate_ = 44100.0;
    float attackMs_ = 5.0f;
    float releaseMs_ = 120.0f;
    float thresholdLinear_ = 0.0158489f;
    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float detectorAttackCoefficient_ = 0.0f;
    float detectorReleaseCoefficient_ = 0.0f;
    float detectorLevel_ = 0.0f;
    float gain_ = 1.0f;
};

} // namespace smartgate
