#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

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
        holdRemainingSamples_ = 0;
        gateOpen_ = false;
    }

    void setTimes(float attackMs, float releaseMs) noexcept {
        attackMs_ = std::isfinite(attackMs) ? std::clamp(attackMs, 0.1f, 500.0f) : 5.0f;
        releaseMs_ = std::isfinite(releaseMs) ? std::clamp(releaseMs, 1.0f, 2000.0f) : 120.0f;
        updateCoefficients();
    }

    void setHoldTime(float holdMs) noexcept {
        const float safeMs = std::isfinite(holdMs) ? std::clamp(holdMs, 0.0f, 1000.0f) : 80.0f;
        holdSamples_ = static_cast<std::uint64_t>(std::llround(
            static_cast<double>(safeMs) * 0.001 * sampleRate_));
        holdRemainingSamples_ = std::min(holdRemainingSamples_, holdSamples_);
    }

    void setThresholdDb(float thresholdDb) noexcept {
        const float safeDb = std::isfinite(thresholdDb)
            ? std::clamp(thresholdDb, -100.0f, 0.0f)
            : -36.0f;
        thresholdDb_ = safeDb;
        updateThresholds();
    }

    void setHysteresisDb(float hysteresisDb) noexcept {
        hysteresisDb_ = std::isfinite(hysteresisDb)
            ? std::clamp(hysteresisDb, 0.0f, 12.0f) : 3.0f;
        updateThresholds();
    }

    // Range is the maximum attenuation applied while the gate is closed.
    // 80 dB approximates a hard mute; smaller values retain room tone.
    void setRangeDb(float rangeDb) noexcept {
        const float safeDb = std::isfinite(rangeDb)
            ? std::clamp(rangeDb, 0.0f, 80.0f) : 80.0f;
        closedGain_ = std::pow(10.0f, -safeDb / 20.0f);
    }

    float process(float inputLevel) noexcept {
        const float safeLevel = std::isfinite(inputLevel) ? std::max(0.0f, inputLevel) : 0.0f;

        // Smooth the rectified detector signal so waveform zero-crossings do not
        // repeatedly toggle the gate around the threshold.
        const float detectorCoefficient = safeLevel > detectorLevel_
            ? detectorAttackCoefficient_
            : detectorReleaseCoefficient_;
        detectorLevel_ = safeLevel + detectorCoefficient * (detectorLevel_ - safeLevel);

        // Separate opening and closing thresholds provide hysteresis: once
        // open, small level fluctuations around the threshold cannot chatter.
        if (detectorLevel_ >= thresholdLinear_)
            gateOpen_ = true;
        else if (detectorLevel_ < closeThresholdLinear_)
            gateOpen_ = false;

        if (gateOpen_)
            holdRemainingSamples_ = holdSamples_;
        else if (holdRemainingSamples_ > 0)
            --holdRemainingSamples_;
        const float target = (gateOpen_ || holdRemainingSamples_ > 0)
            ? 1.0f : closedGain_;
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

    void updateThresholds() noexcept {
        thresholdLinear_ = std::pow(10.0f, thresholdDb_ / 20.0f);
        closeThresholdLinear_ = std::pow(10.0f, (thresholdDb_ - hysteresisDb_) / 20.0f);
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
    float thresholdDb_ = -36.0f;
    float hysteresisDb_ = 3.0f;
    float closedGain_ = 0.0001f;
    float thresholdLinear_ = 0.0158489f;
    float closeThresholdLinear_ = 0.0112202f;
    bool gateOpen_ = false;
    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float detectorAttackCoefficient_ = 0.0f;
    float detectorReleaseCoefficient_ = 0.0f;
    float detectorLevel_ = 0.0f;
    std::uint64_t holdSamples_ = 0;
    std::uint64_t holdRemainingSamples_ = 0;
    float gain_ = 1.0f;
};

} // namespace smartgate
