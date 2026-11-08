#pragma once

#include <algorithm>
#include <cmath>

namespace smartgate {

// A lightweight, allocation-free programme leveler. It follows signal energy,
// applies bounded make-up/attenuation gain, and smooths gain changes to avoid
// audible pumping. It is deliberately independent of VAD/AI inference.
class VoiceLeveler {
public:
    void prepare(double sampleRate) noexcept {
        sampleRate_ = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        envelope_ = 0.0f;
        gain_ = 1.0f;
        updateCoefficients();
    }

    void reset() noexcept {
        envelope_ = 0.0f;
        gain_ = 1.0f;
    }

    float process(float input, float amount) noexcept {
        const float gain = processStereo(input, input, amount);
        return input * gain;
    }

    // One linked detector/gain for both channels keeps the stereo image stable.
    float processStereo(float left, float right, float amount) noexcept {
        if (!std::isfinite(left)) left = 0.0f;
        if (!std::isfinite(right)) right = 0.0f;
        amount = std::clamp(std::isfinite(amount) ? amount : 0.0f, 0.0f, 1.0f);
        if (amount <= 0.0f) {
            reset();
            return 1.0f;
        }

        const float squared = 0.5f * (left * left + right * right);
        const float coefficient = squared > envelope_ ? envelopeAttack_ : envelopeRelease_;
        envelope_ = squared + coefficient * (envelope_ - squared);

        // Do not add gain to silence or the gate's closed-state residue.
        constexpr float activityFloorSquared = 1.0e-5f; // -50 dBFS
        constexpr float target = 0.12589254f; // -18 dBFS
        float desired = 1.0f;
        if (envelope_ > activityFloorSquared) {
            const float measured = std::sqrt(envelope_);
            desired = std::clamp(target / measured, 0.25118864f, 3.9810717f);
        }
        const float shapedAmount = amount * amount;
        const float blended = 1.0f + (desired - 1.0f) * shapedAmount;

        const float gainCoefficient = blended < gain_ ? gainAttack_ : gainRelease_;
        gain_ = blended + gainCoefficient * (gain_ - blended);
        return gain_;
    }

    float gainForDiagnostics() const noexcept { return gain_; }

private:
    void updateCoefficients() noexcept {
        const auto coefficient = [this](double milliseconds) {
            return static_cast<float>(std::exp(-1.0 / (0.001 * milliseconds * sampleRate_)));
        };
        envelopeAttack_ = coefficient(35.0);
        envelopeRelease_ = coefficient(280.0);
        gainAttack_ = coefficient(18.0);
        gainRelease_ = coefficient(220.0);
    }

    double sampleRate_ = 44100.0;
    float envelope_ = 0.0f;
    float gain_ = 1.0f;
    float envelopeAttack_ = 0.0f;
    float envelopeRelease_ = 0.0f;
    float gainAttack_ = 0.0f;
    float gainRelease_ = 0.0f;
};

} // namespace smartgate
