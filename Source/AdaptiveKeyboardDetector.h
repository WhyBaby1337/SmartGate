#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace smartgate {

// DSP-only transient classifier. It estimates keyboard-like percussive content;
// it does not identify a speaker or make speech/non-speech decisions.
class AdaptiveKeyboardDetector {
public:
    enum class Mode { Universal, Mechanical, Membrane };

    void prepare(double sampleRate) noexcept {
        const double rate = std::isfinite(sampleRate) ? std::max(8000.0, sampleRate) : 44100.0;
        lowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 1800.0 / rate));
        reset();
    }

    void setMode(Mode mode) noexcept { mode_ = mode; }

    void reset() noexcept {
        lowPassState_ = 0.0f;
        referenceRms_ = 0.005f;
        hasReference_ = false;
        confidence_ = 0.0f;
    }

    // Call with consecutive mono frames. No allocation, locks, or system calls.
    float processFrame(const float* samples, std::size_t count) noexcept {
        if (samples == nullptr || count == 0)
            return confidence_;

        double energy = 0.0;
        double highEnergy = 0.0;
        float peak = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            const float x = std::isfinite(samples[i]) ? samples[i] : 0.0f;
            lowPassState_ = x + lowPassCoefficient_ * (lowPassState_ - x);
            const float high = x - lowPassState_;
            energy += static_cast<double>(x) * x;
            highEnergy += static_cast<double>(high) * high;
            peak = std::max(peak, std::abs(x));
        }

        const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(count)));
        const float highRatio = static_cast<float>(highEnergy / std::max(energy, 1.0e-12));
        const float crest = peak / std::max(rms, 1.0e-6f);
        const float onset = hasReference_
            ? rms / std::max(referenceRms_, 0.005f) : 1.0f;

        float mechanical = 0.40f * unit((crest - 2.5f) / 7.0f)
                         + 0.40f * unit((highRatio - 0.04f) / 0.45f)
                         + 0.20f * unit((onset - 1.3f) / 4.0f);
        float membrane = 0.35f * unit((crest - 1.8f) / 5.0f)
                        + 0.25f * unit((highRatio - 0.015f) / 0.25f)
                        + 0.40f * unit((onset - 1.2f) / 3.0f);

        float target = 0.0f;
        switch (mode_) {
            case Mode::Mechanical: target = mechanical; break;
            case Mode::Membrane: target = membrane; break;
            case Mode::Universal: target = 0.5f * (mechanical + membrane); break;
        }

        const float smoothing = target > confidence_ ? 0.45f : 0.08f;
        confidence_ += smoothing * (target - confidence_);
        if (hasReference_)
            referenceRms_ = 0.95f * referenceRms_ + 0.05f * rms;
        else {
            referenceRms_ = rms;
            hasReference_ = true;
        }
        return unit(confidence_);
    }

    float getConfidence() const noexcept { return confidence_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    Mode mode_ = Mode::Universal;
    float lowPassCoefficient_ = 0.0f;
    float lowPassState_ = 0.0f;
    float referenceRms_ = 0.005f;
    bool hasReference_ = false;
    float confidence_ = 0.0f;
};

} // namespace smartgate
