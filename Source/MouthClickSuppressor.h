#pragma once

#include <algorithm>
#include <cmath>

namespace smartgate {

// Conservative, zero-latency high-band transient attenuator for short mouth
// clicks. It never turns down the broadband voice signal: only the high-pass
// residual is attenuated, and only when its instantaneous crest is far above
// a slow local reference. Sustained sibilants/noise should therefore pass.
class MouthClickSuppressor {
public:
    void prepare(double sampleRate) noexcept {
        const double rate = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        lowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 2600.0 / rate));
        referenceCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.025 * rate)));
        attackCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.00035 * rate)));
        releaseCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.018 * rate)));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        highBandReference_ = 0.001f;
        highBandGain_ = 1.0f;
    }

    float process(float input, float reduction) noexcept {
        const float x = std::isfinite(input) ? input : 0.0f;
        const float amount = unit(reduction);
        if (amount <= 0.0f) {
            // Keep the filter warm while bypassed, but remain bit-transparent.
            const double sample = static_cast<double>(x);
            lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
                * (lowPassState_ - sample);
            return x;
        }

        const double sample = static_cast<double>(x);
        lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
            * (lowPassState_ - sample);
        const float low = static_cast<float>(lowPassState_);
        const float high = x - low;
        const float magnitude = std::abs(high);
        const float reference = std::max(highBandReference_, 0.001f);
        const float crest = magnitude / reference;

        // Require both a genuinely audible high-band event and a sharp
        // amplitude crest. A sustained /s/ or breath-like noise has a much
        // lower crest against its local reference than an isolated click.
        const float level = unit((magnitude - 0.006f) / 0.035f);
        const float impulsiveness = unit((crest - 7.0f) / 13.0f);
        const float confidence = level * impulsiveness;
        const float targetGain = 1.0f - 0.78f * amount * confidence;
        const float coefficient = targetGain < highBandGain_
            ? attackCoefficient_ : releaseCoefficient_;
        highBandGain_ = targetGain + coefficient * (highBandGain_ - targetGain);

        // Slow reference tracking prevents the reference from following and
        // hiding a short impulse, while adapting to changes in microphone level.
        highBandReference_ = magnitude + referenceCoefficient_
            * (highBandReference_ - magnitude);
        return low + high * unit(highBandGain_);
    }

    float getHighBandGain() const noexcept { return highBandGain_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    float lowPassCoefficient_ = 0.0f;
    float referenceCoefficient_ = 0.0f;
    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    float highBandReference_ = 0.001f;
    float highBandGain_ = 1.0f;
};

} // namespace smartgate
