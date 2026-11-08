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
        // Recover quickly after a detected transient.
        releaseCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.0015 * rate)));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        previousHighBand_ = 0.0f;
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
            previousHighBand_ = x - static_cast<float>(lowPassState_);
            highBandGain_ = 1.0f;
            return x;
        }

        const double sample = static_cast<double>(x);
        lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
            * (lowPassState_ - sample);
        const float low = static_cast<float>(lowPassState_);
        const float high = x - low;
        const float magnitude = std::abs(high);
        const float edge = std::abs(high - previousHighBand_);
        previousHighBand_ = high;

        // A mouth click is a sharp edge; a sustained /s/, /sh/ or breath has
        // continuous high-band energy but much smaller sample-to-sample jumps.
        // This avoids mistaking the onset of a sibilant for a click.
        const float level = unit((magnitude - 0.006f) / 0.035f);
        const float impulsiveness = unit((edge - 0.12f) / 0.28f);
        const float confidence = level * impulsiveness;
        if (confidence > 0.0f) {
            const float targetGain = 1.0f - 0.78f * amount * confidence;
            // Transient-only attack, confined to the high band.
            highBandGain_ = std::min(highBandGain_, targetGain);
        } else {
            highBandGain_ = 1.0f - releaseCoefficient_
                * (1.0f - highBandGain_);
        }

        return low + high * unit(highBandGain_);
    }

    float getHighBandGain() const noexcept { return highBandGain_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    float lowPassCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    float previousHighBand_ = 0.0f;
    float highBandGain_ = 1.0f;
};

} // namespace smartgate
