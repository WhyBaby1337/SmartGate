#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace smartgate {

// Replace invalid host samples with silence before they reach analysis or output.
// This helper is intentionally tiny and safe to call in the real-time callback.
inline float finiteSampleOrZero(float sample) noexcept {
    return std::isfinite(sample) ? sample : 0.0f;
}

// Compact, allocation-free frame description. Band values are energy ratios.
struct AudioFeatures {
    float logRms = -6.0f;
    float zeroCrossingRate = 0.0f;
    float lowBandRatio = 0.0f;
    float midBandRatio = 0.0f;
    float highBandRatio = 0.0f;

    bool isFinite() const noexcept {
        return std::isfinite(logRms)
            && std::isfinite(zeroCrossingRate)
            && std::isfinite(lowBandRatio)
            && std::isfinite(midBandRatio)
            && std::isfinite(highBandRatio);
    }
};

class FeatureExtractor {
public:
    void prepare(double sampleRate) noexcept {
        const double safeRate = std::isfinite(sampleRate)
            ? std::max(8000.0, sampleRate) : 44100.0;
        lowCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 250.0 / safeRate));
        midCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 3000.0 / safeRate));
        reset();
    }

    void reset() noexcept {
        lowState_ = 0.0f;
        midState_ = 0.0f;
    }

    // Call on consecutive, non-overlapping mono frames. No allocation or locks.
    AudioFeatures process(const float* samples, std::size_t count) noexcept {
        if (samples == nullptr || count == 0)
            return {};

        double totalEnergy = 0.0;
        double lowEnergy = 0.0;
        double midEnergy = 0.0;
        double highEnergy = 0.0;
        std::size_t crossings = 0;
        // Match the sanitized first sample used by the loop below. Keeping a raw NaN
        // here would create a false sign transition on the next finite sample.
        float previous = std::isfinite(samples[0]) ? samples[0] : 0.0f;

        for (std::size_t i = 0; i < count; ++i) {
            const float x = std::isfinite(samples[i]) ? samples[i] : 0.0f;
            totalEnergy += static_cast<double>(x) * x;

            // Keep filter state and band residuals in double precision. Finite
            // but out-of-range host samples can make float subtraction overflow
            // (for example, alternating +FLT_MAX/-FLT_MAX), permanently
            // poisoning the recursive filters with Inf/NaN.
            const double sample = static_cast<double>(x);
            lowState_ = sample + static_cast<double>(lowCoefficient_)
                * (lowState_ - sample);
            midState_ = sample + static_cast<double>(midCoefficient_)
                * (midState_ - sample);
            const double low = lowState_;
            const double mid = midState_ - lowState_;
            const double high = sample - midState_;

            lowEnergy += low * low;
            midEnergy += mid * mid;
            highEnergy += high * high;

            if (i > 0 && ((x >= 0.0f) != (previous >= 0.0f)))
                ++crossings;
            previous = x;
        }

        const double rms = std::sqrt(totalEnergy / static_cast<double>(count));
        const double bandEnergy = lowEnergy + midEnergy + highEnergy;
        const double denominator = std::max(bandEnergy, 1.0e-20);

        AudioFeatures result;
        result.logRms = static_cast<float>(std::log10(std::max(rms, 1.0e-6)));
        result.zeroCrossingRate = count > 1
            ? static_cast<float>(crossings) / static_cast<float>(count - 1)
            : 0.0f;
        result.lowBandRatio = static_cast<float>(lowEnergy / denominator);
        result.midBandRatio = static_cast<float>(midEnergy / denominator);
        result.highBandRatio = static_cast<float>(highEnergy / denominator);
        return result;
    }

private:
    float lowCoefficient_ = 0.0f;
    float midCoefficient_ = 0.0f;
    double lowState_ = 0.0;
    double midState_ = 0.0;
};

} // namespace smartgate
