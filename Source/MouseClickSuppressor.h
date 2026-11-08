#pragma once

#include <algorithm>
#include <cmath>

namespace smartgate {

// Dedicated mechanical mouse-switch transient suppressor. It is deliberately
// separate from MouthClickSuppressor: mouse clicks are shorter, sharper and
// often much quieter than mouth clicks, especially when mixed with speech.
// Only the high-frequency residual is attenuated; no broadband voice ducking.
class MouseClickSuppressor {
public:
    void prepare(double sampleRate) noexcept {
        const double rate = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        lowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 3200.0 / rate));
        releaseCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.0025 * rate)));
        envelopeCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.010 * rate)));
        holdSamples_ = std::max(1, static_cast<int>(std::lround(0.0030 * rate)));
        startupMuteSamples_ = std::max(1, static_cast<int>(std::lround(0.0020 * rate)));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        previousHighBand_ = 0.0f;
        highBandEnvelope_ = 0.0f;
        highBandGain_ = 1.0f;
        holdCounter_ = 0;
        startupSamplesRemaining_ = startupMuteSamples_;
        lastEdge_ = 0.0f;
        lastConfidence_ = 0.0f;
        lastLevel_ = 0.0f;
        lastImpulsiveness_ = 0.0f;
        lastNovelty_ = 0.0f;
    }

    float process(float input, float reduction, float alignedFrameContext = 0.0f) noexcept {
        const float x = std::isfinite(input) ? input : 0.0f;
        const float amount = unit(reduction);
        const float context = unit(alignedFrameContext);
        const double sample = static_cast<double>(x);
        lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
            * (lowPassState_ - sample);
        const float low = static_cast<float>(lowPassState_);
        const float high = x - low;

        if (amount <= 0.0f) {
            previousHighBand_ = high;
            highBandGain_ = 1.0f;
            return x;
        }

        const float magnitude = std::abs(high);
        const float edge = std::abs(high - previousHighBand_);
        previousHighBand_ = high;
        lastEdge_ = edge;

        // A mouse switch produces a brief peak above the recent high-band
        // bed, followed by a decaying ring. Noise-like speech has many sharp
        // sample edges too, but its magnitude envelope quickly becomes
        // stationary. Use a slow envelope as a novelty reference so random
        // sibilant peaks do not continuously retrigger attenuation.
        const float novelty = unit((magnitude - 2.0f * highBandEnvelope_) / 0.020f);
        highBandEnvelope_ = magnitude + envelopeCoefficient_
            * (highBandEnvelope_ - magnitude);

        // Context lowers the trigger threshold modestly, but cannot trigger
        // suppression by itself. A local, sample-accurate edge is mandatory.
        // This catches quieter switch clicks over speech without treating a
        // sustained /s/, /sh/ or /ch/ as a mouse event.
        const float levelFloor = 0.0020f - 0.0010f * context;
        const float levelSpan = 0.022f - 0.006f * context;
        const float edgeFloor = 0.035f - 0.009f * context;
        const float edgeSpan = 0.10f - 0.015f * context;
        const float level = unit((magnitude - levelFloor) / levelSpan);
        const float impulsiveness = unit((edge - edgeFloor) / edgeSpan);
        // The analysis filters need a brief settling period after reset.
        // Without it, the first peak of a sustained /s/ can look like a click
        // because the novelty reference has not yet learned the speech bed.
        const bool warmedUp = startupSamplesRemaining_ <= 0;
        if (startupSamplesRemaining_ > 0)
            --startupSamplesRemaining_;
        // Preserve unmistakable full-scale attacks even during startup; only
        // defer low-level decisions until the novelty reference has settled.
        const bool strongStartupTransient = magnitude >= 0.10f;
        const float confidence = (warmedUp || strongStartupTransient)
            ? level * impulsiveness * novelty : 0.0f;
        lastLevel_ = level;
        lastImpulsiveness_ = impulsiveness;
        lastNovelty_ = novelty;
        lastConfidence_ = confidence;
        if (confidence > 0.0f) {
            const float perceptualAmount = std::sqrt(amount);
            const float targetGain = 1.0f - 0.92f * perceptualAmount * confidence;
            highBandGain_ = std::min(highBandGain_, targetGain);
            holdCounter_ = holdSamples_;
        } else if (holdCounter_ > 0) {
            --holdCounter_;
        } else {
            highBandGain_ = 1.0f - releaseCoefficient_
                * (1.0f - highBandGain_);
        }

        return low + high * unit(highBandGain_);
    }

    float getHighBandGain() const noexcept { return highBandGain_; }
    float getLastEdge() const noexcept { return lastEdge_; }
    float getLastConfidence() const noexcept { return lastConfidence_; }
    float getLastLevel() const noexcept { return lastLevel_; }
    float getLastImpulsiveness() const noexcept { return lastImpulsiveness_; }
    float getLastNovelty() const noexcept { return lastNovelty_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    float lowPassCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float envelopeCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    float previousHighBand_ = 0.0f;
    float highBandEnvelope_ = 0.0f;
    float highBandGain_ = 1.0f;
    float lastEdge_ = 0.0f;
    float lastConfidence_ = 0.0f;
    float lastLevel_ = 0.0f;
    float lastImpulsiveness_ = 0.0f;
    float lastNovelty_ = 0.0f;
    int holdSamples_ = 1;
    int holdCounter_ = 0;
    int startupMuteSamples_ = 1;
    int startupSamplesRemaining_ = 1;
};

} // namespace smartgate
