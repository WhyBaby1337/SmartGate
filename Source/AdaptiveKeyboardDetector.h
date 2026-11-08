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
        const double rate = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        lowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 1800.0 / rate));
        reset();
    }

    void setMode(Mode mode) noexcept { mode_ = mode; }

    void reset() noexcept {
        lowPassState_ = 0.0;
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
        float highPeak = 0.0f;
        for (std::size_t i = 0; i < count; ++i) {
            const float x = std::isfinite(samples[i]) ? samples[i] : 0.0f;
            // Keep recursive filter state and the high-band residual in
            // double precision. Alternating extreme finite float samples can
            // overflow the float subtraction and permanently poison the filter.
            const double sample = static_cast<double>(x);
            lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
                * (lowPassState_ - sample);
            const double high = sample - lowPassState_;
            energy += sample * sample;
            highEnergy += high * high;
            peak = std::max(peak, std::abs(x));
            highPeak = std::max(highPeak, static_cast<float>(std::abs(high)));
        }

        const float rms = static_cast<float>(std::sqrt(energy / static_cast<double>(count)));
        const float highRatio = static_cast<float>(highEnergy / std::max(energy, 1.0e-12));
        const float crest = peak / std::max(rms, 1.0e-6f);
        const float highRms = static_cast<float>(
            std::sqrt(highEnergy / static_cast<double>(count)));
        const float highCrest = highPeak / std::max(highRms, 1.0e-6f);
        const float onset = hasReference_
            ? rms / std::max(referenceRms_, 0.005f) : 1.0f;

        // Broadband energy alone is not enough: sustained hiss/noise must
        // score lower than a short, high-crest transient. Keep onset and
        // crest factor dominant, with spectral brightness as supporting data.
        // A click mixed into speech may barely change the full-band crest or
        // RMS. Measure the peak-to-average ratio of the high-pass residual as
        // a second, independent transient cue. Sustained high-frequency speech
        // (for example a long sibilant) has energy but a much lower crest than
        // a short mouse/key switch impulse.
        const float highTransient = unit((highCrest - 3.5f) / 10.0f);
        const float mechanicalBase = 0.45f * unit((crest - 2.5f) / 7.0f)
                                   + 0.15f * unit((highRatio - 0.04f) / 0.45f)
                                   + 0.25f * unit((onset - 1.3f) / 4.0f);
        const float membraneBase = 0.40f * unit((crest - 1.8f) / 5.0f)
                                 + 0.10f * unit((highRatio - 0.015f) / 0.25f)
                                 + 0.30f * unit((onset - 1.2f) / 3.0f);
        const float mechanical = 0.75f * mechanicalBase + 0.25f * highTransient;
        const float membrane = 0.70f * membraneBase + 0.30f * highTransient;

        float target = 0.0f;
        switch (mode_) {
            case Mode::Mechanical: target = mechanical; break;
            case Mode::Membrane: target = membrane; break;
            case Mode::Universal: target = 0.5f * (mechanical + membrane); break;
        }

        // In near-silence there is no speech transient to protect, so clear stale
        // click confidence faster. Keep the conservative release during audible
        // input to avoid pumping or clipping the tail of speech.
        const float releaseSmoothing = rms < 0.002f ? 0.18f : 0.08f;
        const float smoothing = target > confidence_ ? 0.45f : releaseSmoothing;
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

    // Confidence is an estimate, not a probability that should map linearly
    // to attenuation. Squaring it makes uncertain detections less destructive
    // to speech while preserving full reduction for a fully confident click.
    static float gainForReduction(float reduction, float confidence) noexcept {
        const float amount = unit(reduction);
        const float certainty = unit(confidence);
        return 1.0f - amount * certainty * certainty;
    }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    Mode mode_ = Mode::Universal;
    float lowPassCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    float referenceRms_ = 0.005f;
    bool hasReference_ = false;
    float confidence_ = 0.0f;
};

 
// Sample-rate-aware output smoothing for keyboard attenuation. The classifier
// updates once per analysis frame, while this gain is evaluated per audio
// sample so frame-boundary changes cannot create audible steps.
class AdaptiveKeyboardGain {
public:
    void prepare(double sampleRate) noexcept {
        const double rate = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        attackCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.003 * rate)));
        releaseCoefficient_ = static_cast<float>(std::exp(-1.0 / (0.030 * rate)));
        reset();
    }

    void reset() noexcept { gain_ = 1.0f; }

    float process(float reduction, float confidence) noexcept {
        const float target = AdaptiveKeyboardDetector::gainForReduction(
            reduction, confidence);
        const float coefficient = target < gain_
            ? attackCoefficient_ : releaseCoefficient_;
        gain_ = target + coefficient * (gain_ - target);
        return unit(gain_);
    }

    float getGain() const noexcept { return gain_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 1.0f, 0.0f, 1.0f);
    }

    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float gain_ = 1.0f;
};

 
// Sample-accurate keyboard/mouse transient attenuation. Unlike a broadband
// output duck, it only reduces the high-frequency residual around sharp
// impulses, so speech fundamentals and most formant energy remain untouched.
class KeyboardTransientSuppressor {
public:
    void prepare(double sampleRate) noexcept {
        const double rate = std::isfinite(sampleRate)
            ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
        lowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 2400.0 / rate));
        releaseCoefficient_ = static_cast<float>(
            std::exp(-1.0 / (0.0025 * rate)));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        previousHighBand_ = 0.0f;
        highBandGain_ = 1.0f;
    }

    float process(float input, float reduction,
                  float classifierConfidence = 1.0f) noexcept {
        const float x = std::isfinite(input) ? input : 0.0f;
        const float amount = unit(reduction);
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

        // Require both a meaningful high-band event and a sharp edge.
        // Sustained sibilants have high-band energy but do not resemble an
        // isolated impulse; low-frequency speech is outside this path.
        const float level = unit((magnitude - 0.012f) / 0.080f);
        const float impulsiveness = unit((edge - 0.075f) / 0.24f);
        const float localTransient = level * impulsiveness;
        // The frame classifier sees the undelayed input, while this suppressor
        // processes audio after the 30 ms lookahead. Its confidence therefore
        // describes the same transient before it reaches this sample. Keep a
        // 25% local-detector floor so unusual clicks are not completely missed,
        // but let keyboard-profile evidence strengthen genuine mechanical hits.
        const float classifierEvidence = unit((unit(classifierConfidence) - 0.08f) / 0.32f);
        const float classifierWeight = 0.25f + 0.75f * classifierEvidence;
        const float confidence = localTransient * classifierWeight;
        if (confidence > 0.0f) {
            // The UI amount is perceptual, not a linear amplitude multiplier.
            // A linear mapping made everyday values (0.15-0.30) barely audible
            // in effect. The square-root curve gives useful transient reduction
            // at modest settings while retaining a gentle zero end and a hard cap.
            const float perceptualAmount = std::sqrt(amount);
            const float targetGain = 1.0f - 0.92f * perceptualAmount * confidence;
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
