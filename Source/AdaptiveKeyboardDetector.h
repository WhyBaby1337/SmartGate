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
        // A contact attack is often diluted by the voiced body of a 256-sample
        // frame. Give the high-band peak-to-average cue slightly more weight
        // and let it start contributing earlier; sustained sibilants still
        // need a sample-local impulse before the suppressor can attenuate them.
        const float highTransient = unit((highCrest - 3.0f) / 8.0f);
        const float mechanicalBase = 0.45f * unit((crest - 2.5f) / 7.0f)
                                   + 0.15f * unit((highRatio - 0.04f) / 0.45f)
                                   + 0.25f * unit((onset - 1.3f) / 4.0f);
        const float membraneBase = 0.40f * unit((crest - 1.8f) / 5.0f)
                                 + 0.10f * unit((highRatio - 0.015f) / 0.25f)
                                 + 0.30f * unit((onset - 1.2f) / 3.0f);
        const float mechanical = 0.65f * mechanicalBase + 0.35f * highTransient;
        const float membrane = 0.60f * membraneBase + 0.40f * highTransient;

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
        midLowPassCoefficient_ = static_cast<float>(
            std::exp(-2.0 * 3.14159265358979323846 * 1200.0 / rate));
        releaseCoefficient_ = static_cast<float>(
            std::exp(-1.0 / (0.0025 * rate)));
        envelopeCoefficient_ = static_cast<float>(
            std::exp(-1.0 / (0.010 * rate)));
        holdSamples_ = std::max(1, static_cast<int>(std::lround(0.0025 * rate)));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        midLowPassState_ = 0.0;
        previousMidBand_ = 0.0f;
        previousHighBand_ = 0.0f;
        previousHighMagnitude_ = 0.0f;
        previousOnsetConfidence_ = 0.0f;
        highBandEnvelope_ = 0.0f;
        midBandGain_ = 1.0f;
        highBandGain_ = 1.0f;
        midHoldCounter_ = 0;
        highHoldCounter_ = 0;
    }

    float process(float input, float reduction, float alignedFrameConfidence = 0.0f) noexcept {
        const float x = std::isfinite(input) ? input : 0.0f;
        const float amount = unit(reduction);
        const float frameContext = unit(alignedFrameConfidence);
        const double sample = static_cast<double>(x);
        lowPassState_ = sample + static_cast<double>(lowPassCoefficient_)
            * (lowPassState_ - sample);
        midLowPassState_ = sample + static_cast<double>(midLowPassCoefficient_)
            * (midLowPassState_ - sample);
        const float low = static_cast<float>(midLowPassState_);
        const float belowHigh = static_cast<float>(lowPassState_);
        const float mid = belowHigh - low;
        const float high = x - belowHigh;

        if (amount <= 0.0f) {
            previousMidBand_ = mid;
            previousHighBand_ = high;
            previousHighMagnitude_ = std::abs(high);
            previousOnsetConfidence_ = 0.0f;
            highBandEnvelope_ = std::abs(high);
            midBandGain_ = 1.0f;
            highBandGain_ = 1.0f;
            return x;
        }

        const float midMagnitude = std::abs(mid);
        const float midEdge = std::abs(mid - previousMidBand_);
        previousMidBand_ = mid;
        const float magnitude = std::abs(high);
        const float edge = std::abs(high - previousHighBand_);
        const float magnitudeRise = std::max(0.0f, magnitude - previousHighMagnitude_);
        const float magnitudeFallRatio = magnitude
            / std::max(previousHighMagnitude_, 1.0e-6f);
        previousHighBand_ = high;
        previousHighMagnitude_ = magnitude;

        // Only attenuate the 1.2-2.4 kHz band when aligned classifier context
        // and a sample-local impulsive event agree. This band can carry the
        // body of mechanical clicks, but also speech formants and consonants.
        // Requiring both cues keeps the new path inactive on ordinary speech.
        const float midContext = unit((frameContext - 0.30f) / 0.40f);
        const float midLevel = unit((midMagnitude - 0.004f) / 0.030f);
        const float midImpulsiveness = unit((midEdge - 0.015f) / 0.080f);
        const float midTransient = midLevel * midImpulsiveness * midContext;
        if (midTransient > 0.0f) {
            const float perceptualAmount = std::sqrt(amount);
            const float targetMidGain = 1.0f
                - 0.45f * perceptualAmount * midTransient;
            midBandGain_ = std::min(midBandGain_, targetMidGain);
            midHoldCounter_ = holdSamples_;
        } else if (midHoldCounter_ > 0) {
            --midHoldCounter_;
        } else {
            midBandGain_ = 1.0f - releaseCoefficient_
                * (1.0f - midBandGain_);
        }

        // Require both a meaningful high-band event and a sharp edge.
        // Sustained sibilants have high-band energy but do not resemble an
        // isolated impulse; low-frequency speech is outside this path.
        // The aligned frame score is context, not a broadband gain command:
        // sample-local magnitude and edge evidence must still identify the
        // exact transient. A confident keyboard frame modestly lowers the
        // local trigger thresholds so quieter clicks over speech are actionable.
        const float levelFloor = 0.012f - 0.004f * frameContext;
        const float levelSpan = 0.080f - 0.020f * frameContext;
        const float edgeFloor = 0.048f - 0.005f * frameContext;
        const float edgeSpan = 0.24f - 0.06f * frameContext;
        const float level = unit((magnitude - levelFloor) / levelSpan);
        const float impulsiveness = unit((edge - edgeFloor) / edgeSpan);

        // Confirm an isolated attack by its immediate peak-then-drop shape.
        // A noise-like sibilant has many sharp sample edges, but its peaks sit
        // on a sustained high-band bed; they should not repeatedly retrigger.
        const float onsetNovelty = unit(
            (magnitude - 5.0f * highBandEnvelope_) / 0.015f);
        highBandEnvelope_ = magnitude + envelopeCoefficient_
            * (highBandEnvelope_ - magnitude);
        const bool onsetCandidate = magnitudeRise > 0.012f
            && level > 0.3f && impulsiveness > 0.005f && onsetNovelty > 0.95f;
        const float onsetDrop = unit((0.55f - magnitudeFallRatio) / 0.25f);
        const float confidence = unit(previousOnsetConfidence_ * onsetDrop
                                      * (1.0f + 0.15f * frameContext));
        previousOnsetConfidence_ = onsetCandidate ? level * onsetNovelty : 0.0f;
        if (confidence > 0.0f) {
            // The UI amount is perceptual, not a linear amplitude multiplier.
            // A linear mapping made everyday values (0.15-0.30) barely audible
            // in effect. The square-root curve gives useful transient reduction
            // at modest settings while retaining a gentle zero end and a hard cap.
            const float perceptualAmount = std::sqrt(amount);
            const float targetGain = 1.0f - 0.92f * perceptualAmount * confidence;
            highBandGain_ = std::min(highBandGain_, targetGain);
            highHoldCounter_ = holdSamples_;
        } else if (highHoldCounter_ > 0) {
            --highHoldCounter_;
        } else {
            highBandGain_ = 1.0f - releaseCoefficient_
                * (1.0f - highBandGain_);
        }

        return low + mid * unit(midBandGain_) + high * unit(highBandGain_);
    }

    float getHighBandGain() const noexcept { return highBandGain_; }
    float getMidBandGain() const noexcept { return midBandGain_; }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    float lowPassCoefficient_ = 0.0f;
    float midLowPassCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    double midLowPassState_ = 0.0;
    float previousMidBand_ = 0.0f;
    float previousHighBand_ = 0.0f;
    float previousHighMagnitude_ = 0.0f;
    float previousOnsetConfidence_ = 0.0f;
    float highBandEnvelope_ = 0.0f;
    float envelopeCoefficient_ = 0.999f;
    float midBandGain_ = 1.0f;
    float highBandGain_ = 1.0f;
    int holdSamples_ = 1;
    int midHoldCounter_ = 0;
    int highHoldCounter_ = 0;
};

} // namespace smartgate
