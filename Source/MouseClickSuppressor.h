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
        // Keep the short ring-confirmation window responsive to quiet
        // switch clicks while preserving its elapsed time across host rates.
        decayWindowSamples_ = std::max(1, static_cast<int>(
            std::lround(rate * (24.0 / 48000.0))));
        reset();
    }

    void reset() noexcept {
        lowPassState_ = 0.0;
        previousHighBand_ = 0.0f;
        previousHighMagnitude_ = 0.0f;
        previousOnsetConfidence_ = 0.0f;
        pendingOnsetConfidence_ = 0.0f;
        pendingOnsetFirstPeak_ = 0.0f;
        pendingOnsetSecondPeak_ = 0.0f;
        pendingOnsetEarlyFirst_ = 0.0f;
        pendingOnsetEarlySecond_ = 0.0f;
        pendingOnsetSamples_ = 0;
        highBandEnvelope_ = 0.0f;
        highBandGain_ = 1.0f;
        holdCounter_ = 0;
        lastEdge_ = 0.0f;
        lastConfidence_ = 0.0f;
        lastLevel_ = 0.0f;
        lastImpulsiveness_ = 0.0f;
        lastNovelty_ = 0.0f;
        lastOnsetCandidate_ = false;
        onsetCandidateCount_ = 0;
        lastDecayRatio_ = 0.0f;
        historyWriteIndex_ = 0;
        historyCount_ = 0;
        for (float& sample : highBandHistory_)
            sample = 0.0f;
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
        highBandHistory_[historyWriteIndex_] = high;
        historyWriteIndex_ = (historyWriteIndex_ + 1) % 32;
        historyCount_ = std::min(32, historyCount_ + 1);

        if (amount <= 0.0f) {
            previousHighBand_ = high;
            previousHighMagnitude_ = std::abs(high);
            previousOnsetConfidence_ = 0.0f;
            pendingOnsetConfidence_ = 0.0f;
            pendingOnsetFirstPeak_ = 0.0f;
            pendingOnsetSecondPeak_ = 0.0f;
            pendingOnsetEarlyFirst_ = 0.0f;
            pendingOnsetEarlySecond_ = 0.0f;
            pendingOnsetSamples_ = 0;
            highBandEnvelope_ = std::abs(high);
            highBandGain_ = 1.0f;
            return x;
        }

        const float magnitude = std::abs(high);
        const float edge = std::abs(high - previousHighBand_);
        const float magnitudeRise = std::max(0.0f, magnitude - previousHighMagnitude_);
        const float magnitudeFallRatio = magnitude / std::max(previousHighMagnitude_, 1.0e-6f);
        previousHighBand_ = high;
        previousHighMagnitude_ = magnitude;
        lastEdge_ = edge;

        // A mouse switch produces a brief peak above the recent high-band
        // bed, followed by a decaying ring. Noise-like speech has many sharp
        // sample edges too, but its magnitude envelope quickly becomes
        // stationary. Use a slow envelope as a novelty reference so random
        // sibilant peaks do not continuously retrigger attenuation.
        const float novelty = unit((magnitude - 2.0f * highBandEnvelope_) / 0.020f);
        // A true switch strike can begin at a resonant peak and collapse on
        // the very next sample. Require that isolated peak-and-drop shape;
        // a sustained sibilant rises into its first peak instead of collapsing.
        // The stronger pre-event bed test also rejects random peaks in steady
        // noise, without changing the existing general transient detector.
        const float onsetNovelty = unit((magnitude - 5.0f * highBandEnvelope_) / 0.015f);
        highBandEnvelope_ = magnitude + envelopeCoefficient_
            * (highBandEnvelope_ - magnitude);

        // Context lowers the trigger threshold modestly, but cannot trigger
        // suppression by itself. A local, sample-accurate edge is mandatory.
        // This catches quieter switch clicks over speech without treating a
        // sustained /s/, /sh/ or /ch/ as a mouse event.
        const float levelFloor = 0.0020f - 0.0010f * context;
        const float levelSpan = 0.022f - 0.006f * context;
        // Quieter switch rings can have a smaller sample-to-sample edge
        // than the established profiles. Keep the peak-then-drop confirmation
        // as the primary speech guard while allowing this weaker edge through.
        const float edgeFloor = 0.020f - 0.006f * context;
        const float edgeSpan = 0.13f - 0.020f * context;
        const float level = unit((magnitude - levelFloor) / levelSpan);
        const float impulsiveness = unit((edge - edgeFloor) / edgeSpan);

        // Qualify a switch strike by its short decay envelope, not by a
        // one-sample drop. Two short, sample-rate-scaled RMS windows confirm
        // rings over a consistent elapsed time. The stricter decay threshold
        // rejects phase/envelope fluctuations in sustained sibilants.
        const bool onsetCandidate = edge > edgeFloor && level > 0.3f
            && impulsiveness > 0.005f && novelty > 0.90f
            && onsetNovelty > 0.25f;
        lastOnsetCandidate_ = onsetCandidate;
        if (onsetCandidate)
            ++onsetCandidateCount_;
        if (onsetCandidate && pendingOnsetSamples_ == 0) {
            pendingOnsetConfidence_ = level * novelty;
            pendingOnsetFirstPeak_ = magnitude;
            pendingOnsetSecondPeak_ = 0.0f;
            pendingOnsetEarlyFirst_ = magnitude;
            pendingOnsetEarlySecond_ = 0.0f;
            pendingOnsetSamples_ = 1;
        } else if (pendingOnsetSamples_ > 0) {
            if (pendingOnsetSamples_ <= 18)
                pendingOnsetEarlyFirst_ += magnitude;
            else if (pendingOnsetSamples_ <= 36)
                pendingOnsetEarlySecond_ += magnitude;
            if (pendingOnsetSamples_ <= decayWindowSamples_) {
                pendingOnsetFirstPeak_ += magnitude;
            } else {
                pendingOnsetSecondPeak_ += magnitude;
            }
            ++pendingOnsetSamples_;
        }

        float onsetConfidence = 0.0f;
        const float periodicity = getRecentPeriodicity();
        // Preserve the immediate path for a genuinely broadband, isolated
        // switch impulse. Its high-pass residual collapses sharply on the
        // next sample, unlike the slower resonant ring handled by the window.
        if (pendingOnsetSamples_ > 1 && edge > 0.080f
            && magnitudeFallRatio < 0.60f) {
            onsetConfidence = pendingOnsetConfidence_
                * unit((0.80f - magnitudeFallRatio) / 0.25f);
            pendingOnsetSamples_ = 0;
            pendingOnsetConfidence_ = 0.0f;
            pendingOnsetFirstPeak_ = 0.0f;
            pendingOnsetSecondPeak_ = 0.0f;
            pendingOnsetEarlyFirst_ = 0.0f;
            pendingOnsetEarlySecond_ = 0.0f;
        }
        if (pendingOnsetSamples_ > 36) {
            const float earlyDecay = pendingOnsetEarlySecond_
                / std::max(pendingOnsetEarlyFirst_, 1.0e-12f);
            if (earlyDecay < 0.84f
                && periodicity > 0.75f && periodicity < 0.96f) {
                // Very short, strongly decaying resonances need an earlier
                // decision than the longer phase-stable ring confirmation.
                // Also require some short-window periodic structure: a random
                // fluctuation in a rising noise-like fricative can mimic a
                // modest two-window decay, but it does not form a resonant ring.
                onsetConfidence = pendingOnsetConfidence_
                    * unit((0.84f - earlyDecay) / 0.08f);
                pendingOnsetSamples_ = 0;
                pendingOnsetConfidence_ = 0.0f;
                pendingOnsetFirstPeak_ = 0.0f;
                pendingOnsetSecondPeak_ = 0.0f;
                pendingOnsetEarlyFirst_ = 0.0f;
                pendingOnsetEarlySecond_ = 0.0f;
            }
        }
        if (pendingOnsetSamples_ > 2 * decayWindowSamples_) {
            const float decay = pendingOnsetSecondPeak_
                / std::max(pendingOnsetFirstPeak_, 1.0e-12f);
            lastDecayRatio_ = decay;
            // Require a clear drop between adjacent windows. Small phase/envelope
            // fluctuations in sustained sibilants must not count as switch-ring decay.
            // Average absolute magnitude is less sensitive than short-window
            // RMS to the phase of sustained tones at non-48 kHz sample rates.
            // A decaying switch ring still loses mean magnitude between windows.
            const float decayConfidence = periodicity < 0.96f
                ? unit((0.92f - decay) / 0.04f) : 0.0f;
            onsetConfidence = pendingOnsetConfidence_ * decayConfidence;
            pendingOnsetSamples_ = 0;
            pendingOnsetConfidence_ = 0.0f;
            pendingOnsetFirstPeak_ = 0.0f;
            pendingOnsetSecondPeak_ = 0.0f;
            pendingOnsetEarlyFirst_ = 0.0f;
            pendingOnsetEarlySecond_ = 0.0f;
        }

        // The broad edge/novelty score remains diagnostic only. A candidate
        // must show measurable decay across the short observation
        // window before it can attenuate the mouse high band.
        const float confidence = onsetConfidence;
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
    bool getLastOnsetCandidate() const noexcept { return lastOnsetCandidate_; }
    unsigned getOnsetCandidateCount() const noexcept { return onsetCandidateCount_; }
    int getPendingOnsetSamples() const noexcept { return pendingOnsetSamples_; }
    float getLastDecayRatio() const noexcept { return lastDecayRatio_; }

private:
    float getRecentPeriodicity() const noexcept {
        if (historyCount_ < 32)
            return 0.0f;
        float maximumCorrelation = 0.0f;
        for (int lag = 4; lag <= 20; ++lag) {
            float cross = 0.0f;
            float firstEnergy = 0.0f;
            float secondEnergy = 0.0f;
            for (int i = 0; i < 32 - lag; ++i) {
                const float first = highBandHistory_[(historyWriteIndex_ + i) % 32];
                const float second = highBandHistory_[(historyWriteIndex_ + i + lag) % 32];
                cross += first * second;
                firstEnergy += first * first;
                secondEnergy += second * second;
            }
            const float denominator = std::sqrt(
                std::max(firstEnergy * secondEnergy, 1.0e-20f));
            maximumCorrelation = std::max(maximumCorrelation,
                cross / denominator);
        }
        return unit(maximumCorrelation);
    }

    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    float lowPassCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float envelopeCoefficient_ = 0.0f;
    double lowPassState_ = 0.0;
    float previousHighBand_ = 0.0f;
    float previousHighMagnitude_ = 0.0f;
    float previousOnsetConfidence_ = 0.0f;
    float pendingOnsetConfidence_ = 0.0f;
    float pendingOnsetFirstPeak_ = 0.0f;
    float pendingOnsetSecondPeak_ = 0.0f;
    float pendingOnsetEarlyFirst_ = 0.0f;
    float pendingOnsetEarlySecond_ = 0.0f;
    int pendingOnsetSamples_ = 0;
    float highBandEnvelope_ = 0.0f;
    float highBandGain_ = 1.0f;
    float lastEdge_ = 0.0f;
    float lastConfidence_ = 0.0f;
    float lastLevel_ = 0.0f;
    float lastImpulsiveness_ = 0.0f;
    float lastNovelty_ = 0.0f;
    bool lastOnsetCandidate_ = false;
    unsigned onsetCandidateCount_ = 0;
    float lastDecayRatio_ = 0.0f;
    int holdSamples_ = 1;
    int decayWindowSamples_ = 16;
    float highBandHistory_[32]{};
    int historyWriteIndex_ = 0;
    int historyCount_ = 0;
    int holdCounter_ = 0;
};

} // namespace smartgate
