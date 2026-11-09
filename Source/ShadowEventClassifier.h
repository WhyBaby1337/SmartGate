#pragma once

#include "AudioFeatures.h"

#include <algorithm>
#include <cmath>

namespace smartgate {

// Shadow-only event fusion. It combines existing speech/keyboard estimates
// with frame-to-frame acoustic novelty to flag possible mouse-like events.
// It never changes audio or controls a suppressor.
struct EventScores {
    float speech = 0.0f;
    float keyboard = 0.0f;
    float mouse = 0.0f;
    float unknown = 1.0f;

    bool isFinite() const noexcept {
        return std::isfinite(speech) && std::isfinite(keyboard)
            && std::isfinite(mouse) && std::isfinite(unknown);
    }
};

class ShadowEventClassifier {
public:
    void reset() noexcept {
        previous_ = {};
        hasPrevious_ = false;
    }

    EventScores process(const AudioFeatures& features,
                        float speechConfidence,
                        float keyboardConfidence) noexcept {
        if (!features.isFinite())
            return {};

        const float speech = unit(speechConfidence);
        const float keyboard = unit(keyboardConfidence);

        float novelty = 0.0f;
        if (hasPrevious_) {
            const float levelChange = unit(
                std::abs(features.logRms - previous_.logRms) / 0.65f);
            const float highChange = unit(
                std::abs(features.highBandRatio - previous_.highBandRatio) / 0.35f);
            const float crossingChange = unit(
                std::abs(features.zeroCrossingRate - previous_.zeroCrossingRate) / 0.18f);
            novelty = unit(0.30f * levelChange + 0.45f * highChange
                           + 0.25f * crossingChange);
        }

        // A mouse candidate needs both a bright, noisy spectrum and a new
        // frame-level change. This is deliberately a suspicion score, not a
        // claim that every such event is a mouse click.
        const float brightness = unit((features.highBandRatio - 0.18f) / 0.62f);
        const float crossings = unit((features.zeroCrossingRate - 0.035f) / 0.30f);
        const float mouse = unit(brightness * crossings * novelty
                                 * (1.0f - 0.55f * keyboard));

        previous_ = features;
        hasPrevious_ = true;

        EventScores result;
        result.speech = speech;
        result.keyboard = keyboard;
        result.mouse = mouse;
        result.unknown = unit(1.0f - std::max({speech, keyboard, mouse}));
        return result;
    }

private:
    static float unit(float value) noexcept {
        return std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f);
    }

    AudioFeatures previous_{};
    bool hasPrevious_ = false;
};

} // namespace smartgate
