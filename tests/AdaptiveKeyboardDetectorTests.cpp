#include "AdaptiveKeyboardDetector.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void requireKeyboard(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runAdaptiveKeyboardDetectorTests() {
    constexpr std::size_t frameSize = 256;
    smartgate::AdaptiveKeyboardDetector detector;
    detector.prepare(48000.0);

    smartgate::AdaptiveKeyboardDetector maximumRateDetector;
    smartgate::AdaptiveKeyboardDetector extremeRateDetector;
    maximumRateDetector.prepare(768000.0);
    extremeRateDetector.prepare(std::numeric_limits<double>::max());
    std::vector<float> rateProbe(frameSize, 0.0f);
    rateProbe[100] = 0.8f;
    const float maximumRateConfidence = maximumRateDetector.processFrame(
        rateProbe.data(), rateProbe.size());
    const float extremeRateConfidence = extremeRateDetector.processFrame(
        rateProbe.data(), rateProbe.size());
    requireKeyboard(std::abs(maximumRateConfidence - extremeRateConfidence) < 1.0e-6f,
                    "Extreme finite sample rates must clamp to the supported maximum");

    std::vector<float> clickFrame(frameSize, 0.0f);
    clickFrame[100] = 0.8f;
    float clickConfidence = 0.0f;
    for (int i = 0; i < 5; ++i)
        clickConfidence = detector.processFrame(clickFrame.data(), clickFrame.size());
    requireKeyboard(clickConfidence > 0.35f,
                    "Sharp broadband transients should raise keyboard confidence");

    detector.reset();
    std::vector<float> voiceLikeTone(frameSize);
    for (int frame = 0; frame < 12; ++frame) {
        for (std::size_t i = 0; i < frameSize; ++i) {
            const float phase = static_cast<float>(frame * frameSize + i);
            voiceLikeTone[i] = 0.18f * std::sin(
                2.0f * 3.14159265f * 180.0f * phase / 48000.0f);
        }
        detector.processFrame(voiceLikeTone.data(), voiceLikeTone.size());
    }
    requireKeyboard(detector.getConfidence() < 0.2f,
                    "A stable low-frequency tone should not look like keyboard clicks");

    // Long-running voiced speech has periodic syllable onsets and harmonic
    // energy. It must not accumulate keyboard confidence merely because its
    // amplitude rises and falls over many consecutive analysis frames.
    detector.reset();
    std::vector<float> voicedSpeech(frameSize);
    float maximumSpeechConfidence = 0.0f;
    for (int frame = 0; frame < 500; ++frame) {
        const float syllablePhase = static_cast<float>(frame % 18);
        const float time = static_cast<float>(frame * frameSize) / 48000.0f;
        const float syllableEnvelope = syllablePhase < 2.0f
            ? 0.0f
            : std::min(1.0f, (syllablePhase - 1.0f) / 2.0f);
        for (std::size_t i = 0; i < frameSize; ++i) {
            const float t = time + static_cast<float>(i) / 48000.0f;
            const float modulation = 0.55f + 0.45f
                * std::sin(2.0f * 3.14159265f * 3.0f * t)
                * std::sin(2.0f * 3.14159265f * 3.0f * t);
            voicedSpeech[i] = 0.18f * syllableEnvelope * modulation
                * (std::sin(2.0f * 3.14159265f * 130.0f * t)
                   + 0.45f * std::sin(2.0f * 3.14159265f * 260.0f * t)
                   + 0.20f * std::sin(2.0f * 3.14159265f * 520.0f * t));
        }
        maximumSpeechConfidence = std::max(maximumSpeechConfidence,
            detector.processFrame(voicedSpeech.data(), voicedSpeech.size()));
    }
    requireKeyboard(maximumSpeechConfidence < 0.40f,
                    "Long-running voiced speech must not accumulate strong keyboard confidence");

    // Speech remains present while occasional broadband key transients occur.
    // The mixed signal should score higher than speech alone, but must not
    // drive the detector into near-certain keyboard classification.
    smartgate::AdaptiveKeyboardDetector mixedDetector;
    mixedDetector.prepare(48000.0);
    float maximumMixedConfidence = 0.0f;
    for (int frame = 0; frame < 500; ++frame) {
        const float syllablePhase = static_cast<float>(frame % 18);
        const float time = static_cast<float>(frame * frameSize) / 48000.0f;
        const float syllableEnvelope = syllablePhase < 2.0f
            ? 0.0f
            : std::min(1.0f, (syllablePhase - 1.0f) / 2.0f);
        for (std::size_t i = 0; i < frameSize; ++i) {
            const float t = time + static_cast<float>(i) / 48000.0f;
            const float modulation = 0.55f + 0.45f
                * std::sin(2.0f * 3.14159265f * 3.0f * t)
                * std::sin(2.0f * 3.14159265f * 3.0f * t);
            voicedSpeech[i] = 0.18f * syllableEnvelope * modulation
                * (std::sin(2.0f * 3.14159265f * 130.0f * t)
                   + 0.45f * std::sin(2.0f * 3.14159265f * 260.0f * t)
                   + 0.20f * std::sin(2.0f * 3.14159265f * 520.0f * t));
        }
        if (frame % 8 == 4)
            voicedSpeech[100] += 0.35f;
        maximumMixedConfidence = std::max(maximumMixedConfidence,
            mixedDetector.processFrame(voicedSpeech.data(), voicedSpeech.size()));
    }
    requireKeyboard(maximumMixedConfidence > maximumSpeechConfidence + 0.02f,
                    "Intermittent keyboard transients should remain detectable over speech");
    requireKeyboard(maximumMixedConfidence < 0.50f,
                    "Mixed speech and keyboard input must not accumulate near-certain keyboard confidence");

    // Exercise each user-selectable classifier mode with voiced speech. The
    // more sensitive membrane weighting must still avoid a strong false
    // keyboard classification during ordinary sustained speech.
    for (int mode = 0; mode < 3; ++mode) {
        smartgate::AdaptiveKeyboardDetector modeDetector;
        modeDetector.prepare(48000.0);
        modeDetector.setMode(static_cast<smartgate::AdaptiveKeyboardDetector::Mode>(mode));
        float modeSpeechMaximum = 0.0f;
        for (int frame = 0; frame < 500; ++frame) {
            const float syllablePhase = static_cast<float>(frame % 18);
            const float time = static_cast<float>(frame * frameSize) / 48000.0f;
            const float syllableEnvelope = syllablePhase < 2.0f
                ? 0.0f
                : std::min(1.0f, (syllablePhase - 1.0f) / 2.0f);
            for (std::size_t i = 0; i < frameSize; ++i) {
                const float t = time + static_cast<float>(i) / 48000.0f;
                const float modulation = 0.55f + 0.45f
                    * std::sin(2.0f * 3.14159265f * 3.0f * t)
                    * std::sin(2.0f * 3.14159265f * 3.0f * t);
                voicedSpeech[i] = 0.18f * syllableEnvelope * modulation
                    * (std::sin(2.0f * 3.14159265f * 130.0f * t)
                       + 0.45f * std::sin(2.0f * 3.14159265f * 260.0f * t)
                       + 0.20f * std::sin(2.0f * 3.14159265f * 520.0f * t));
            }
            modeSpeechMaximum = std::max(modeSpeechMaximum,
                modeDetector.processFrame(voicedSpeech.data(), voicedSpeech.size()));
        }
        requireKeyboard(modeSpeechMaximum < 0.46f,
                        "No keyboard mode should strongly classify sustained voiced speech");
    }

    std::vector<float> silenceFrame(frameSize, 0.0f);
    for (int frame = 0; frame < 30; ++frame)
        detector.processFrame(silenceFrame.data(), silenceFrame.size());
    requireKeyboard(detector.getConfidence() < 0.01f,
                    "Keyboard confidence must decay during sustained silence");

    // Sustained broadband noise is bright, but it is not a sequence of
    // keyboard-like transients. The classifier must not rely on brightness
    // alone to create a high confidence score.
    detector.reset();
    std::uint32_t randomState = 0x12345678u;
    std::vector<float> broadbandNoise(frameSize);
    for (int frame = 0; frame < 40; ++frame) {
        for (std::size_t i = 0; i < frameSize; ++i) {
            float gaussianLike = 0.0f;
            for (int k = 0; k < 6; ++k) {
                randomState ^= randomState << 13;
                randomState ^= randomState >> 17;
                randomState ^= randomState << 5;
                gaussianLike += static_cast<float>(randomState & 0xffffu) / 65535.0f;
            }
            broadbandNoise[i] = (gaussianLike - 3.0f) * 0.045f;
        }
        detector.processFrame(broadbandNoise.data(), broadbandNoise.size());
    }
    requireKeyboard(detector.getConfidence() < 0.35f,
                    "Sustained broadband noise must not look strongly like keyboard clicks");

    detector.setMode(smartgate::AdaptiveKeyboardDetector::Mode::Mechanical);
    detector.reset();
    requireKeyboard(detector.processFrame(nullptr, frameSize) == 0.0f,
                    "Null frames must not produce a false positive");

    std::vector<float> invalidFrame(frameSize,
        std::numeric_limits<float>::quiet_NaN());
    const float invalidConfidence = detector.processFrame(
        invalidFrame.data(), invalidFrame.size());
    requireKeyboard(std::isfinite(invalidConfidence),
                    "NaN samples must not poison keyboard confidence");

    invalidFrame[17] = std::numeric_limits<float>::infinity();
    const float infiniteConfidence = detector.processFrame(
        invalidFrame.data(), invalidFrame.size());
    requireKeyboard(std::isfinite(infiniteConfidence),
                    "Infinite samples must not poison keyboard confidence");

    // Extreme but finite input must not overflow recursive low-pass state.
    // The following normal frames must remain finite after the detector reset.
    std::vector<float> extremeFrame(frameSize);
    const float largestFinite = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < frameSize; ++i)
        extremeFrame[i] = (i & 1u) == 0u ? largestFinite : -largestFinite;
    const float extremeConfidence = detector.processFrame(
        extremeFrame.data(), extremeFrame.size());
    requireKeyboard(std::isfinite(extremeConfidence),
                    "Extreme finite samples must not overflow keyboard confidence");
    detector.reset();
    const float recoveredConfidence = detector.processFrame(
        clickFrame.data(), clickFrame.size());
    requireKeyboard(std::isfinite(recoveredConfidence),
                    "Keyboard detector must recover after reset following extreme samples");

    // Disabling keyboard reduction pauses classification. Reproduce a
    // confident click sequence, then verify the disable/enable reset discards
    // both its confidence and its learned reference RMS.
    detector.reset();
    for (int i = 0; i < 5; ++i)
        detector.processFrame(clickFrame.data(), clickFrame.size());
    requireKeyboard(detector.getConfidence() > 0.35f,
                    "Precondition: click sequence should establish stale confidence");
    detector.reset();
    requireKeyboard(detector.getConfidence() == 0.0f,
                    "Reset must clear accumulated keyboard confidence");

    // Compare the reset instance against a genuinely fresh detector. Matching
    // output across the same frames verifies that reset also clears hidden
    // filter history and the learned reference RMS, not only public confidence.
    smartgate::AdaptiveKeyboardDetector freshDetector;
    freshDetector.prepare(48000.0);
    freshDetector.setMode(smartgate::AdaptiveKeyboardDetector::Mode::Mechanical);
    for (int i = 0; i < 5; ++i) {
        const float resetConfidence = detector.processFrame(
            clickFrame.data(), clickFrame.size());
        const float freshConfidence = freshDetector.processFrame(
            clickFrame.data(), clickFrame.size());
        requireKeyboard(std::abs(resetConfidence - freshConfidence) < 1.0e-6f,
                        "Reset detector must behave like a fresh detector");
    }
}
