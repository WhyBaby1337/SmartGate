#include "AdaptiveKeyboardDetector.h"
#include "MouthClickSuppressor.h"
#include "MouseClickSuppressor.h"
#include "AudioFeatures.h"
#include "TargetProfile.h"
#include "LookaheadDelay.h"

#include <cmath>
#include <cstdio>
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
    // Frame-level classifier metadata must be delayed by exactly the same
    // number of samples as the audio lookahead path.
    smartgate::ScalarLookaheadDelay confidenceDelay;
    confidenceDelay.prepare(48000.0, 30.0);
    const auto confidenceDelaySamples = confidenceDelay.latencySamples();
    requireKeyboard(confidenceDelaySamples == 1440,
                    "Classifier metadata delay must match the 30 ms audio lookahead");
    for (std::size_t i = 0; i < confidenceDelaySamples; ++i)
        requireKeyboard(confidenceDelay.process(0.75f) == 0.0f,
                        "Classifier confidence must not arrive before its delayed audio");
    requireKeyboard(std::abs(confidenceDelay.process(0.25f) - 0.75f) < 1.0e-6f,
                    "Classifier confidence must emerge at the exact audio lookahead position");
    confidenceDelay.reset();
    requireKeyboard(confidenceDelay.process(1.0f) == 0.0f,
                    "Reset must clear stale delayed classifier confidence");

    // Output gain must move smoothly at sample rate instead of stepping when
    // the 256-sample classifier confidence changes at frame boundaries.
    smartgate::AdaptiveKeyboardGain smoothedGain;
    smoothedGain.prepare(48000.0);
    const float firstAttackSample = smoothedGain.process(1.0f, 1.0f);
    requireKeyboard(firstAttackSample < 1.0f && firstAttackSample > 0.98f,
                    "Keyboard attenuation must ramp rather than step on its first sample");
    float previousGain = firstAttackSample;
    for (int i = 1; i < 144; ++i) {
        const float gain = smoothedGain.process(1.0f, 1.0f);
        requireKeyboard(gain <= previousGain && gain >= 0.0f && std::isfinite(gain),
                        "Keyboard attack ramp must be finite, bounded, and monotonic");
        previousGain = gain;
    }
    requireKeyboard(smoothedGain.getGain() < 0.40f,
                    "Keyboard attenuation should reach useful level within its attack time");
    previousGain = smoothedGain.getGain();
    for (int i = 0; i < 1440; ++i) {
        const float gain = smoothedGain.process(1.0f, 0.0f);
        requireKeyboard(gain >= previousGain && gain <= 1.0f && std::isfinite(gain),
                        "Keyboard release ramp must be finite, bounded, and monotonic");
        previousGain = gain;
    }
    requireKeyboard(smoothedGain.getGain() > 0.75f,
                    "Keyboard attenuation should release smoothly without a long tail");
    smoothedGain.reset();
    requireKeyboard(smoothedGain.getGain() == 1.0f,
                    "Reset must restore unity keyboard output gain");

    // The attack and release are specified in milliseconds, not samples.
    // Their response should therefore remain stable across common host rates.
    float referenceAttackGain = -1.0f;
    float referenceReleaseGain = -1.0f;
    for (const double rate : {44100.0, 48000.0, 96000.0}) {
        smartgate::AdaptiveKeyboardGain rateGain;
        rateGain.prepare(rate);
        const int attackSamples = static_cast<int>(std::lround(0.003 * rate));
        const int releaseSamples = static_cast<int>(std::lround(0.030 * rate));
        float attackGain = 1.0f;
        for (int i = 0; i < attackSamples; ++i)
            attackGain = rateGain.process(1.0f, 1.0f);
        float releaseGain = attackGain;
        for (int i = 0; i < releaseSamples; ++i)
            releaseGain = rateGain.process(1.0f, 0.0f);
        if (referenceAttackGain < 0.0f) {
            referenceAttackGain = attackGain;
            referenceReleaseGain = releaseGain;
        } else {
            requireKeyboard(std::abs(attackGain - referenceAttackGain) < 0.002f,
                            "Keyboard attack time must be sample-rate independent");
            requireKeyboard(std::abs(releaseGain - referenceReleaseGain) < 0.002f,
                            "Keyboard release time must be sample-rate independent");
        }
    }

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
    requireKeyboard(smartgate::AdaptiveKeyboardDetector::gainForReduction(
                        1.0f, clickConfidence) < 0.75f,
                    "Confident keyboard transients should still receive useful reduction");
    requireKeyboard(smartgate::AdaptiveKeyboardDetector::gainForReduction(
                        0.0f, 1.0f) == 1.0f
                    && smartgate::AdaptiveKeyboardDetector::gainForReduction(
                        1.0f, 0.0f) == 1.0f
                    && smartgate::AdaptiveKeyboardDetector::gainForReduction(
                        1.0f, 1.0f) == 0.0f,
                    "Keyboard reduction curve must preserve its endpoint behavior");

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
    requireKeyboard(maximumSpeechConfidence < smartgate::TargetProfile::keyboardLearningConfidenceLimit,
                    "Ordinary voiced speech must remain eligible for target learning");
    requireKeyboard(smartgate::TargetProfile::isKeyboardQualifiedForLearning(
                        maximumSpeechConfidence),
                    "Ordinary voiced speech must pass the target-learning keyboard filter");
    requireKeyboard(smartgate::AdaptiveKeyboardDetector::gainForReduction(
                        1.0f, maximumSpeechConfidence) > 0.80f,
                    "Uncertain voiced speech must not be heavily attenuated");

    // Speech remains present while occasional broadband key transients occur.
    // High-band peak-to-average analysis should make those short impulses
    // actionable even though full-band speech energy masks their RMS change.
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
    requireKeyboard(!smartgate::TargetProfile::isKeyboardQualifiedForLearning(
                        maximumMixedConfidence),
                    "Mixed speech and keyboard transients must pause target learning");
    requireKeyboard(maximumMixedConfidence < 0.65f,
                    "Mixed speech and keyboard input must not accumulate near-certain keyboard confidence");
    const float mixedSpeechGain = smartgate::AdaptiveKeyboardDetector::gainForReduction(
        1.0f, maximumMixedConfidence);
    requireKeyboard(mixedSpeechGain < 0.85f && mixedSpeechGain > 0.55f,
                    "Keyboard transients over speech must receive useful but bounded attenuation");

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
        requireKeyboard(smartgate::AdaptiveKeyboardDetector::gainForReduction(
                            1.0f, modeSpeechMaximum) > 0.80f,
                        "Every keyboard mode must preserve most sustained speech level");
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

    // Stereo analysis must not use (L + R) / 2: equal-level anti-phase
    // channels would cancel a valid voice before VAD and feature extraction.
    smartgate::StereoAnalysisFoldDown stereoFoldDown;
    stereoFoldDown.prepare(48000.0);
    float antiPhasePeak = 0.0f;
    for (int i = 0; i < 512; ++i) {
        const float sample = 0.4f * std::sin(
            2.0f * 3.14159265f * 220.0f * static_cast<float>(i) / 48000.0f);
        const float analysis = stereoFoldDown.process(sample, -sample);
        antiPhasePeak = std::max(antiPhasePeak, std::abs(analysis));
    }
    requireKeyboard(antiPhasePeak > 0.39f,
                    "Anti-phase stereo must not disappear from mono analysis");

    // The analysis feed should follow the stronger channel instead of
    // attenuating it through averaging when channel levels differ.
    stereoFoldDown.reset();
    float selectedRightPeak = 0.0f;
    for (int i = 0; i < 2048; ++i) {
        const float phase = static_cast<float>(i);
        const float leftSample = 0.05f * std::sin(
            2.0f * 3.14159265f * 180.0f * phase / 48000.0f);
        const float rightSample = 0.35f * std::sin(
            2.0f * 3.14159265f * 180.0f * phase / 48000.0f);
        selectedRightPeak = std::max(selectedRightPeak,
            std::abs(stereoFoldDown.process(leftSample, rightSample)));
    }
    requireKeyboard(selectedRightPeak > 0.34f,
                    "Stereo analysis should select the stronger right channel");

    // Reset is used at mono/disabled-analysis boundaries. A previous stereo
    // layout must not bias channel selection when stereo analysis resumes.
    stereoFoldDown.reset();
    float selectedLeftAfterResetPeak = 0.0f;
    for (int i = 0; i < 256; ++i) {
        const float phase = static_cast<float>(i);
        const float leftSample = 0.35f * std::sin(
            2.0f * 3.14159265f * 180.0f * phase / 48000.0f);
        const float rightSample = 0.05f * std::sin(
            2.0f * 3.14159265f * 180.0f * phase / 48000.0f);
        selectedLeftAfterResetPeak = std::max(selectedLeftAfterResetPeak,
            std::abs(stereoFoldDown.process(leftSample, rightSample)));
    }
    requireKeyboard(selectedLeftAfterResetPeak > 0.34f,
                    "Reset must discard stale channel-energy history before stereo analysis resumes");

    // Mono input remains bit-transparent in the analysis path.
    stereoFoldDown.reset();
    for (int i = 0; i < 128; ++i) {
        const float sample = 0.2f * std::sin(static_cast<float>(i) * 0.1f);
        requireKeyboard(std::abs(stereoFoldDown.process(sample, sample) - sample) < 1.0e-6f,
                        "Mono input must pass through stereo analysis unchanged");
    }

    // Mouth-click suppression must be exactly transparent when disabled.
    smartgate::MouthClickSuppressor mouthBypass;
    mouthBypass.prepare(48000.0);
    for (int i = 0; i < 256; ++i) {
        const float sample = 0.2f * std::sin(2.0f * 3.14159265f * 4200.0f
                                             * static_cast<float>(i) / 48000.0f);
        requireKeyboard(mouthBypass.process(sample, 0.0f) == sample,
                        "Zero mouth-click reduction must remain bit-transparent");
    }

    // A sharp click-like impulse should reduce the high-band tail while the
    // low-frequency voice component remains effectively unchanged.
    smartgate::MouthClickSuppressor mouthClick;
    mouthClick.prepare(48000.0);
    float minimumHighBandGain = 1.0f;
    float maximumLowToneError = 0.0f;
    for (int i = 0; i < 1800; ++i) {
        const float tone = 0.12f * std::sin(2.0f * 3.14159265f * 180.0f
                                            * static_cast<float>(i) / 48000.0f);
        const float click = i == 300 ? 0.9f : (i > 300 && i < 340
            ? 0.20f * std::exp(-0.12f * static_cast<float>(i - 300))
                * std::sin(2.0f * 3.14159265f * 6200.0f
                           * static_cast<float>(i - 300) / 48000.0f)
            : 0.0f);
        const float output = mouthClick.process(tone + click, 1.0f);
        minimumHighBandGain = std::min(minimumHighBandGain,
                                       mouthClick.getHighBandGain());
        if (i > 700)
            maximumLowToneError = std::max(maximumLowToneError, std::abs(output - tone));
    }
    requireKeyboard(minimumHighBandGain < 0.8f,
                    "A sharp high-band click must trigger transient attenuation");
    requireKeyboard(maximumLowToneError < 0.002f,
                    "Mouth-click processing must leave low-frequency voice nearly unchanged");

    // A modest UI value should still produce useful reduction for a clear
    // mouth click, without changing the low-frequency voice path.
    smartgate::MouthClickSuppressor mouthLowSetting;
    mouthLowSetting.prepare(48000.0);
    float mouthLowSettingGain = 1.0f;
    for (int i = 0; i < 1200; ++i) {
        const float voice = 0.12f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        const float impulse = i == 300 ? 0.9f : 0.0f;
        mouthLowSetting.process(voice + impulse, 0.20f);
        mouthLowSettingGain = std::min(mouthLowSettingGain,
            mouthLowSetting.getHighBandGain());
    }
    requireKeyboard(mouthLowSettingGain < 0.82f,
                    "A modest 0.20 mouth-click setting must meaningfully attenuate a click");

    // Sustained sibilant-like energy must not be classified as a short click.
    smartgate::MouthClickSuppressor sibilantGuard;
    sibilantGuard.prepare(48000.0);
    float minimumSibilantGain = 1.0f;
    for (int i = 0; i < 12000; ++i) {
        const float sibilant = 0.08f * std::sin(2.0f * 3.14159265f * 6200.0f
                                                * static_cast<float>(i) / 48000.0f);
        sibilantGuard.process(sibilant, 1.0f);
        minimumSibilantGain = std::min(minimumSibilantGain,
                                       sibilantGuard.getHighBandGain());
    }
    requireKeyboard(minimumSibilantGain > 0.90f,
                    "Sustained high-frequency speech must not trigger mouth-click attenuation");

    stereoFoldDown.prepare(96000.0);
    requireKeyboard(std::abs(stereoFoldDown.process(0.1f, 0.3f) - 0.3f) < 1.0e-6f,
                    "Reprepare must clear stereo channel history and select the first stronger channel");

    // Keep dedicated mouse attenuation engaged through the mechanical
    // switch's short resonant tail, while retaining an exact zero-setting bypass.
    smartgate::MouseClickSuppressor mouseTailHold;
    mouseTailHold.prepare(48000.0);
    (void) mouseTailHold.process(0.8f, 1.0f);
    // The mouse detector confirms an isolated peak on the following sample,
    // when the expected sharp decay is observable. This one-sample decision
    // delay is intentional and prevents sustained/noise-like sibilants from
    // being attenuated on their ordinary waveform edges.
    (void) mouseTailHold.process(0.0f, 1.0f);
    const float gainAfterMouseAttack = mouseTailHold.getHighBandGain();
    requireKeyboard(gainAfterMouseAttack < 0.80f,
                    "A sharp mouse switch attack must trigger attenuation");
    for (int i = 0; i < 100; ++i)
        (void) mouseTailHold.process(0.0f, 1.0f);
    requireKeyboard(mouseTailHold.getHighBandGain()
                        <= gainAfterMouseAttack + 0.01f,
                    "Mouse attenuation must hold through the first two milliseconds of switch ringdown");
    smartgate::MouseClickSuppressor mouseZeroBypass;
    mouseZeroBypass.prepare(48000.0);
    for (int i = 0; i < 256; ++i) {
        const float sample = 0.17f * std::sin(2.0f * 3.14159265f
            * 7100.0f * static_cast<float>(i) / 48000.0f);
        requireKeyboard(mouseZeroBypass.process(sample, 0.0f, 1.0f) == sample,
                        "Zero mouse reduction must remain an exact audio bypass");
    }

    // Dedicated mouse stage: model a short switch click as a decaying,
    // high-frequency burst over a voiced fundamental. The attenuation must
    // persist across the click body, not only the first edge sample.
    smartgate::MouseClickSuppressor mouseClick;
    mouseClick.prepare(48000.0);
    float mouseBurstInputEnergy = 0.0f;
    float mouseBurstOutputEnergy = 0.0f;
    float mouseVoiceError = 0.0f;
    for (int i = 0; i < 2400; ++i) {
        const float voice = 0.12f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        float click = 0.0f;
        if (i >= 600 && i < 792) {
            const float age = static_cast<float>(i - 600);
            const float envelope = std::exp(-age / 55.0f);
            // A switch click is a short, broadband resonant transient,
            // not a perfectly sustained single-frequency sine burst.
            // Two high-frequency modes model its sharp attack and ringing.
            const float ageSeconds = static_cast<float>(i - 600) / 48000.0f;
            click = 0.12f * envelope
                * (std::sin(2.0f * 3.14159265f * 6200.0f * ageSeconds + 1.5707963f)
                   + 0.4f * std::sin(2.0f * 3.14159265f * 8700.0f * ageSeconds
                                     + 1.5707963f));
        }
        const float input = voice + click;
        const float output = mouseClick.process(input, 1.0f, 0.65f);
        if (i >= 600 && i < 792) {
            mouseBurstInputEnergy += click * click;
            const float outputHigh = output - voice;
            mouseBurstOutputEnergy += outputHigh * outputHigh;
        }
        if (i > 1200)
            mouseVoiceError = std::max(mouseVoiceError, std::abs(output - voice));
    }
    requireKeyboard(mouseBurstOutputEnergy < mouseBurstInputEnergy * 0.75f,
                    "Dedicated mouse stage must reduce a short switch-click burst over speech");
    requireKeyboard(mouseVoiceError < 0.003f,
                    "Mouse click suppression must preserve the low-frequency voice body");

    // A quieter mechanical click should still be reduced when it sits over
    // speech. This guards the useful lower end of the mouse detector without
    // broadening it into a continuous high-frequency voice attenuator.
    smartgate::MouseClickSuppressor quietMouseClick;
    quietMouseClick.prepare(48000.0);
    float quietMouseInputEnergy = 0.0f;
    float quietMouseOutputEnergy = 0.0f;
    for (int i = 0; i < 2400; ++i) {
        const float voice = 0.12f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        float click = 0.0f;
        if (i >= 600 && i < 792) {
            const float age = static_cast<float>(i - 600);
            const float envelope = std::exp(-age / 55.0f);
            const float ageSeconds = static_cast<float>(i - 600) / 48000.0f;
            click = 0.06f * envelope
                * (std::sin(2.0f * 3.14159265f * 6200.0f * ageSeconds + 1.5707963f)
                   + 0.4f * std::sin(2.0f * 3.14159265f * 8700.0f * ageSeconds
                                     + 1.5707963f));
        }
        const float output = quietMouseClick.process(voice + click, 1.0f, 0.65f);
        if (i >= 600 && i < 792) {
            quietMouseInputEnergy += click * click;
            const float outputHigh = output - voice;
            quietMouseOutputEnergy += outputHigh * outputHigh;
        }
    }
    requireKeyboard(quietMouseOutputEnergy < quietMouseInputEnergy * 0.97f,
                    "Mouse stage must detect and reduce a quieter switch click over speech");

    // Exercise different switch-ring profiles rather than tuning only for one
    // convenient resonance. These are synthetic stress cases, not claims about
    // the exact acoustic signature of any particular mouse model.
    struct MouseClickProfile {
        float firstFrequency;
        float secondFrequency;
        float amplitude;
        float decaySamples;
        int durationSamples;
    };
    const MouseClickProfile mouseProfiles[] = {
        {5200.0f, 7600.0f, 0.12f, 96.0f, 240},
        {9000.0f, 11500.0f, 0.04f, 45.0f, 144},
        {6800.0f, 9800.0f, 0.06f, 80.0f, 192}
    };
    for (const auto& profile : mouseProfiles) {
        smartgate::MouseClickSuppressor profiledMouseClick;
        smartgate::MouseClickSuppressor voiceOnlyReference;
        profiledMouseClick.prepare(48000.0);
        voiceOnlyReference.prepare(48000.0);
        float inputEnergy = 0.0f;
        float outputEnergy = 0.0f;
        float minimumProfileGain = 1.0f;
        float maximumProfileEdge = 0.0f;
        float maximumProfileConfidence = 0.0f;
        float maximumProfileLevel = 0.0f;
        float maximumProfileImpulsiveness = 0.0f;
        float maximumProfileNovelty = 0.0f;
        for (int i = 0; i < 2400; ++i) {
            const float voice = 0.12f * std::sin(2.0f * 3.14159265f
                * 180.0f * static_cast<float>(i) / 48000.0f);
            const float voiceOnlyOutput = voiceOnlyReference.process(
                voice, 1.0f, 0.65f);
            float click = 0.0f;
            if (i >= 600 && i < 600 + profile.durationSamples) {
                const float age = static_cast<float>(i - 600);
                const float time = age / 48000.0f;
                const float envelope = std::exp(-age / profile.decaySamples);
                click = profile.amplitude * envelope * (
                    std::sin(2.0f * 3.14159265f * profile.firstFrequency
                             * time + 1.5707963f)
                    + 0.4f * std::sin(2.0f * 3.14159265f
                             * profile.secondFrequency * time + 1.5707963f));
            }
            const float output = profiledMouseClick.process(
                voice + click, 1.0f, 0.65f);
            minimumProfileGain = std::min(minimumProfileGain,
                profiledMouseClick.getHighBandGain());
            if (i >= 600 && i < 600 + profile.durationSamples) {
                maximumProfileEdge = std::max(maximumProfileEdge,
                    profiledMouseClick.getLastEdge());
                maximumProfileConfidence = std::max(maximumProfileConfidence,
                    profiledMouseClick.getLastConfidence());
                maximumProfileLevel = std::max(maximumProfileLevel,
                    profiledMouseClick.getLastLevel());
                maximumProfileImpulsiveness = std::max(maximumProfileImpulsiveness,
                    profiledMouseClick.getLastImpulsiveness());
                maximumProfileNovelty = std::max(maximumProfileNovelty,
                    profiledMouseClick.getLastNovelty());
            }
            if (i >= 600 && i < 600 + profile.durationSamples) {
                inputEnergy += click * click;
                const float outputClick = output - voiceOnlyOutput;
                outputEnergy += outputClick * outputClick;
            }
        }
        if (!(outputEnergy < inputEnergy * 0.90f)) {
            const double ratio = inputEnergy > 0.0f
                ? static_cast<double>(outputEnergy / inputEnergy) : 0.0;
            std::fprintf(stderr,
                "Mouse profile %.0f/%.0f Hz: input=%.9g output=%.9g ratio=%.6f minGain=%.6f maxEdge=%.6f maxConfidence=%.6f level=%.6f impulse=%.6f novelty=%.6f\\n",
                profile.firstFrequency, profile.secondFrequency,
                static_cast<double>(inputEnergy), static_cast<double>(outputEnergy), ratio,
                static_cast<double>(minimumProfileGain),
                static_cast<double>(maximumProfileEdge),
                static_cast<double>(maximumProfileConfidence),
                static_cast<double>(maximumProfileLevel),
                static_cast<double>(maximumProfileImpulsiveness),
                static_cast<double>(maximumProfileNovelty));
        }
        requireKeyboard(outputEnergy < inputEnergy * 0.90f,
                        "Mouse detector must reduce varied resonant click profiles over speech");
    }

    smartgate::MouseClickSuppressor mouseSibilantGuard;
    mouseSibilantGuard.prepare(48000.0);
    float minimumMouseSibilantGain = 1.0f;
    float maximumSibilantEdge = 0.0f;
    float maximumSibilantConfidence = 0.0f;
    float maximumSibilantImpulsiveness = 0.0f;
    for (int i = 0; i < 12000; ++i) {
        const float sibilant = 0.08f * std::sin(2.0f * 3.14159265f * 6200.0f
                                                * static_cast<float>(i) / 48000.0f);
        mouseSibilantGuard.process(sibilant, 1.0f, 0.65f);
        minimumMouseSibilantGain = std::min(minimumMouseSibilantGain,
            mouseSibilantGuard.getHighBandGain());
        maximumSibilantEdge = std::max(maximumSibilantEdge,
            mouseSibilantGuard.getLastEdge());
        maximumSibilantConfidence = std::max(maximumSibilantConfidence,
            mouseSibilantGuard.getLastConfidence());
        maximumSibilantImpulsiveness = std::max(maximumSibilantImpulsiveness,
            mouseSibilantGuard.getLastImpulsiveness());
    }
    std::fprintf(stderr,
        "Sibilant guard: minGain=%.6f maxEdge=%.6f maxConfidence=%.6f maxImpulse=%.6f\\n",
        static_cast<double>(minimumMouseSibilantGain),
        static_cast<double>(maximumSibilantEdge),
        static_cast<double>(maximumSibilantConfidence),
        static_cast<double>(maximumSibilantImpulsiveness));
    requireKeyboard(minimumMouseSibilantGain > 0.90f,
                    "Sustained /s/, /sh/ and /ch/ energy must not be treated as mouse clicks");

    // Real sibilants are noise-like rather than clean sine waves. A sustained
    // high-band noise bed must not repeatedly retrigger the mouse transient
    // detector simply because adjacent waveform samples differ sharply.
    smartgate::MouseClickSuppressor noisySibilantGuard;
    noisySibilantGuard.prepare(48000.0);
    std::uint32_t sibilantNoiseState = 0x6d2b79f5u;
    float minimumNoisySibilantGain = 1.0f;
    for (int i = 0; i < 12000; ++i) {
        sibilantNoiseState ^= sibilantNoiseState << 13;
        sibilantNoiseState ^= sibilantNoiseState >> 17;
        sibilantNoiseState ^= sibilantNoiseState << 5;
        const float noise = (static_cast<float>(sibilantNoiseState & 0xffffu)
                             / 32767.5f - 1.0f) * 0.06f;
        (void) noisySibilantGuard.process(noise, 1.0f, 0.65f);
        if (i >= 2400)
            minimumNoisySibilantGain = std::min(minimumNoisySibilantGain,
                noisySibilantGuard.getHighBandGain());
    }
    requireKeyboard(minimumNoisySibilantGain > 0.95f,
                    "Steady-state noise-like sibilants must not be repeatedly attenuated as mouse clicks");

    smartgate::MouseClickSuppressor mouseBypass;
    mouseBypass.prepare(48000.0);
    const float bypassSample = 0.1234567f;
    requireKeyboard(mouseBypass.process(bypassSample, 0.0f, 1.0f) == bypassSample,
                    "Zero mouse reduction must remain bit-transparent");

    // Keyboard/mouse suppression must not attenuate the entire voice signal.
    // A broadband impulse is reduced in the high band while a low voice tone
    // remains effectively unchanged.
    smartgate::KeyboardTransientSuppressor keyboardClick;
    keyboardClick.prepare(48000.0);
    float clickInput = 0.0f;
    float clickOutput = 0.0f;
    float lowVoiceError = 0.0f;
    for (int i = 0; i < 2400; ++i) {
        const float voice = 0.14f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        const float impulse = i == 600 ? 0.8f : 0.0f;
        const float input = voice + impulse;
        const float output = keyboardClick.process(input, 1.0f);
        if (i == 600) {
            clickInput = input;
            clickOutput = output;
        }
        if (i > 1200)
            lowVoiceError = std::max(lowVoiceError, std::abs(output - voice));
    }
    requireKeyboard(std::abs(clickOutput) < std::abs(clickInput) * 0.55f,
                    "Keyboard/mouse impulse must be attenuated immediately");
    requireKeyboard(lowVoiceError < 0.002f,
                    "Keyboard transient suppression must preserve low-frequency voice");

    // Everyday UI values must produce audible-enough transient attenuation;
    // the former linear amount curve made 0.20 almost ineffective.
    smartgate::KeyboardTransientSuppressor lowSettingClick;
    lowSettingClick.prepare(48000.0);
    float lowSettingMinimumGain = 1.0f;
    for (int i = 0; i < 1200; ++i) {
        const float voice = 0.14f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        const float impulse = i == 300 ? 0.8f : 0.0f;
        lowSettingClick.process(voice + impulse, 0.20f);
        lowSettingMinimumGain = std::min(lowSettingMinimumGain,
            lowSettingClick.getHighBandGain());
    }
    requireKeyboard(lowSettingMinimumGain < 0.75f,
                    "A modest 0.20 keyboard setting must meaningfully attenuate a confident click");

    // Mechanical keyboard switches produce a short resonant tail after the
    // initial attack. Keep the detected attenuation stable through that tail
    // instead of releasing immediately after the first high-band peak.
    smartgate::KeyboardTransientSuppressor keyboardTailHold;
    keyboardTailHold.prepare(48000.0);
    (void) keyboardTailHold.process(0.8f, 1.0f);
    const float gainAfterKeyboardAttack = keyboardTailHold.getHighBandGain();
    requireKeyboard(gainAfterKeyboardAttack < 0.80f,
                    "A sharp keyboard switch attack must trigger attenuation");
    for (int i = 0; i < 100; ++i)
        (void) keyboardTailHold.process(0.0f, 1.0f);
    requireKeyboard(keyboardTailHold.getHighBandGain()
                        <= gainAfterKeyboardAttack + 0.01f,
                    "Keyboard attenuation must hold through the first two milliseconds of switch ringdown");

    // The fast suppressor uses only sample-local transient evidence. Its
    // classifier remains a separate frame-level signal for learning decisions;
    // it is deliberately not fed into the 30 ms delayed audio path.
    // A modest aligned classifier score should make a quieter impulse more
    // actionable without allowing frame confidence to attenuate non-transients.
    smartgate::KeyboardTransientSuppressor contextualClick;
    contextualClick.prepare(48000.0);
    smartgate::KeyboardTransientSuppressor localOnlyClick;
    localOnlyClick.prepare(48000.0);
    float contextualMinimumGain = 1.0f;
    float localOnlyMinimumGain = 1.0f;
    for (int i = 0; i < 1200; ++i) {
        const float voice = 0.10f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        const float impulse = i == 300 ? 0.24f : 0.0f;
        contextualClick.process(voice + impulse, 0.35f, 0.75f);
        localOnlyClick.process(voice + impulse, 0.35f, 0.0f);
        contextualMinimumGain = std::min(contextualMinimumGain,
            contextualClick.getHighBandGain());
        localOnlyMinimumGain = std::min(localOnlyMinimumGain,
            localOnlyClick.getHighBandGain());
    }
    requireKeyboard(contextualMinimumGain < localOnlyMinimumGain,
                    "Time-aligned classifier context must improve detection of a quieter click");
    requireKeyboard(contextualMinimumGain > 0.0f && std::isfinite(contextualMinimumGain),
                    "Context-assisted click attenuation must remain finite and bounded");

    // A quieter mechanical keyboard switch has a short resonant tail rather
    // than a full-scale single-sample impulse. Verify that aligned classifier
    // context can still catch it over a speaking voice at an everyday setting.
    smartgate::KeyboardTransientSuppressor quietMechanicalKeyboard;
    quietMechanicalKeyboard.prepare(48000.0);
    float quietKeyboardMinimumGain = 1.0f;
    float quietKeyboardVoiceError = 0.0f;
    for (int i = 0; i < 2400; ++i) {
        const float voice = 0.10f * std::sin(2.0f * 3.14159265f * 180.0f
                                             * static_cast<float>(i) / 48000.0f);
        float switchRing = 0.0f;
        if (i >= 300) {
            const float elapsed = static_cast<float>(i - 300);
            const float envelope = std::exp(-elapsed / 96.0f);
            switchRing = 0.12f * envelope * (
                std::sin(2.0f * 3.14159265f * 4100.0f * elapsed / 48000.0f)
                + 0.45f * std::sin(2.0f * 3.14159265f * 6700.0f * elapsed / 48000.0f));
        }
        const float output = quietMechanicalKeyboard.process(
            voice + switchRing, 0.35f, 0.75f);
        quietKeyboardMinimumGain = std::min(quietKeyboardMinimumGain,
            quietMechanicalKeyboard.getHighBandGain());
        if (i > 1200)
            quietKeyboardVoiceError = std::max(quietKeyboardVoiceError,
                std::abs(output - voice));
    }
    requireKeyboard(quietKeyboardMinimumGain < 0.95f,
                    "Aligned keyboard detection must reduce a quiet resonant switch click");
    requireKeyboard(quietKeyboardVoiceError < 0.003f,
                    "Quiet keyboard click suppression must preserve the low voice fundamental");

    // Cover a range of resonant switch signatures rather than tuning the
    // transient path to one synthetic keyboard frequency. The voice-only
    // reference removes filter phase from the energy comparison.
    struct KeyboardClickProfile {
        float frequencyA;
        float frequencyB;
        float amplitude;
        float decaySamples;
        int durationSamples;
    };
    const KeyboardClickProfile keyboardProfiles[] = {
        { 3100.0f,  4600.0f, 0.075f,  52.0f, 180},
        { 4200.0f,  6900.0f, 0.060f,  84.0f, 240},
        { 5600.0f,  8200.0f, 0.050f, 110.0f, 300},
        { 7600.0f, 10400.0f, 0.045f,  42.0f, 150},
        // A much quieter contact strike checks that suppression is not tuned
        // only for obvious, high-amplitude switch sounds.
        { 3900.0f,  5800.0f, 0.025f,  70.0f, 210}
    };

    // Verify that the real frame classifier recognizes these switch attacks
    // over voiced audio. The suppressor-only profiles below intentionally use
    // a fixed context value; this separate check prevents that fixture from
    // hiding a classifier that never supplies useful context in the processor.
    for (const auto& profile : keyboardProfiles) {
        smartgate::AdaptiveKeyboardDetector clickDetector;
        smartgate::AdaptiveKeyboardDetector speechDetector;
        clickDetector.prepare(48000.0);
        speechDetector.prepare(48000.0);
        std::vector<float> frame(frameSize, 0.0f);

        for (int frameIndex = 0; frameIndex < 3; ++frameIndex) {
            for (std::size_t i = 0; i < frameSize; ++i) {
                const int sampleIndex = frameIndex * static_cast<int>(frameSize)
                    + static_cast<int>(i);
                frame[i] = 0.10f * std::sin(2.0f * 3.14159265f * 180.0f
                    * static_cast<float>(sampleIndex) / 48000.0f);
            }
            (void) clickDetector.processFrame(frame.data(), frame.size());
            (void) speechDetector.processFrame(frame.data(), frame.size());
        }

        for (std::size_t i = 0; i < frameSize; ++i) {
            const int sampleIndex = 3 * static_cast<int>(frameSize)
                + static_cast<int>(i);
            const float voice = 0.10f * std::sin(2.0f * 3.14159265f * 180.0f
                * static_cast<float>(sampleIndex) / 48000.0f);
            const int age = static_cast<int>(i) - 100;
            float click = 0.0f;
            if (age >= 0 && age < profile.durationSamples) {
                const float elapsed = static_cast<float>(age);
                const float envelope = std::exp(-elapsed / profile.decaySamples);
                const float time = elapsed / 48000.0f;
                click = profile.amplitude * envelope * (
                    std::sin(2.0f * 3.14159265f * profile.frequencyA * time
                             + 1.5707963f)
                    + 0.40f * std::sin(2.0f * 3.14159265f
                        * profile.frequencyB * time + 1.5707963f));
                if (age == 0)
                    click += 3.0f * profile.amplitude;
            }
            frame[i] = voice + click;
        }

        std::vector<float> speechFrame(frameSize, 0.0f);
        for (std::size_t i = 0; i < frameSize; ++i) {
            const int sampleIndex = 3 * static_cast<int>(frameSize)
                + static_cast<int>(i);
            speechFrame[i] = 0.10f * std::sin(2.0f * 3.14159265f * 180.0f
                * static_cast<float>(sampleIndex) / 48000.0f);
        }
        const float clickContext = clickDetector.processFrame(frame.data(), frame.size());
        const float speechContext = speechDetector.processFrame(
            speechFrame.data(), speechFrame.size());
        std::fprintf(stderr,
            "Keyboard classifier %.0f/%.0f Hz: click=%.6f voiced=%.6f\n",
            static_cast<double>(profile.frequencyA),
            static_cast<double>(profile.frequencyB), clickContext, speechContext);
        requireKeyboard(clickContext > speechContext + 0.10f,
            "Keyboard classifier must distinguish switch attacks from voiced speech");
    }
    for (const auto& profile : keyboardProfiles) {
        smartgate::KeyboardTransientSuppressor profileSuppressor;
        smartgate::KeyboardTransientSuppressor voiceReference;
        profileSuppressor.prepare(48000.0);
        voiceReference.prepare(48000.0);
        double inputClickEnergy = 0.0;
        double outputClickEnergy = 0.0;
        for (int i = 0; i < 1800; ++i) {
            const float voice = 0.10f * std::sin(
                2.0f * 3.14159265f * 180.0f * static_cast<float>(i) / 48000.0f);
            float click = 0.0f;
            const int age = i - 360;
            if (age >= 0 && age < profile.durationSamples) {
                const float elapsed = static_cast<float>(age);
                const float envelope = std::exp(-elapsed / profile.decaySamples);
                const float time = elapsed / 48000.0f;
                click = profile.amplitude * envelope * (
                    std::sin(2.0f * 3.14159265f * profile.frequencyA * time
                             + 1.5707963f)
                    + 0.40f * std::sin(
                        2.0f * 3.14159265f * profile.frequencyB * time
                        + 1.5707963f));
                // Model the short broadband contact attack that excites the
                // resonant tail; a steady-state sine burst is not a switch click.
                if (age == 0)
                    click += 3.0f * profile.amplitude;
            }
            const float output = profileSuppressor.process(
                voice + click, 0.35f, 0.75f);
            const float voiceOnly = voiceReference.process(voice, 0.0f, 0.0f);
            if (age >= 0 && age < profile.durationSamples) {
                const float residual = output - voiceOnly;
                inputClickEnergy += static_cast<double>(click) * click;
                outputClickEnergy += static_cast<double>(residual) * residual;
            }
        }
        requireKeyboard(inputClickEnergy > 1.0e-8,
                        "Keyboard profile fixture must contain measurable click energy");
        const double profileRatio = outputClickEnergy / inputClickEnergy;
        std::fprintf(stderr,
            "Keyboard profile %.0f/%.0f Hz: residual ratio=%.6f\\n",
            static_cast<double>(profile.frequencyA),
            static_cast<double>(profile.frequencyB), profileRatio);
        requireKeyboard(profileRatio < 0.98,
                        "Keyboard transient suppression must reduce varied resonant switch profiles");
    }

    // End-to-end path: derive frame context from the actual mixed voice+click
    // signal, then feed that context into the sample-accurate suppressor. This
    // avoids a hard-coded confidence that could hide a classifier/suppressor
    // integration regression.
    for (const auto& profile : keyboardProfiles) {
        smartgate::AdaptiveKeyboardDetector integratedDetector;
        smartgate::KeyboardTransientSuppressor integratedSuppressor;
        smartgate::KeyboardTransientSuppressor integratedVoiceReference;
        integratedDetector.prepare(48000.0);
        integratedSuppressor.prepare(48000.0);
        integratedVoiceReference.prepare(48000.0);

        std::vector<float> analysisFrame(frameSize, 0.0f);
        for (int warmup = 0; warmup < 3; ++warmup) {
            for (std::size_t i = 0; i < frameSize; ++i) {
                const int sampleIndex = warmup * static_cast<int>(frameSize)
                    + static_cast<int>(i);
                analysisFrame[i] = 0.10f * std::sin(
                    2.0f * 3.14159265f * 180.0f
                    * static_cast<float>(sampleIndex) / 48000.0f);
            }
            (void) integratedDetector.processFrame(
                analysisFrame.data(), analysisFrame.size());
        }

        float frameContexts[2] = {};
        for (int frameIndex = 0; frameIndex < 2; ++frameIndex) {
            for (std::size_t i = 0; i < frameSize; ++i) {
                const int streamIndex = frameIndex * static_cast<int>(frameSize)
                    + static_cast<int>(i);
                const int age = streamIndex - 100;
                const int sampleIndex = 3 * static_cast<int>(frameSize)
                    + streamIndex;
                const float voice = 0.10f * std::sin(
                    2.0f * 3.14159265f * 180.0f
                    * static_cast<float>(sampleIndex) / 48000.0f);
                float click = 0.0f;
                if (age >= 0 && age < profile.durationSamples) {
                    const float elapsed = static_cast<float>(age);
                    const float envelope = std::exp(-elapsed / profile.decaySamples);
                    const float time = elapsed / 48000.0f;
                    click = profile.amplitude * envelope * (
                        std::sin(2.0f * 3.14159265f
                            * profile.frequencyA * time + 1.5707963f)
                        + 0.40f * std::sin(2.0f * 3.14159265f
                            * profile.frequencyB * time + 1.5707963f));
                    if (age == 0)
                        click += 3.0f * profile.amplitude;
                }
                analysisFrame[i] = voice + click;
            }
            frameContexts[frameIndex] = integratedDetector.processFrame(
                analysisFrame.data(), analysisFrame.size());
        }

        double integratedInputEnergy = 0.0;
        double integratedOutputEnergy = 0.0;
        for (int i = 0; i < 2 * static_cast<int>(frameSize); ++i) {
            const int age = i - 100;
            const int sampleIndex = 3 * static_cast<int>(frameSize) + i;
            const float voice = 0.10f * std::sin(
                2.0f * 3.14159265f * 180.0f
                * static_cast<float>(sampleIndex) / 48000.0f);
            float click = 0.0f;
            if (age >= 0 && age < profile.durationSamples) {
                const float elapsed = static_cast<float>(age);
                const float envelope = std::exp(-elapsed / profile.decaySamples);
                const float time = elapsed / 48000.0f;
                click = profile.amplitude * envelope * (
                    std::sin(2.0f * 3.14159265f
                        * profile.frequencyA * time + 1.5707963f)
                    + 0.40f * std::sin(2.0f * 3.14159265f
                        * profile.frequencyB * time + 1.5707963f));
                if (age == 0)
                    click += 3.0f * profile.amplitude;
            }
            const float context = frameContexts[i / static_cast<int>(frameSize)];
            const float output = integratedSuppressor.process(
                voice + click, 0.35f, context);
            const float voiceOnly = integratedVoiceReference.process(
                voice, 0.0f, 0.0f);
            if (age >= 0 && age < profile.durationSamples) {
                const float residual = output - voiceOnly;
                integratedInputEnergy += static_cast<double>(click) * click;
                integratedOutputEnergy += static_cast<double>(residual) * residual;
            }
        }
        requireKeyboard(frameContexts[0] > 0.0f,
            "Integrated keyboard path must derive nonzero context from a mixed click frame");
        requireKeyboard(integratedInputEnergy > 1.0e-8,
            "Integrated keyboard fixture must contain measurable click energy");
        requireKeyboard(integratedOutputEnergy / integratedInputEnergy < 0.98,
            "Classifier-driven keyboard suppression must reduce varied switch profiles");
    }

    // Sustained /s/ and /sh/-like high-frequency energy is speech, not a click.
    smartgate::KeyboardTransientSuppressor sibilantKeyboardGuard;
    sibilantKeyboardGuard.prepare(48000.0);
    float minimumKeyboardSibilantGain = 1.0f;
    for (int i = 0; i < 12000; ++i) {
        const float sibilant = 0.08f * std::sin(2.0f * 3.14159265f * 6200.0f
                                                * static_cast<float>(i) / 48000.0f);
        sibilantKeyboardGuard.process(sibilant, 1.0f);
        minimumKeyboardSibilantGain = std::min(minimumKeyboardSibilantGain,
            sibilantKeyboardGuard.getHighBandGain());
    }
    requireKeyboard(minimumKeyboardSibilantGain > 0.95f,
                    "Keyboard suppression must preserve sustained sibilant speech");

    // A stale/high aligned frame context can follow a real click into
    // nearby speech. Even at maximum context, a sustained high-frequency
    // sibilant must not be mistaken for a sequence of isolated switch attacks.
    smartgate::KeyboardTransientSuppressor contextualSibilantGuard;
    contextualSibilantGuard.prepare(48000.0);
    float minimumContextualSibilantGain = 1.0f;
    for (int i = 0; i < 12000; ++i) {
        const float sibilant = 0.08f * std::sin(
            2.0f * 3.14159265f * 6200.0f * static_cast<float>(i) / 48000.0f);
        (void) contextualSibilantGuard.process(sibilant, 1.0f, 1.0f);
        minimumContextualSibilantGain = std::min(minimumContextualSibilantGain,
            contextualSibilantGuard.getHighBandGain());
    }
    requireKeyboard(minimumContextualSibilantGain > 0.95f,
        "Maximum aligned keyboard context must not attenuate sustained sibilant speech");

    // Mid-band attenuation must react to a classifier-aligned mechanical
    // impulse, but remain effectively transparent for a sustained speech-like
    // tone in the same band (a conservative guard for formants and consonants).
    smartgate::KeyboardTransientSuppressor midBandClick;
    midBandClick.prepare(48000.0);
    (void) midBandClick.process(0.0f, 0.25f, 1.0f);
    (void) midBandClick.process(0.50f, 0.25f, 1.0f);
    requireKeyboard(midBandClick.getMidBandGain() < 0.95f,
                    "Aligned mechanical impulses should attenuate the mid band");

    smartgate::KeyboardTransientSuppressor midBandSpeech;
    midBandSpeech.prepare(48000.0);
    float minimumMidSpeechGain = 1.0f;
    for (int i = 0; i < 12000; ++i) {
        const float speechBand = 0.08f * std::sin(
            2.0f * 3.14159265f * 1800.0f * static_cast<float>(i) / 48000.0f);
        (void) midBandSpeech.process(speechBand, 0.25f, 1.0f);
        minimumMidSpeechGain = std::min(minimumMidSpeechGain,
            midBandSpeech.getMidBandGain());
    }
    requireKeyboard(minimumMidSpeechGain > 0.98f,
                    "Sustained speech-like mid-band energy must not be attenuated");

    // Zero reduction is bit-transparent even though the analysis filter stays warm.
    smartgate::KeyboardTransientSuppressor keyboardBypass;
    keyboardBypass.prepare(48000.0);
    for (int i = 0; i < 256; ++i) {
        const float sample = 0.2f * std::sin(2.0f * 3.14159265f * 4200.0f
                                             * static_cast<float>(i) / 48000.0f);
        requireKeyboard(keyboardBypass.process(sample, 0.0f) == sample,
                        "Zero keyboard reduction must remain bit-transparent");
    }

}
