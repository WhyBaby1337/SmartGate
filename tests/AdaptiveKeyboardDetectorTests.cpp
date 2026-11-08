#include "AdaptiveKeyboardDetector.h"
#include "MouthClickSuppressor.h"
#include "AudioFeatures.h"
#include "TargetProfile.h"
#include "LookaheadDelay.h"

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
