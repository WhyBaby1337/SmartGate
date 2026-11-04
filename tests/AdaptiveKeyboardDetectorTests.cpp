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

    detector.reset();
    requireKeyboard(detector.getConfidence() == 0.0f,
                    "Reset must clear accumulated keyboard confidence");
}
