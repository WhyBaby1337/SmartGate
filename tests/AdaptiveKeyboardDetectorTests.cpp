#include "AdaptiveKeyboardDetector.h"

#include <cmath>
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

    detector.setMode(smartgate::AdaptiveKeyboardDetector::Mode::Mechanical);
    detector.reset();
    requireKeyboard(detector.processFrame(nullptr, frameSize) == 0.0f,
                    "Null frames must not produce a false positive");
}
