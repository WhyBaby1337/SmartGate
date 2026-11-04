#include "AudioFeatures.h"
#include "TargetProfile.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void requireFeature(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runTargetProfileTests() {
    smartgate::FeatureExtractor extractor;
    extractor.prepare(48000.0);

    std::vector<float> frame(480);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const float t = static_cast<float>(i) / 48000.0f;
        frame[i] = 0.2f * std::sin(2.0f * 3.14159265f * 180.0f * t)
                 + 0.04f * std::sin(2.0f * 3.14159265f * 1200.0f * t);
    }

    const auto extracted = extractor.process(frame.data(), frame.size());
    requireFeature(extracted.isFinite(), "Extracted features must be finite");
    requireFeature(extracted.logRms < 0.0f && extracted.logRms > -2.0f,
                   "Log RMS must represent a valid non-silent frame");
    const float ratioSum = extracted.lowBandRatio + extracted.midBandRatio
                         + extracted.highBandRatio;
    requireFeature(std::abs(ratioSum - 1.0f) < 1.0e-4f,
                   "Band energy ratios must sum to one");

    smartgate::TargetProfile profile;
    requireFeature(profile.confidence(extracted) == 0.0f,
                   "Untrained profile must not report confidence");
    for (int i = 0; i < 32; ++i)
        requireFeature(profile.learn(extracted), "Valid feature frame must be learned");
    requireFeature(profile.isReady(), "Profile must become ready after minimum training");
    requireFeature(profile.frameCount() == 32, "Profile frame count must be accurate");
    requireFeature(profile.confidence(extracted) > 0.99f,
                   "Profile must strongly match its training features");

    const smartgate::AudioFeatures unrelated{
        -3.0f, 0.45f, 0.02f, 0.08f, 0.90f
    };
    requireFeature(profile.confidence(unrelated) < 0.1f,
                   "Profile must reject strongly dissimilar features");

    const smartgate::AudioFeatures invalid{
        std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, 0.0f
    };
    requireFeature(!profile.learn(invalid), "Invalid feature frames must be rejected");
    requireFeature(profile.frameCount() == 32,
                   "Rejected frames must not modify profile statistics");

    profile.clear();
    requireFeature(!profile.isReady() && profile.frameCount() == 0,
                   "Clearing profile must reset training state");
}
