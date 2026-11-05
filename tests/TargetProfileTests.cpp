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

    smartgate::AtomicTargetProfileSnapshot mailbox;
    std::uint64_t mailboxVersion = 0;
    smartgate::TargetProfile::Snapshot mailboxCopy;
    requireFeature(!mailbox.read(mailboxCopy, mailboxVersion),
                   "An unpublished mailbox must not expose a snapshot");
    mailbox.publish(profile.makeSnapshot());
    requireFeature(mailbox.read(mailboxCopy, mailboxVersion),
                   "A published mailbox snapshot must be readable");
    requireFeature(mailboxVersion != 0 && mailboxCopy.count == profile.frameCount(),
                   "Mailbox must preserve profile version and frame count");
    smartgate::TargetProfile mailboxRestored;
    requireFeature(mailboxRestored.restore(mailboxCopy)
                   && std::abs(mailboxRestored.confidence(extracted)
                               - profile.confidence(extracted)) < 1.0e-6f,
                   "Mailbox snapshot must preserve profile statistics");

    const auto saved = profile.makeSnapshot();
    smartgate::TargetProfile restored;
    requireFeature(restored.restore(saved), "A valid profile snapshot must restore");
    requireFeature(restored.frameCount() == profile.frameCount() && restored.isReady(),
                   "Restored profile must preserve training progress");
    requireFeature(std::abs(restored.confidence(extracted) - profile.confidence(extracted)) < 1.0e-6f,
                   "Restored profile must preserve confidence");

    auto corrupt = saved;
    corrupt.m2[2] = -1.0;
    requireFeature(!restored.restore(corrupt),
                   "Negative accumulated variance must be rejected");
    corrupt = saved;
    corrupt.mean[0] = std::numeric_limits<double>::infinity();
    requireFeature(!restored.restore(corrupt),
                   "Non-finite profile values must be rejected");
    corrupt = saved;
    corrupt.count = smartgate::TargetProfile::maximumRestoredFrames + 1;
    requireFeature(!restored.restore(corrupt),
                   "Unreasonably large restored frame counts must be rejected");

    // The serialized count is stored as a signed XML integer. Keep learning
    // within the same validated bound used by restore() so it cannot overflow.
    auto atLimit = saved;
    atLimit.count = smartgate::TargetProfile::maximumRestoredFrames;
    smartgate::TargetProfile capped;
    requireFeature(capped.restore(atLimit), "Maximum supported profile count must restore");
    requireFeature(!capped.learn(extracted), "Learning must stop at the supported frame-count limit");
    requireFeature(capped.frameCount() == smartgate::TargetProfile::maximumRestoredFrames,
                   "Rejected over-limit learning must not increment the frame count");

    profile.clear();
    requireFeature(!profile.isReady() && profile.frameCount() == 0,
                   "Clearing profile must reset training state");
}
