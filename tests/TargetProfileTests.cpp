#include "AudioFeatures.h"
#include "TargetProfile.h"

#include <cmath>
#include <array>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void requireFeature(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runTargetProfileTests() {
    requireFeature(smartgate::finiteSampleOrZero(0.25f) == 0.25f,
                   "Finite audio samples must be preserved");
    requireFeature(smartgate::finiteSampleOrZero(
                       std::numeric_limits<float>::quiet_NaN()) == 0.0f,
                   "NaN audio samples must be replaced with silence");
    requireFeature(smartgate::finiteSampleOrZero(
                       std::numeric_limits<float>::infinity()) == 0.0f,
                   "Infinite audio samples must be replaced with silence");
    requireFeature(smartgate::finiteSampleOrZero(
                       -std::numeric_limits<float>::infinity()) == 0.0f,
                   "Negative infinite audio samples must be replaced with silence");

    smartgate::FeatureExtractor extractor;
    extractor.prepare(48000.0);

    std::vector<float> frame(480);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const float t = static_cast<float>(i) / 48000.0f;
        frame[i] = 0.2f * std::sin(2.0f * 3.14159265f * 180.0f * t)
                 + 0.04f * std::sin(2.0f * 3.14159265f * 1200.0f * t);
    }

    // A non-finite first sample is sanitized to zero and must not create a
    // spurious zero-crossing against the next positive sample.
    smartgate::FeatureExtractor nonFiniteStartExtractor;
    nonFiniteStartExtractor.prepare(48000.0);
    const std::vector<float> nonFiniteStartFrame{
        std::numeric_limits<float>::quiet_NaN(), 0.1f, 0.1f, 0.1f
    };
    const auto nonFiniteStartFeatures = nonFiniteStartExtractor.process(
        nonFiniteStartFrame.data(), nonFiniteStartFrame.size());
    requireFeature(nonFiniteStartFeatures.isFinite(),
                   "A non-finite input sample must be sanitized");
    requireFeature(nonFiniteStartFeatures.zeroCrossingRate == 0.0f,
                   "Sanitized first sample must not create a false zero-crossing");

    // Very large but finite samples must not overflow recursive float filter
    // state and poison every subsequent frame after the input returns to normal.
    smartgate::FeatureExtractor extremeExtractor;
    extremeExtractor.prepare(48000.0);
    const float largestFinite = std::numeric_limits<float>::max();
    const std::array<float, 8> extremeFrame{
        largestFinite, -largestFinite, largestFinite, -largestFinite,
        largestFinite, -largestFinite, largestFinite, -largestFinite
    };
    const auto extremeFeatures = extremeExtractor.process(
        extremeFrame.data(), extremeFrame.size());
    requireFeature(extremeFeatures.isFinite(),
                   "Extreme finite samples must not overflow feature filter state");
    const auto recoveredFeatures = extremeExtractor.process(frame.data(), frame.size());
    requireFeature(recoveredFeatures.isFinite()
                   && recoveredFeatures.logRms < 0.0f
                   && recoveredFeatures.logRms > -2.0f,
                   "Feature extraction must recover when normal audio follows extreme samples");

    const auto extracted = extractor.process(frame.data(), frame.size());
    requireFeature(extracted.isFinite(), "Extracted features must be finite");

    // Direct DSP users receive the same bounded rate behavior as the plugin
    // processor, even when a host supplies an extreme finite sample rate.
    smartgate::FeatureExtractor maximumRateExtractor;
    smartgate::FeatureExtractor extremeRateExtractor;
    maximumRateExtractor.prepare(768000.0);
    extremeRateExtractor.prepare(std::numeric_limits<double>::max());
    const auto maximumRateFeatures = maximumRateExtractor.process(frame.data(), frame.size());
    const auto extremeRateFeatures = extremeRateExtractor.process(frame.data(), frame.size());
    requireFeature(std::abs(maximumRateFeatures.logRms - extremeRateFeatures.logRms) < 1.0e-6f
                   && std::abs(maximumRateFeatures.lowBandRatio - extremeRateFeatures.lowBandRatio) < 1.0e-6f
                   && std::abs(maximumRateFeatures.midBandRatio - extremeRateFeatures.midBandRatio) < 1.0e-6f
                   && std::abs(maximumRateFeatures.highBandRatio - extremeRateFeatures.highBandRatio) < 1.0e-6f,
                   "Extreme finite sample rates must clamp to the supported maximum");
    requireFeature(extracted.logRms < 0.0f && extracted.logRms > -2.0f,
                   "Log RMS must represent a valid non-silent frame");
    const float ratioSum = extracted.lowBandRatio + extracted.midBandRatio
                         + extracted.highBandRatio;
    requireFeature(std::abs(ratioSum - 1.0f) < 1.0e-4f,
                   "Band energy ratios must sum to one");

    requireFeature(smartgate::TargetProfile::isValidLearningFrame(extracted),
                   "A finite speech-level frame must be eligible for target learning");
    requireFeature(!smartgate::TargetProfile::isValidLearningFrame(
                       {-3.0f, 0.2f, 0.2f, 0.3f, 0.5f}),
                   "Silent frames must not be eligible for target learning");
    requireFeature(!smartgate::TargetProfile::isValidLearningFrame(
                       {-0.01f, 0.2f, 0.2f, 0.3f, 0.5f}),
                   "Heavily clipped frames must not be eligible for target learning");
    requireFeature(!smartgate::TargetProfile::isValidLearningFrame(
                       {std::numeric_limits<float>::quiet_NaN(), 0.2f, 0.2f, 0.3f, 0.5f}),
                   "Non-finite frames must not be eligible for target learning");

    requireFeature(smartgate::TargetProfile::isSpeechQualifiedForLearning(false, 0.0f),
                   "Feature-only fallback must remain available when Silero is unavailable");
    requireFeature(smartgate::TargetProfile::isSpeechQualifiedForLearning(true, 0.50f),
                   "Silero confidence at the learning threshold must be accepted");
    requireFeature(smartgate::TargetProfile::isSpeechQualifiedForLearning(true, 0.75f),
                   "Confident Silero speech must be accepted");
    requireFeature(!smartgate::TargetProfile::isSpeechQualifiedForLearning(true, 0.49f),
                   "Low Silero confidence must pause target learning");
    requireFeature(!smartgate::TargetProfile::isSpeechQualifiedForLearning(
                       true, std::numeric_limits<float>::quiet_NaN()),
                   "NaN Silero confidence must never qualify target learning");
    requireFeature(!smartgate::TargetProfile::isSpeechQualifiedForLearning(
                       true, std::numeric_limits<float>::infinity()),
                   "Infinite Silero confidence must never qualify target learning");

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
    requireFeature(!profile.learn({-3.0f, 0.2f, 0.2f, 0.3f, 0.5f}),
                   "Out-of-window finite RMS values must be rejected");
    requireFeature(!profile.learn({-1.0f, 1.2f, 0.2f, 0.3f, 0.5f}),
                   "Out-of-range zero-crossing rates must be rejected");
    requireFeature(!profile.learn({-1.0f, 0.2f, 0.8f, 0.3f, 0.5f}),
                   "Spectral ratios that do not sum to one must be rejected");
    requireFeature(profile.frameCount() == 32,
                   "Rejected frames must not modify profile statistics");

    smartgate::AtomicTargetProfileSnapshot mailbox;
    std::uint64_t mailboxVersion = 0;
    smartgate::TargetProfile::Snapshot mailboxCopy;
    requireFeature(!mailbox.read(mailboxCopy, mailboxVersion),
                   "An unpublished mailbox must not expose a snapshot");
    requireFeature(!mailbox.readForPersistence(mailboxCopy, mailboxVersion),
                   "An unpublished mailbox must represent an empty pre-prepare state");
    requireFeature(!mailbox.read(mailboxCopy, mailboxVersion, 0),
                   "A mailbox read with zero attempts must fail without spinning");
    mailbox.publish(profile.makeSnapshot());
    requireFeature(mailbox.readForPersistence(mailboxCopy, mailboxVersion),
                   "Persistence must read a published profile snapshot");
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
    corrupt = saved;
    corrupt.mean[0] = 1.0e100;
    requireFeature(!restored.restore(corrupt),
                   "Finite but impossible feature means must be rejected");
    corrupt = saved;
    corrupt.m2[1] = static_cast<double>(corrupt.count);
    requireFeature(!restored.restore(corrupt),
                   "Variance beyond the mathematical feature-range bound must be rejected");
    corrupt = saved;
    corrupt.mean[2] += 0.1;
    requireFeature(!restored.restore(corrupt),
                   "Spectral ratios that cannot sum to one must be rejected");

    requireFeature(restored.frameCount() == saved.count
                   && std::abs(restored.confidence(extracted)
                               - profile.confidence(extracted)) < 1.0e-6f,
                   "A rejected restore must leave the previously valid profile untouched");

    requireFeature(restored.restore(smartgate::TargetProfile::Snapshot{}),
                   "An empty snapshot must restore successfully");
    requireFeature(restored.frameCount() == 0 && !restored.isReady()
                   && restored.confidence(extracted) == 0.0f,
                   "Restoring an empty profile must clear all previous training state");
    requireFeature(restored.restore(saved),
                   "A valid profile must still restore after an explicit clear");

    // The serialized count is stored as a signed XML integer. Keep learning
    // within the same validated bound used by restore() so it cannot overflow.
    auto atLimit = saved;
    atLimit.count = smartgate::TargetProfile::maximumRestoredFrames;
    smartgate::TargetProfile capped;
    requireFeature(capped.restore(atLimit), "Maximum supported profile count must restore");
    requireFeature(!capped.learn(extracted), "Learning must stop at the supported frame-count limit");
    requireFeature(capped.frameCount() == smartgate::TargetProfile::maximumRestoredFrames,
                   "Rejected over-limit learning must not increment the frame count");

    // A just-restored pending snapshot must be saved before the audio thread
    // applies it, then the published snapshot must win after application.
    smartgate::AtomicTargetProfileSnapshot pendingForSave;
    smartgate::AtomicTargetProfileSnapshot publishedForSave;
    std::atomic<std::uint64_t> appliedVersion{0};
    smartgate::TargetProfile::Snapshot selected;
    pendingForSave.publish(saved);
    requireFeature(smartgate::readLatestTargetProfileForPersistence(
                       pendingForSave, publishedForSave, appliedVersion, selected),
                   "A pending restored profile must be available for immediate save");
    requireFeature(selected.count == saved.count,
                   "Immediate save must select the pending restored profile");

    auto newer = saved;
    newer.count += 1;
    // This is a valid synthetic snapshot with the same bounded statistics.
    pendingForSave.publish(saved);
    publishedForSave.publish(newer);
    std::uint64_t pendingVersion = 0;
    smartgate::TargetProfile::Snapshot pendingCopy;
    requireFeature(pendingForSave.read(pendingCopy, pendingVersion),
                   "Pending profile version must be readable");
    appliedVersion.store(pendingVersion, std::memory_order_release);
    requireFeature(smartgate::readLatestTargetProfileForPersistence(
                       pendingForSave, publishedForSave, appliedVersion, selected),
                   "Published profile must remain available after restore application");
    requireFeature(selected.count == newer.count,
                   "After restore application, save must select the latest learned profile");

    // Exercise the atomic snapshot mailbox with a writer and reader on
    // separate threads. Every successful read must contain one coherent
    // version; mixing fields from adjacent publications would fail these
    // deliberately correlated values.
    {
        smartgate::AtomicTargetProfileSnapshot concurrentMailbox;
        std::atomic<bool> writerFinished{false};
        std::thread writer([&] {
            for (std::uint64_t version = 1; version <= 50000; ++version) {
                smartgate::TargetProfile::Snapshot value;
                value.count = version;
                for (std::size_t i = 0; i < value.mean.size(); ++i) {
                    value.mean[i] = static_cast<double>(version);
                    value.m2[i] = static_cast<double>(version) * 2.0;
                }
                concurrentMailbox.publish(value);
            }
            writerFinished.store(true, std::memory_order_release);
        });

        std::uint64_t successfulReads = 0;
        bool snapshotsCoherent = true;
        do {
            smartgate::TargetProfile::Snapshot value;
            std::uint64_t version = 0;
            if (!concurrentMailbox.read(value, version, 64))
                continue;
            ++successfulReads;
            snapshotsCoherent = snapshotsCoherent
                && version != 0 && (version & 1U) == 0;
            for (std::size_t i = 0; i < value.mean.size(); ++i) {
                snapshotsCoherent = snapshotsCoherent
                    && value.mean[i] == static_cast<double>(value.count)
                    && value.m2[i] == static_cast<double>(value.count) * 2.0;
            }
        } while (!writerFinished.load(std::memory_order_acquire));

        writer.join();
        smartgate::TargetProfile::Snapshot finalValue;
        std::uint64_t finalVersion = 0;
        const bool finalReadable = concurrentMailbox.read(finalValue, finalVersion);
        requireFeature(snapshotsCoherent,
                       "Concurrent snapshot reads must not mix adjacent publications");
        requireFeature(finalReadable,
                       "Final concurrent snapshot must remain readable");
        requireFeature(finalValue.count == 50000 && successfulReads > 0,
                       "Concurrent mailbox must publish the writer's final snapshot");
    }

    profile.clear();
    requireFeature(!profile.isReady() && profile.frameCount() == 0,
                   "Clearing profile must reset training state");
}
