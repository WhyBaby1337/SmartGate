#include "PluginProcessor.h"

#include <cmath>
#include <cstdint>
#include <limits>

SmartGateAudioProcessor::SmartGateAudioProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "PARAMETERS", createParameterLayout()) {}

juce::AudioProcessorValueTreeState::ParameterLayout
SmartGateAudioProcessor::createParameterLayout() {
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"threshold", 1}, "Threshold",
        juce::NormalisableRange<float>{-80.0f, 0.0f, 0.1f}, -36.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"attack", 1}, "Attack",
        juce::NormalisableRange<float>{0.1f, 100.0f, 0.1f, 0.5f}, 5.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("ms")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"release", 1}, "Release",
        juce::NormalisableRange<float>{1.0f, 1000.0f, 1.0f, 0.5f}, 120.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("ms")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"hold", 1}, "Hold",
        juce::NormalisableRange<float>{0.0f, 500.0f, 1.0f}, 80.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("ms")));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"aiEnabled", 1}, "AI Engine", true));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"learnTarget", 1}, "Learn Target", false));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"targetAmount", 1}, "Target Influence",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.5f));
    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{"keyboardMode", 1}, "Keyboard Type",
        juce::StringArray{"Universal", "Mechanical", "Membrane"}, 0));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"keyboardReduction", 1}, "Keyboard Reduction",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.0f));
    return layout;
}

void SmartGateAudioProcessor::prepareToPlay(double sampleRate, int) {
    gateEnvelope_.prepare(sampleRate);
    featureExtractor_.prepare(sampleRate);
    keyboardDetector_.prepare(sampleRate);
    keyboardConfidence_ = 0.0f;
    featureFramePosition_ = 0;
    featureFrame_.fill(0.0f);
    targetConfidenceTarget_ = 1.0f;
    targetConfidenceSmoothed_ = 1.0f;
    targetConfidenceSmoothingCoefficient_ = static_cast<float>(
        std::exp(-1.0 / (0.030 * (std::isfinite(sampleRate) ? sampleRate : 44100.0))));
    wasLearningTarget_ = false;
    wasAiEnabled_ = *parameters_.getRawParameterValue("aiEnabled") >= 0.5f;
    publishedTargetProfile_.publish(targetProfile_.makeSnapshot());
}

void SmartGateAudioProcessor::releaseResources() {}

bool SmartGateAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();
    return input == output
        && (input == juce::AudioChannelSet::mono()
            || input == juce::AudioChannelSet::stereo());
}

void SmartGateAudioProcessor::finishFeatureFrame() noexcept {
    const auto features = featureExtractor_.process(featureFrame_.data(), featureFrame_.size());
    if (!features.isFinite())
        return;

    const bool learning = *parameters_.getRawParameterValue("learnTarget") >= 0.5f;
    // Ignore silence and heavily clipped frames during target acquisition.
    if (learning && features.logRms > -2.5f && features.logRms < -0.03f
        && targetProfile_.learn(features))
        publishedTargetProfile_.publish(targetProfile_.makeSnapshot());

    targetConfidenceTarget_ = targetProfile_.isReady()
        ? targetProfile_.confidence(features) : 1.0f;
}

void SmartGateAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;

    gateEnvelope_.setThresholdDb(*parameters_.getRawParameterValue("threshold"));
    gateEnvelope_.setTimes(*parameters_.getRawParameterValue("attack"),
                           *parameters_.getRawParameterValue("release"));
    gateEnvelope_.setHoldTime(*parameters_.getRawParameterValue("hold"));

    const bool aiEnabled = *parameters_.getRawParameterValue("aiEnabled") >= 0.5f;
    const bool learningRequested = *parameters_.getRawParameterValue("learnTarget") >= 0.5f;

    // AI OFF is a true bypass: no feature extraction or target inference is
    // performed. Re-entry starts with a clean frame to avoid stale samples.
    if (aiEnabled != wasAiEnabled_) {
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
        wasAiEnabled_ = aiEnabled;
    }

    if (learningRequested && !wasLearningTarget_)
        pendingLearningStart_ = true;
    if (!learningRequested)
        pendingLearningStart_ = false;

    // A requested Learn Target transition is remembered while AI is disabled.
    // Turning AI off pauses analysis without silently erasing an existing model.
    if (aiEnabled && pendingLearningStart_) {
        targetProfile_.clear();
        publishedTargetProfile_.publish(targetProfile_.makeSnapshot());
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
        pendingLearningStart_ = false;
    }
    wasLearningTarget_ = learningRequested;

    smartgate::TargetProfile::Snapshot restored;
    std::uint64_t restoredVersion = 0;
    if (pendingTargetProfile_.read(restored, restoredVersion)
        && restoredVersion != lastAppliedTargetProfileVersion_
        && targetProfile_.restore(restored)) {
        lastAppliedTargetProfileVersion_ = restoredVersion;
        publishedTargetProfile_.publish(restored);
        targetConfidenceTarget_ = targetConfidenceSmoothed_ = 1.0f;
    }

    const float targetAmount = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("targetAmount")->load(std::memory_order_relaxed));
    const float keyboardReduction = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("keyboardReduction")->load(std::memory_order_relaxed));
    const int keyboardMode = juce::jlimit(0, 2, static_cast<int>(std::lround(
        parameters_.getRawParameterValue("keyboardMode")->load(std::memory_order_relaxed))));
    keyboardDetector_.setMode(static_cast<smartgate::AdaptiveKeyboardDetector::Mode>(keyboardMode));

    const int channels = buffer.getNumChannels();
    const int samples = buffer.getNumSamples();

    // Peak-linked detector keeps the stereo image centered and stable. Target
    // features are extracted from a mono average in fixed-size frames.
    for (int sample = 0; sample < samples; ++sample) {
        float linkedPeak = 0.0f;
        float mono = 0.0f;
        for (int channel = 0; channel < channels; ++channel) {
            const float value = buffer.getSample(channel, sample);
            linkedPeak = juce::jmax(linkedPeak, std::abs(value));
            mono += value;
        }
        mono /= static_cast<float>(juce::jmax(1, channels));

        // The keyboard classifier is a DSP subsystem and keeps running even
        // when the optional AI/target-analysis engine is disabled.
        featureFrame_[featureFramePosition_++] = mono;
        if (featureFramePosition_ == featureFrameSize_) {
            keyboardConfidence_ = keyboardDetector_.processFrame(
                featureFrame_.data(), featureFrame_.size());
            if (aiEnabled)
                finishFeatureFrame();
            featureFramePosition_ = 0;
        }

        targetConfidenceSmoothed_ = targetConfidenceTarget_
            + targetConfidenceSmoothingCoefficient_
                * (targetConfidenceSmoothed_ - targetConfidenceTarget_);
        const float targetGain = aiEnabled
            ? (1.0f - targetAmount) + targetAmount * targetConfidenceSmoothed_
            : 1.0f;
        const float keyboardDetectorGain = 1.0f
            - keyboardReduction * keyboardConfidence_;
        const float gain = gateEnvelope_.process(
            linkedPeak * targetGain * keyboardDetectorGain);

        for (int channel = 0; channel < channels; ++channel)
            buffer.setSample(channel, sample, buffer.getSample(channel, sample) * gain);
    }
}

juce::AudioProcessorEditor* SmartGateAudioProcessor::createEditor() {
    return new juce::GenericAudioProcessorEditor(*this);
}

void SmartGateAudioProcessor::getStateInformation(juce::MemoryBlock& destination) {
    if (auto xml = parameters_.copyState().createXml()) {
        smartgate::TargetProfile::Snapshot snapshot;
        std::uint64_t version = 0;
        if (publishedTargetProfile_.read(snapshot, version)) {
            auto* node = xml->createNewChildElement("TARGET_PROFILE");
            node->setAttribute("count", static_cast<int>(snapshot.count));
            for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
                const auto index = juce::String(static_cast<int>(i));
                node->setAttribute("mean" + index, snapshot.mean[i]);
                node->setAttribute("m2_" + index, snapshot.m2[i]);
            }
        }
        copyXmlToBinary(*xml, destination);
    }
}

void SmartGateAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    if (auto xml = getXmlFromBinary(data, sizeInBytes)) {
        if (!xml->hasTagName(parameters_.state.getType()))
            return;
        parameters_.replaceState(juce::ValueTree::fromXml(*xml));

        if (auto* node = xml->getChildByName("TARGET_PROFILE")) {
            smartgate::TargetProfile::Snapshot snapshot;
            const auto count = node->getIntAttribute("count", -1);
            bool valid = count >= 0;
            if (valid) {
                snapshot.count = static_cast<std::uint64_t>(count);
                for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
                    const auto index = juce::String(static_cast<int>(i));
                    snapshot.mean[i] = node->getDoubleAttribute("mean" + index,
                        std::numeric_limits<double>::quiet_NaN());
                    snapshot.m2[i] = node->getDoubleAttribute("m2_" + index,
                        std::numeric_limits<double>::quiet_NaN());
                }
                smartgate::TargetProfile validator;
                valid = validator.restore(snapshot);
            }

            // Invalid or incomplete profile data must clear the previous
            // profile instead of leaving stale training from another session.
            pendingTargetProfile_.publish(valid
                ? snapshot : smartgate::TargetProfile::Snapshot{});
        } else {
            // Older sessions have no target profile; clear any profile left
            // from the currently loaded instance.
            pendingTargetProfile_.publish(smartgate::TargetProfile::Snapshot{});
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new SmartGateAudioProcessor();
}
