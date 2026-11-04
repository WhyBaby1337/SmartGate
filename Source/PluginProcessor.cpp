#include "PluginProcessor.h"

#include <cmath>

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
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"learnTarget", 1}, "Learn Target", false));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"targetAmount", 1}, "Target Influence",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.5f));
    return layout;
}

void SmartGateAudioProcessor::prepareToPlay(double sampleRate, int) {
    gateEnvelope_.prepare(sampleRate);
    featureExtractor_.prepare(sampleRate);
    featureFramePosition_ = 0;
    featureFrame_.fill(0.0f);
    targetConfidenceTarget_ = 1.0f;
    targetConfidenceSmoothed_ = 1.0f;
    targetConfidenceSmoothingCoefficient_ = static_cast<float>(
        std::exp(-1.0 / (0.030 * (std::isfinite(sampleRate) ? sampleRate : 44100.0))));
    wasLearningTarget_ = false;
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
    if (learning && features.logRms > -2.5f && features.logRms < -0.03f)
        targetProfile_.learn(features);

    targetConfidenceTarget_ = targetProfile_.isReady()
        ? targetProfile_.confidence(features) : 1.0f;
}

void SmartGateAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;

    gateEnvelope_.setThresholdDb(*parameters_.getRawParameterValue("threshold"));
    gateEnvelope_.setTimes(*parameters_.getRawParameterValue("attack"),
                           *parameters_.getRawParameterValue("release"));

    const bool learningTarget = *parameters_.getRawParameterValue("learnTarget") >= 0.5f;
    if (learningTarget && !wasLearningTarget_) {
        // A new Learn Target session starts with a clean profile. All profile
        // access stays on the audio thread, so no locks or cross-thread copies
        // are needed in processBlock.
        targetProfile_.clear();
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
    }
    wasLearningTarget_ = learningTarget;

    const float targetAmount = juce::jlimit(
        0.0f, 1.0f, *parameters_.getRawParameterValue("targetAmount"));
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

        featureFrame_[featureFramePosition_++] = mono;
        if (featureFramePosition_ == featureFrameSize_) {
            finishFeatureFrame();
            featureFramePosition_ = 0;
        }

        targetConfidenceSmoothed_ = targetConfidenceTarget_
            + targetConfidenceSmoothingCoefficient_
                * (targetConfidenceSmoothed_ - targetConfidenceTarget_);
        const float targetGain = (1.0f - targetAmount)
            + targetAmount * targetConfidenceSmoothed_;
        const float gain = gateEnvelope_.process(linkedPeak * targetGain);

        for (int channel = 0; channel < channels; ++channel)
            buffer.setSample(channel, sample, buffer.getSample(channel, sample) * gain);
    }
}

juce::AudioProcessorEditor* SmartGateAudioProcessor::createEditor() {
    return nullptr;
}

void SmartGateAudioProcessor::getStateInformation(juce::MemoryBlock& destination) {
    if (auto xml = parameters_.copyState().createXml())
        copyXmlToBinary(*xml, destination);
}

void SmartGateAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    if (auto xml = getXmlFromBinary(data, sizeInBytes)) {
        if (xml->hasTagName(parameters_.state.getType()))
            parameters_.replaceState(juce::ValueTree::fromXml(*xml));
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new SmartGateAudioProcessor();
}
