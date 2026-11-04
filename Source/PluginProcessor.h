#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "AudioFeatures.h"
#include "GateEnvelope.h"
#include "TargetProfile.h"

#include <array>
#include <cstddef>

class SmartGateAudioProcessor final : public juce::AudioProcessor {
public:
    SmartGateAudioProcessor();

    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return false; }
    const juce::String getName() const override { return "Smart Gate"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destination) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void finishFeatureFrame() noexcept;

    juce::AudioProcessorValueTreeState parameters_;
    smartgate::GateEnvelope gateEnvelope_;
    smartgate::FeatureExtractor featureExtractor_;
    smartgate::TargetProfile targetProfile_;
    smartgate::AtomicTargetProfileSnapshot publishedTargetProfile_;
    smartgate::AtomicTargetProfileSnapshot pendingTargetProfile_;
    std::uint64_t lastAppliedTargetProfileVersion_ = 0;

    static constexpr std::size_t featureFrameSize_ = 256;
    std::array<float, featureFrameSize_> featureFrame_{};
    std::size_t featureFramePosition_ = 0;
    float targetConfidenceTarget_ = 1.0f;
    float targetConfidenceSmoothed_ = 1.0f;
    float targetConfidenceSmoothingCoefficient_ = 0.0f;
    bool wasLearningTarget_ = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SmartGateAudioProcessor)
};
