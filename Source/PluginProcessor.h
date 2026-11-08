#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "AdaptiveKeyboardDetector.h"
#include "AudioFeatures.h"
#include "GateEnvelope.h"
#include "TargetProfile.h"

#if defined(SMARTGATE_HAS_SILERO)
#include "SileroVadInference.h"
#include "VadFrameAssembler.h"
#include "VadFrameQueue.h"
#include "VadInferenceWorker.h"
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

class SmartGateAudioProcessor final : public juce::AudioProcessor {
    friend class SmartGateAudioProcessorEditor;
public:
    SmartGateAudioProcessor();

    void prepareToPlay(double sampleRate, int maximumExpectedSamplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Smart Gate"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    // The gate has no delay or generated output after its input stops; Hold and
    // Release only control gain while input audio is present.
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
#if defined(SMARTGATE_HAS_SILERO)
    using VadQueue = smartgate::VadFrameQueue<smartgate::SileroVadInference::frameSize, 8>;
    static float runSileroInference(const VadQueue::Frame& frame,
                                    std::uint64_t generation, std::uint64_t sourceSamplePosition,
                                    void* context) noexcept;
#endif

    juce::AudioProcessorValueTreeState parameters_;
    smartgate::GateEnvelope gateEnvelope_;
    smartgate::FeatureExtractor featureExtractor_;
    smartgate::StereoAnalysisFoldDown stereoAnalysisFoldDown_;
    smartgate::AdaptiveKeyboardDetector keyboardDetector_;
    smartgate::TargetProfile targetProfile_;
    smartgate::AtomicTargetProfileSnapshot publishedTargetProfile_;
    smartgate::AtomicTargetProfileSnapshot pendingTargetProfile_;
    // Audio-thread restore version is also observed by the non-real-time
    // state-save path to distinguish pending data from the live snapshot.
    std::atomic<std::uint64_t> lastAppliedTargetProfileVersion_{0};

    static constexpr std::size_t featureFrameSize_ = 256;
    std::array<float, featureFrameSize_> featureFrame_{};
    std::size_t featureFramePosition_ = 0;
    float targetConfidenceTarget_ = 1.0f;
    float targetConfidenceSmoothed_ = 1.0f;
    float targetConfidenceSmoothingCoefficient_ = 0.0f;
    float keyboardConfidence_ = 0.0f;
    smartgate::AdaptiveKeyboardSuppressor keyboardSuppressor_;
    bool wasLearningTarget_ = false;
    bool pendingLearningStart_ = false;
    // setStateInformation may run after prepareToPlay; synchronize the restored
    // Learn Target toggle on the audio thread without treating it as a user edge.
    std::atomic<bool> stateRestorePending_{false};
    std::atomic<bool> restoredLearningTarget_{false};
    // Lock-free UI telemetry snapshots written by the audio thread and polled by the editor.
    std::atomic<float> inputPeakDbForUi_{-100.0f};
    std::atomic<float> outputPeakDbForUi_{-100.0f};
    std::atomic<float> gateGainForUi_{0.0f};
    std::atomic<float> cpuLoadPercentForUi_{0.0f};
    std::atomic<float> cpuPeakPercentForUi_{0.0f};
    float cpuLoadAverage_ = 0.0f;
    float cpuPeakHold_ = 0.0f;
    float cpuPeakHoldSeconds_ = 0.0f;
    std::atomic<std::uint64_t> learnedFramesForUi_{0};
    std::atomic<bool> vadReadyForUi_{false};
    bool wasAiEnabled_ = true;
    bool wasKeyboardAnalysisEnabled_ = false;
#if defined(SMARTGATE_HAS_SILERO)
    // Declaration order is intentional: worker is destroyed before its model
    // context and queue, so its thread can never outlive either dependency.
    VadQueue vadQueue_;
    smartgate::VadFrameAssembler<VadQueue> vadAssembler_{vadQueue_};
    std::unique_ptr<smartgate::SileroVadInference> sileroInference_;
    smartgate::VadInferenceWorker<VadQueue> vadWorker_{vadQueue_};
    std::uint64_t vadGeneration_ = 1;
    float vadProbabilitySmoothed_ = 0.0f;
    float vadProbabilitySmoothingCoefficient_ = 0.0f;
    std::uint64_t audioSamplePosition_ = 0;
    std::uint64_t vadMaxAgeSamples_ = 0;
    bool vadAvailable_ = false;
#endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SmartGateAudioProcessor)
};
