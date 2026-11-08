#include "PluginProcessor.h"
#include "MouthClickSuppressor.h"
#include "PluginEditor.h"

#if defined(SMARTGATE_HAS_SILERO)
#include <BinaryData.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"hysteresis", 1}, "Hysteresis",
        juce::NormalisableRange<float>{0.0f, 12.0f, 0.1f}, 3.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"adaptiveHold", 1}, "Adaptive Hold",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.5f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"range", 1}, "Range",
        juce::NormalisableRange<float>{-80.0f, 0.0f, 0.1f}, -80.0f,
        juce::AudioParameterFloatAttributes{}.withLabel("dB")));
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
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"keyboardSuppressionEnabled", 1}, "Keyboard and Mouse Suppression", true));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"keyboardReduction", 1}, "Keyboard Reduction",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.0f));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"mouthClickSuppressionEnabled", 1}, "Mouth Click Suppression", true));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"mouthNoiseReduction", 1}, "Mouth Click Reduction",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.25f));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"voiceLeveling", 1}, "Voice Leveling",
        juce::NormalisableRange<float>{0.0f, 1.0f, 0.01f}, 0.0f));
    return layout;
}

#if defined(SMARTGATE_HAS_SILERO)
float SmartGateAudioProcessor::runSileroInference(const VadQueue::Frame& frame,
                                                   std::uint64_t generation,
                                                   std::uint64_t sourceSamplePosition,
                                                   void* context) noexcept {
    auto* inference = static_cast<smartgate::SileroVadInference*>(context);
    return inference != nullptr
        ? inference->infer(frame, generation, sourceSamplePosition) : 0.0f;
}
#endif

void SmartGateAudioProcessor::prepareToPlay(double sampleRate, int) {
    vadReadyForUi_.store(false, std::memory_order_relaxed);
    cpuLoadAverage_ = 0.0f;
    cpuPeakHold_ = 0.0f;
    cpuPeakHoldSeconds_ = 0.0f;
    cpuLoadPercentForUi_.store(0.0f, std::memory_order_relaxed);
    cpuPeakPercentForUi_.store(0.0f, std::memory_order_relaxed);
    inputPeakDbForUi_.store(-100.0f, std::memory_order_relaxed);
    outputPeakDbForUi_.store(-100.0f, std::memory_order_relaxed);
    gateGainForUi_.store(0.0f, std::memory_order_relaxed);
    gateDetectorLevelForUi_.store(0.0f, std::memory_order_relaxed);
    gateOpeningThresholdForUi_.store(0.0f, std::memory_order_relaxed);
    // Use one validated host rate for every time-based subsystem. A malformed
    // or extreme host value must not make smoothing unstable or desynchronize
    // the gate, feature extractors, keyboard detector, and VAD resampler.
    const double safeSampleRate = std::isfinite(sampleRate)
        ? std::clamp(sampleRate, 8000.0, 768000.0) : 44100.0;
#if defined(SMARTGATE_HAS_SILERO)
    audioSamplePosition_ = 0;
    vadMaxAgeSamples_ = static_cast<std::uint64_t>(safeSampleRate * 0.250);
#endif
#if defined(SMARTGATE_HAS_SILERO)
    // prepareToPlay is a non-processing lifecycle boundary. Model/session
    // creation and thread startup happen here, never in processBlock().
    vadWorker_.stop();
    sileroInference_.reset();
    vadAvailable_ = false;
    vadGeneration_ = 1;
    vadProbabilitySmoothed_ = 0.0f;
    vadAssembler_.reset();
    if (safeSampleRate >= 16000.0
        && safeSampleRate <= smartgate::VadResampler::maximumInputRate) {
        try {
            sileroInference_ = std::make_unique<smartgate::SileroVadInference>(
                BinaryData::silero_vad_onnx,
                static_cast<std::size_t>(BinaryData::silero_vad_onnxSize),
                safeSampleRate);
            if (vadAssembler_.prepare(safeSampleRate, 16000.0, vadGeneration_)
                && vadWorker_.start(&SmartGateAudioProcessor::runSileroInference,
                                    sileroInference_.get(), vadGeneration_)) {
                vadAvailable_ = true;
                vadReadyForUi_.store(true, std::memory_order_relaxed);
            } else {
                vadReadyForUi_.store(false, std::memory_order_relaxed);
                sileroInference_.reset();
            }
        } catch (...) {
            // Optional AI failure is fail-open: the conventional gate remains usable.
            vadWorker_.stop();
            sileroInference_.reset();
            vadAvailable_ = false;
            vadReadyForUi_.store(false, std::memory_order_relaxed);
        }
    }
#endif
    lookaheadDelay_.prepare(safeSampleRate, lookaheadMilliseconds_);
    voiceLeveler_.prepare(safeSampleRate);
    setLatencySamples(static_cast<int>(lookaheadDelay_.latencySamples()));
    gateEnvelope_.prepare(safeSampleRate);
    featureExtractor_.prepare(safeSampleRate);
    stereoAnalysisFoldDown_.prepare(safeSampleRate);
    keyboardDetector_.prepare(safeSampleRate);
    mouthClickSuppressorLeft_.prepare(safeSampleRate);
    mouthClickSuppressorRight_.prepare(safeSampleRate);
    keyboardConfidence_ = 0.0f;
    keyboardSuppressorLeft_.prepare(safeSampleRate);
    keyboardSuppressorRight_.prepare(safeSampleRate);
    featureFramePosition_ = 0;
    featureFrame_.fill(0.0f);
    targetConfidenceTarget_ = 1.0f;
    targetConfidenceSmoothed_ = 1.0f;
    targetConfidenceSmoothingCoefficient_ = static_cast<float>(
        std::exp(-1.0 / (0.030 * safeSampleRate)));
#if defined(SMARTGATE_HAS_SILERO)
    vadProbabilitySmoothingCoefficient_ = targetConfidenceSmoothingCoefficient_;
#endif
    // Match the restored APVTS toggle at the lifecycle boundary. If a session
    // was saved with Learn Target enabled, its first audio block must continue
    // that session rather than treating the restored value as a fresh request
    // and clearing the target profile that was just restored.
    wasLearningTarget_ = *parameters_.getRawParameterValue("learnTarget") >= 0.5f;
    pendingLearningStart_ = false;
    stateRestorePending_.store(false, std::memory_order_relaxed);
    wasAiEnabled_ = *parameters_.getRawParameterValue("aiEnabled") >= 0.5f;
    const auto initialProfile = targetProfile_.makeSnapshot();
    publishedTargetProfile_.publish(initialProfile);
    learnedFramesForUi_.store(initialProfile.count, std::memory_order_relaxed);
}

void SmartGateAudioProcessor::requestTargetProfileReset() noexcept {
    targetProfileResetRequested_.store(true, std::memory_order_release);
}

void SmartGateAudioProcessor::releaseResources() {
#if defined(SMARTGATE_HAS_SILERO)
    // Host calls release after processing has stopped; joining here is safe.
    vadWorker_.stop();
    sileroInference_.reset();
    vadAvailable_ = false;
    vadReadyForUi_.store(false, std::memory_order_relaxed);
    vadProbabilitySmoothed_ = 0.0f;
#endif
}

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
    bool vadAllowsLearning = true;
#if defined(SMARTGATE_HAS_SILERO)
    // Stale/invalid/low-confidence Silero results pause target acquisition.
    vadAllowsLearning = smartgate::TargetProfile::isSpeechQualifiedForLearning(
        vadAvailable_, vadProbabilitySmoothed_);
#endif
    const bool keyboardAllowsLearning =
        smartgate::TargetProfile::isKeyboardQualifiedForLearning(keyboardConfidence_);
    if (learning && vadAllowsLearning && keyboardAllowsLearning
        && smartgate::TargetProfile::isValidLearningFrame(features)
        && targetProfile_.learn(features)) {
        const auto snapshot = targetProfile_.makeSnapshot();
        publishedTargetProfile_.publish(snapshot);
        learnedFramesForUi_.store(snapshot.count, std::memory_order_relaxed);
    }

    targetConfidenceTarget_ = targetProfile_.isReady()
        ? targetProfile_.confidence(features) : 1.0f;
}

void SmartGateAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                            juce::MidiBuffer&) {
    const auto processStartTicks = juce::Time::getHighResolutionTicks();
    juce::ScopedNoDenormals noDenormals;

    gateEnvelope_.setThresholdDb(*parameters_.getRawParameterValue("threshold"));
    gateEnvelope_.setTimes(*parameters_.getRawParameterValue("attack"),
                           *parameters_.getRawParameterValue("release"));
    // The detector hears the live input while output audio is delayed. Add the
    // same lookahead interval to Hold so closing does not cut the delayed tail.
    gateEnvelope_.setHoldTime(*parameters_.getRawParameterValue("hold")
                              + static_cast<float>(lookaheadMilliseconds_));
    const float baseAdaptiveHold = juce::jlimit(0.0f, 1.0f,
        parameters_.getRawParameterValue("adaptiveHold")->load(std::memory_order_relaxed));
    gateEnvelope_.setHysteresisDb(*parameters_.getRawParameterValue("hysteresis"));
    gateEnvelope_.setRangeDb(*parameters_.getRawParameterValue("range"));

    const bool aiEnabled = *parameters_.getRawParameterValue("aiEnabled") >= 0.5f;
    const bool learningRequested = *parameters_.getRawParameterValue("learnTarget") >= 0.5f;

    // AI OFF disables target-profile analysis and Silero inference. The
    // independent keyboard classifier may still run when Keyboard Reduction
    // is enabled. Re-entry starts with a clean frame to avoid stale samples.
    if (aiEnabled != wasAiEnabled_) {
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
        wasAiEnabled_ = aiEnabled;
#if defined(SMARTGATE_HAS_SILERO)
        if (vadAvailable_) {
            ++vadGeneration_;
            if (vadGeneration_ == 0)
                vadGeneration_ = 1;
            vadWorker_.setGeneration(vadGeneration_);
            vadAssembler_.setGeneration(vadGeneration_);
            vadProbabilitySmoothed_ = 0.0f;
        }
#endif
    }

    // A host may restore a session while the processor is already prepared.
    // Synchronize the restored toggle before edge detection so loading a session
    // with Learn Target enabled does not masquerade as a fresh user request.
    if (stateRestorePending_.exchange(false, std::memory_order_acquire)) {
        // Compare the live parameter against the value that was actually
        // restored. A host/user change arriving in the same block must remain
        // a real edge (and an explicit Learn Target request must win).
        wasLearningTarget_ = restoredLearningTarget_.load(std::memory_order_relaxed);
        pendingLearningStart_ = false;
    }

    if (learningRequested && !wasLearningTarget_)
        pendingLearningStart_ = true;
    if (!learningRequested)
        pendingLearningStart_ = false;

    // Apply a pending session restore before resuming Learn Target.
    // Toggling Learn Target is pause/resume: preserve the accumulated profile
    // and only discard the incomplete feature frame at the transition.
    smartgate::TargetProfile::Snapshot restored;
    std::uint64_t restoredVersion = 0;
    if (pendingTargetProfile_.read(restored, restoredVersion)
        && restoredVersion != lastAppliedTargetProfileVersion_.load(std::memory_order_relaxed)
        && targetProfile_.restore(restored)) {
        publishedTargetProfile_.publish(restored);
        learnedFramesForUi_.store(restored.count, std::memory_order_relaxed);
        lastAppliedTargetProfileVersion_.store(restoredVersion, std::memory_order_release);
        targetConfidenceTarget_ = targetConfidenceSmoothed_ = 1.0f;
    }

    // An explicit UI reset is the only operation that discards the learned
    // profile. Apply it on the audio thread to keep TargetProfile single-writer.
    if (targetProfileResetRequested_.exchange(false, std::memory_order_acq_rel)) {
        targetProfile_.clear();
        const auto emptyProfile = targetProfile_.makeSnapshot();
        publishedTargetProfile_.publish(emptyProfile);
        learnedFramesForUi_.store(emptyProfile.count, std::memory_order_relaxed);
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
    }

    // A requested Learn Target transition is remembered while AI is disabled.
    // Learning is resumable: toggling the control never clears accumulated data.
    if (aiEnabled && pendingLearningStart_) {
        // Start/resume collection without discarding a previously learned
        // target. The next complete feature frame belongs to this session.
        featureExtractor_.reset();
        featureFramePosition_ = 0;
        targetConfidenceTarget_ = 1.0f;
        targetConfidenceSmoothed_ = 1.0f;
        pendingLearningStart_ = false;
    }
    wasLearningTarget_ = learningRequested;

    const float targetAmount = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("targetAmount")->load(std::memory_order_relaxed));
    const float keyboardReduction = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("keyboardReduction")->load(std::memory_order_relaxed));
    const bool keyboardSuppressionEnabled =
        parameters_.getRawParameterValue("keyboardSuppressionEnabled")->load(std::memory_order_relaxed) >= 0.5f;
    const float effectiveKeyboardReduction = keyboardSuppressionEnabled
        ? keyboardReduction : 0.0f;
    const float mouthNoiseReduction = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("mouthNoiseReduction")->load(std::memory_order_relaxed));
    const bool mouthClickSuppressionEnabled =
        parameters_.getRawParameterValue("mouthClickSuppressionEnabled")->load(std::memory_order_relaxed) >= 0.5f;
    const float effectiveMouthNoiseReduction = mouthClickSuppressionEnabled
        ? mouthNoiseReduction : 0.0f;
    const float voiceLeveling = juce::jlimit(
        0.0f, 1.0f, parameters_.getRawParameterValue("voiceLeveling")->load(std::memory_order_relaxed));
    const int keyboardMode = juce::jlimit(0, 2, static_cast<int>(std::lround(
        parameters_.getRawParameterValue("keyboardMode")->load(std::memory_order_relaxed))));
    // Run the keyboard transient detector either for audible reduction or
    // while learning a target. Learning-time analysis prevents keyboard clicks
    // from contaminating the profile even when the user leaves reduction at 0.
    const bool keyboardReductionEnabled = keyboardSuppressionEnabled && keyboardReduction > 0.0f;
    const bool keyboardAnalysisEnabled = keyboardReductionEnabled
        || (aiEnabled && learningRequested);
    if (keyboardAnalysisEnabled) {
        if (!wasKeyboardAnalysisEnabled_)
            keyboardDetector_.reset();
        keyboardDetector_.setMode(static_cast<smartgate::AdaptiveKeyboardDetector::Mode>(keyboardMode));
    } else {
        if (wasKeyboardAnalysisEnabled_)
            keyboardDetector_.reset();
        keyboardConfidence_ = 0.0f;
    }
    wasKeyboardAnalysisEnabled_ = keyboardAnalysisEnabled;

    const int samples = buffer.getNumSamples();
#if defined(SMARTGATE_HAS_SILERO)
    // A failed ONNX evaluation must not silently masquerade as ordinary
    // silence forever. Disable only the optional VAD path for this playback
    // session; the conventional gate and feature-only target learning remain
    // available. This check is an atomic read on the audio thread.
    if (vadAvailable_ && sileroInference_ != nullptr
        && sileroInference_->inferenceFailureCount() != 0) {
        vadAvailable_ = false;
        vadReadyForUi_.store(false, std::memory_order_relaxed);
        ++vadGeneration_;
        if (vadGeneration_ == 0)
            vadGeneration_ = 1;
        vadWorker_.setGeneration(vadGeneration_);
        vadAssembler_.setGeneration(vadGeneration_);
        vadWorker_.requestStop();
        vadProbabilitySmoothed_ = 0.0f;
    }
#endif

#if defined(SMARTGATE_HAS_SILERO)
    // Validate freshness through the end of this host block, not just at its
    // first sample. A very large offline/render block must not reuse a VAD
    // result that becomes stale while this block is being processed.
    const auto blockSampleCount = samples > 0
        ? static_cast<std::uint64_t>(samples) : std::uint64_t{0};
    const auto blockEndSamplePosition =
        blockSampleCount > std::numeric_limits<std::uint64_t>::max() - audioSamplePosition_
            ? std::numeric_limits<std::uint64_t>::max()
            : audioSamplePosition_ + blockSampleCount;
    const float vadProbability = aiEnabled && vadAvailable_
        ? vadWorker_.latestProbability(vadGeneration_, blockEndSamplePosition,
                                       vadMaxAgeSamples_) : 0.0f;
#endif
    const float thresholdLinear = std::pow(10.0f,
        *parameters_.getRawParameterValue("threshold") / 20.0f);

    const bool analyzeTarget = aiEnabled;
    const bool analyzeKeyboard = keyboardAnalysisEnabled;
    const bool analyzeFrames = analyzeTarget || analyzeKeyboard;

    // Do not carry a partially collected frame across a period in which both
    // analyzers are inactive. The next activation must start with fresh audio.
    if (!analyzeFrames)
        featureFramePosition_ = 0;

    const int channels = buffer.getNumChannels();
    if (channels <= 0 || samples <= 0)
        return;

    // This processor only advertises mono and stereo layouts. Cache raw
    // channel pointers once per block to avoid repeated AudioBuffer accessor
    // calls and bounds checks in the per-sample real-time loop.
    float* const left = buffer.getWritePointer(0);
    float* const right = channels > 1 ? buffer.getWritePointer(1) : nullptr;
    float inputPeak = 0.0f;
    float outputPeak = 0.0f;

    // Peak-linked detection keeps the stereo image centered and stable.
    // Analysis selects the stronger channel instead of averaging L/R, which
    // would cancel anti-phase content before VAD and feature extraction.
    // When both analyzers are disabled, skip the fold-down follower entirely
    // and clear its history so re-enabling analysis starts from fresh audio.
    if (!analyzeFrames || right == nullptr)
        stereoAnalysisFoldDown_.reset();

    for (int sample = 0; sample < samples; ++sample) {
        // A malformed upstream sample must not poison the detector or leak
        // NaN/Inf into the host output. Sanitization is allocation- and lock-free.
        const float leftSample = smartgate::finiteSampleOrZero(left[sample]);
        const float rightSample = right != nullptr
            ? smartgate::finiteSampleOrZero(right[sample]) : leftSample;
        const float linkedPeak = juce::jmax(std::abs(leftSample), std::abs(rightSample));
        float delayedLeft = 0.0f;
        float delayedRight = 0.0f;
        lookaheadDelay_.process(leftSample, rightSample, delayedLeft, delayedRight);
        inputPeak = juce::jmax(inputPeak, linkedPeak);
        const float mono = analyzeFrames && right != nullptr
            ? stereoAnalysisFoldDown_.process(leftSample, rightSample)
            : leftSample;

#if defined(SMARTGATE_HAS_SILERO)
        if (aiEnabled && vadAvailable_)
            (void) vadAssembler_.processSample(mono, audioSamplePosition_);
        ++audioSamplePosition_;
        vadProbabilitySmoothed_ = vadProbability
            + vadProbabilitySmoothingCoefficient_
                * (vadProbabilitySmoothed_ - vadProbability);
#endif

        // Share one low-rate mono frame between the optional target analyzer
        // and keyboard classifier. If both are inactive, skip frame assembly
        // and all classifier work rather than burning CPU on unused features.
        if (analyzeFrames) {
            featureFrame_[featureFramePosition_++] = mono;
            if (featureFramePosition_ == featureFrameSize_) {
                if (analyzeKeyboard) {
                    keyboardConfidence_ = keyboardDetector_.processFrame(
                        featureFrame_.data(), featureFrame_.size());
                }
                if (analyzeTarget)
                    finishFeatureFrame();
                featureFramePosition_ = 0;
            }
        }

        targetConfidenceSmoothed_ = targetConfidenceTarget_
            + targetConfidenceSmoothingCoefficient_
                * (targetConfidenceSmoothed_ - targetConfidenceTarget_);

        // A spectral target profile is not a phoneme-complete speech model.
        // A quiet /s/, /sh/ or /ch/ can legitimately differ from the learned
        // vowel-heavy profile, so Target Influence must never attenuate audio.
        // Use a positive-only match assist to extend adaptive phrase hold;
        // Silero remains the independent speech-probability assist for opening.
        const float targetHoldAssist = aiEnabled && targetProfile_.isReady()
            ? targetAmount * juce::jlimit(0.0f, 1.0f, targetConfidenceSmoothed_)
            : 0.0f;
        const float effectiveAdaptiveHold = baseAdaptiveHold
            + (1.0f - baseAdaptiveHold) * targetHoldAssist;
        gateEnvelope_.setAdaptiveHoldAmount(effectiveAdaptiveHold);
        // Silero is an assist to the conventional level detector, not a hard
        // gate override. A confident speech result gently lifts the detector
        // above threshold; silence/no result leaves classic gate behavior intact.
        float vadAssist = 0.0f;
#if defined(SMARTGATE_HAS_SILERO)
        vadAssist = aiEnabled && vadAvailable_
            ? juce::jlimit(0.0f, 1.0f, (vadProbabilitySmoothed_ - 0.35f) / 0.40f)
            : 0.0f;
#endif
        const float vadAssistedPeak = vadAssist > 0.0f
            ? juce::jmax(linkedPeak, thresholdLinear * (0.75f + 0.75f * vadAssist))
            : linkedPeak;
        // Target matching must never sit in the gate detector path: quiet
        // consonants can be acoustically dissimilar to the learned vowel-heavy
        // profile and would otherwise fail to open the gate. Apply its smooth
        // confidence gain only after the gate decision. Silero remains the
        // independent speech assist for opening/holding the gate.
        const float gateGain = gateEnvelope_.process(vadAssistedPeak);
        // Keyboard/mouse clicks are attenuated locally in the high band,
        // rather than ducking the entire voice signal for several milliseconds.
        const float keyboardProcessedLeft = keyboardSuppressorLeft_.process(
            delayedLeft, effectiveKeyboardReduction);
        const float keyboardProcessedRight = right != nullptr
            ? keyboardSuppressorRight_.process(delayedRight, effectiveKeyboardReduction)
            : keyboardProcessedLeft;
        const float outputGain = gateGain;

        // Mouth clicks use a separate conservative high-band transient stage.
        const float mouthProcessedLeft = mouthClickSuppressorLeft_.process(
            keyboardProcessedLeft, effectiveMouthNoiseReduction);
        const float mouthProcessedRight = right != nullptr
            ? mouthClickSuppressorRight_.process(keyboardProcessedRight, effectiveMouthNoiseReduction)
            : mouthProcessedLeft;
        const float levelerGain = voiceLeveler_.processStereo(
            mouthProcessedLeft * outputGain,
            right != nullptr ? mouthProcessedRight * outputGain
                             : mouthProcessedLeft * outputGain,
            voiceLeveling);
        left[sample] = mouthProcessedLeft * outputGain * levelerGain;
        if (right != nullptr)
            right[sample] = mouthProcessedRight * outputGain * levelerGain;
        outputPeak = juce::jmax(outputPeak,
            juce::jmax(std::abs(left[sample]), right != nullptr ? std::abs(right[sample]) : 0.0f));
    }

    // Measure this plugin's processing time against the real-time budget for
    // the current block. The timer is read only twice per block; all smoothing
    // and peak-hold state is audio-thread local, with atomic UI snapshots.
    const auto processEndTicks = juce::Time::getHighResolutionTicks();
    const auto ticksPerSecond = juce::Time::getHighResolutionTicksPerSecond();
    const double hostSampleRate = getSampleRate();
    if (processEndTicks > processStartTicks && ticksPerSecond > 0
        && std::isfinite(hostSampleRate) && hostSampleRate > 0.0) {
        const double blockSeconds = static_cast<double>(samples) / hostSampleRate;
        const double elapsedSeconds = static_cast<double>(processEndTicks - processStartTicks)
            / static_cast<double>(ticksPerSecond);
        if (blockSeconds > 0.0 && std::isfinite(elapsedSeconds)) {
            const float loadPercent = static_cast<float>(juce::jlimit(
                0.0, 1000.0, 100.0 * elapsedSeconds / blockSeconds));
            // A 250 ms time constant keeps the display readable without
            // hiding meaningful changes when the host buffer size changes.
            const float smoothing = static_cast<float>(
                juce::jlimit(0.0, 1.0, blockSeconds / 0.250));
            cpuLoadAverage_ += smoothing * (loadPercent - cpuLoadAverage_);
            if (loadPercent >= cpuPeakHold_) {
                cpuPeakHold_ = loadPercent;
                cpuPeakHoldSeconds_ = 1.0f;
            } else {
                cpuPeakHoldSeconds_ -= static_cast<float>(blockSeconds);
                if (cpuPeakHoldSeconds_ <= 0.0f) {
                    cpuPeakHold_ = loadPercent;
                    cpuPeakHoldSeconds_ = 0.0f;
                }
            }
            cpuLoadPercentForUi_.store(cpuLoadAverage_, std::memory_order_relaxed);
            cpuPeakPercentForUi_.store(cpuPeakHold_, std::memory_order_relaxed);
        }
    }

    const auto toDb = [](float peak) noexcept {
        return juce::jlimit(-100.0f, 0.0f, 20.0f * std::log10(juce::jmax(peak, 1.0e-5f)));
    };
    inputPeakDbForUi_.store(toDb(inputPeak), std::memory_order_relaxed);
    outputPeakDbForUi_.store(toDb(outputPeak), std::memory_order_relaxed);
    gateGainForUi_.store(gateEnvelope_.getGain(), std::memory_order_relaxed);
    gateDetectorLevelForUi_.store(gateEnvelope_.getDetectorLevel(), std::memory_order_relaxed);
    gateOpeningThresholdForUi_.store(gateEnvelope_.getOpeningThreshold(), std::memory_order_relaxed);
}

juce::AudioProcessorEditor* SmartGateAudioProcessor::createEditor() {
    return new SmartGateAudioProcessorEditor(*this);
}

void SmartGateAudioProcessor::getStateInformation(juce::MemoryBlock& destination) {
    auto state = parameters_.copyState();

    // TARGET_PROFILE is plugin-owned metadata, not an APVTS parameter child.
    // Strip any copy left by older sessions before writing exactly one profile.
    for (int i = state.getNumChildren(); --i >= 0;) {
        if (state.getChild(i).getType().toString() == "TARGET_PROFILE")
            state.removeChild(i, nullptr);
    }

    if (auto xml = state.createXml()) {
        smartgate::TargetProfile::Snapshot snapshot;
        // This method is called by the host's non-real-time state-save path.
        // Wait out a concurrent audio-thread publish rather than silently
        // serializing parameters without the trained target profile.
        // A host may request a save after setStateInformation() but before
        // the next audio callback applies the pending profile. Serialize that
        // restored snapshot rather than the previous live profile (or empty
        // pre-prepare mailbox), so an immediate save/load round trip is lossless.
        const bool hasSnapshot = smartgate::readLatestTargetProfileForPersistence(
            pendingTargetProfile_, publishedTargetProfile_,
            lastAppliedTargetProfileVersion_, snapshot);

        if (hasSnapshot) {
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

        auto restoredState = juce::ValueTree::fromXml(*xml);
        if (!restoredState.isValid())
            return;

        // Keep plugin metadata out of APVTS. Otherwise each subsequent save
        // can serialize the previous TARGET_PROFILE and append another copy.
        for (int i = restoredState.getNumChildren(); --i >= 0;) {
            if (restoredState.getChild(i).getType().toString() == "TARGET_PROFILE")
                restoredState.removeChild(i, nullptr);
        }

        // APVTS normalisation clamps ordinary out-of-range values, but a NaN
        // can pass through comparison-based clamping unchanged. Reject malformed
        // or non-finite values for known parameters so APVTS recreates their
        // canonical nodes with the declared defaults.
        for (int i = restoredState.getNumChildren(); --i >= 0;) {
            const auto child = restoredState.getChild(i);
            if (!child.hasType("PARAM"))
                continue;

            const auto id = child.getProperty("id").toString();
            if (parameters_.getParameter(id) == nullptr)
                continue;

            const auto valueText = child.getProperty("value").toString();
            const auto* valueStart = valueText.toRawUTF8();
            char* valueEnd = nullptr;
            const auto parsedValue = std::strtod(valueStart, &valueEnd);
            const bool validValue = valueEnd != valueStart
                && valueEnd != nullptr && *valueEnd == '\0'
                && std::isfinite(parsedValue);
            if (!validValue)
                restoredState.removeChild(i, nullptr);
        }

        // APVTS binds parameter nodes by ID while walking the tree. If a
        // corrupted/hand-edited session contains duplicate PARAM IDs, the
        // last node silently wins. Remove every copy of each duplicated known
        // parameter so that parameter falls back to its declared default;
        // unrelated valid parameters and unknown future IDs remain intact.
        juce::StringArray duplicateParameterIds;
        for (int i = 0; i < restoredState.getNumChildren(); ++i) {
            const auto child = restoredState.getChild(i);
            if (!child.hasType("PARAM"))
                continue;

            const auto id = child.getProperty("id").toString();
            if (parameters_.getParameter(id) == nullptr
                || duplicateParameterIds.contains(id))
                continue;

            int occurrences = 0;
            for (int j = 0; j < restoredState.getNumChildren(); ++j) {
                const auto candidate = restoredState.getChild(j);
                if (candidate.hasType("PARAM")
                    && candidate.getProperty("id").toString() == id)
                    ++occurrences;
            }
            if (occurrences > 1)
                duplicateParameterIds.add(id);
        }

        if (!duplicateParameterIds.isEmpty()) {
            for (int i = restoredState.getNumChildren(); --i >= 0;) {
                const auto child = restoredState.getChild(i);
                if (child.hasType("PARAM")
                    && duplicateParameterIds.contains(
                        child.getProperty("id").toString()))
                    restoredState.removeChild(i, nullptr);
            }
        }

        parameters_.replaceState(restoredState);
        restoredLearningTarget_.store(
            *parameters_.getRawParameterValue("learnTarget") >= 0.5f,
            std::memory_order_relaxed);
        stateRestorePending_.store(true, std::memory_order_release);

        juce::XmlElement* profileNode = nullptr;
        int profileNodeCount = 0;
        for (auto* child = xml->getFirstChildElement(); child != nullptr;
             child = child->getNextElement()) {
            if (child->hasTagName("TARGET_PROFILE")) {
                profileNode = child;
                ++profileNodeCount;
            }
        }

        // A profile is a single versioned snapshot. Multiple copies are
        // ambiguous (for example, a stale and a newly trained profile mixed
        // by an older host/save path), so reject the metadata rather than
        // silently choosing whichever XML child happens to appear first.
        if (profileNodeCount == 1) {
            smartgate::TargetProfile::Snapshot snapshot;
            const auto count = profileNode->getIntAttribute("count", -1);
            bool valid = count >= 0;
            if (valid) {
                snapshot.count = static_cast<std::uint64_t>(count);
                for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
                    const auto index = juce::String(static_cast<int>(i));
                    snapshot.mean[i] = profileNode->getDoubleAttribute("mean" + index,
                        std::numeric_limits<double>::quiet_NaN());
                    snapshot.m2[i] = profileNode->getDoubleAttribute("m2_" + index,
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
            // Older sessions have no profile; duplicate profiles are
            // ambiguous. Both cases clear any profile left by another session.
            pendingTargetProfile_.publish(smartgate::TargetProfile::Snapshot{});
        }
    }
}

bool SmartGateAudioProcessor::exportTargetProfile(const juce::File& file) const {
    if (file == juce::File{} || !file.getParentDirectory().isDirectory())
        return false;

    smartgate::TargetProfile::Snapshot snapshot;
    if (!smartgate::readLatestTargetProfileForPersistence(
            pendingTargetProfile_, publishedTargetProfile_,
            lastAppliedTargetProfileVersion_, snapshot)
        || snapshot.count == 0)
        return false;

    smartgate::TargetProfile validator;
    if (!validator.restore(snapshot))
        return false;

    juce::XmlElement xml("SMARTGATE_TARGET_PROFILE");
    xml.setAttribute("version", 1);
    xml.setAttribute("count", juce::String(static_cast<juce::int64>(snapshot.count)));
    for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
        const auto index = juce::String(static_cast<int>(i));
        xml.setAttribute("mean" + index, snapshot.mean[i]);
        xml.setAttribute("m2_" + index, snapshot.m2[i]);
    }
    return file.replaceWithText(xml.toString());
}

bool SmartGateAudioProcessor::importTargetProfile(const juce::File& file) {
    if (!file.existsAsFile() || file.getSize() <= 0
        || file.getSize() > 64 * 1024)
        return false;

    auto xml = juce::XmlDocument::parse(file);
    if (xml == nullptr || !xml->hasTagName("SMARTGATE_TARGET_PROFILE")
        || xml->getIntAttribute("version", -1) != 1)
        return false;

    // Version 1 has exactly two metadata attributes plus five mean/M2 pairs.
    // Reject unknown fields and nested content rather than partially importing
    // a file produced by an incompatible or malformed format.
    if (xml->getNumAttributes() != 12 || xml->getNumChildElements() != 0)
        return false;

    const auto count = xml->getStringAttribute("count").getLargeIntValue();
    if (count <= 0)
        return false;

    smartgate::TargetProfile::Snapshot snapshot;
    snapshot.count = static_cast<std::uint64_t>(count);
    for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
        const auto index = juce::String(static_cast<int>(i));
        snapshot.mean[i] = xml->getDoubleAttribute("mean" + index,
            std::numeric_limits<double>::quiet_NaN());
        snapshot.m2[i] = xml->getDoubleAttribute("m2_" + index,
            std::numeric_limits<double>::quiet_NaN());
    }

    smartgate::TargetProfile validator;
    if (!validator.restore(snapshot))
        return false;

    // Publish a validated immutable snapshot. The audio thread applies it at
    // the next block boundary; no profile object is mutated from the UI thread.
    pendingTargetProfile_.publish(snapshot);
    return true;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new SmartGateAudioProcessor();
}
