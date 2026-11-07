#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <limits>

namespace {
[[noreturn]] void fail(const char* message) {
    std::cerr << "Processor state test failed: " << message << '\n';
    std::exit(1);
}
void require(bool condition, const char* message) {
    if (!condition) fail(message);
}
std::unique_ptr<juce::XmlElement> saveXml(SmartGateAudioProcessor& processor,
                                          juce::MemoryBlock& data) {
    processor.getStateInformation(data);
    auto xml = juce::AudioProcessor::getXmlFromBinary(
        data.getData(), static_cast<int>(data.getSize()));
    require(xml != nullptr, "saved processor state must decode");
    return xml;
}
void encode(SmartGateAudioProcessor& processor, const juce::XmlElement& xml,
            juce::MemoryBlock& data) {
    processor.copyXmlToBinary(xml, data);
}
juce::XmlElement* profile(juce::XmlElement& xml) {
    return xml.getChildByName("TARGET_PROFILE");
}
void addValidProfile(juce::XmlElement& xml, int count) {
    auto* node = xml.createNewChildElement("TARGET_PROFILE");
    node->setAttribute("count", count);
    node->setAttribute("mean0", -1.0);
    node->setAttribute("mean1", 0.25);
    node->setAttribute("mean2", 0.30);
    node->setAttribute("mean3", 0.40);
    node->setAttribute("mean4", 0.30);
    for (int i = 0; i < 5; ++i)
        node->setAttribute("m2_" + juce::String(i), 0.0);
}
juce::XmlElement* parameter(juce::XmlElement& xml, const juce::String& id) {
    for (auto* child = xml.getFirstChildElement(); child != nullptr;
         child = child->getNextElement()) {
        if (child->hasTagName("PARAM") && child->getStringAttribute("id") == id)
            return child;
    }
    return nullptr;
}
void setParameter(SmartGateAudioProcessor& processor, const juce::String& id,
                  float plainValue) {
    for (auto* parameterBase : processor.getParameters()) {
        auto* parameterWithId =
            dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
        auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
        if (parameterWithId != nullptr && ranged != nullptr
            && parameterWithId->paramID == id) {
            parameterBase->setValueNotifyingHost(ranged->convertTo0to1(plainValue));
            return;
        }
    }
    fail("runtime test parameter must exist");
}
void testHostMetadataAndAutomationParameters() {
    SmartGateAudioProcessor processor;
    require(processor.getTailLengthSeconds() == 0.0,
            "feed-forward gate must not advertise a generated or delayed audio tail");
    require(!processor.acceptsMidi() && !processor.producesMidi() && !processor.isMidiEffect(),
            "Smart Gate must advertise its audio-effect MIDI capabilities accurately");
    require(processor.hasEditor(), "host metadata must advertise the plugin editor");

    const juce::StringArray expectedIds{
        "threshold", "attack", "release", "hold", "hysteresis", "adaptiveHold",
        "range", "aiEnabled", "learnTarget", "targetAmount", "keyboardMode",
        "keyboardReduction"
    };
    require(processor.getParameters().size() == expectedIds.size(),
            "host parameter count must match the stable parameter layout");
    for (int i = 0; i < expectedIds.size(); ++i) {
        auto* parameterWithId = dynamic_cast<juce::AudioProcessorParameterWithID*>(
            processor.getParameters()[i]);
        require(parameterWithId != nullptr && parameterWithId->paramID == expectedIds[i],
                "host automation parameter IDs and ordering must remain stable");
        require(processor.getParameters()[i]->isAutomatable(),
                "every exposed Smart Gate control must support host automation");
    }
}

void testRuntimeBlockAndToggleTransitions() {
    SmartGateAudioProcessor processor;
    juce::MidiBuffer midi;
    constexpr int blockSizes[] = {1, 63, 256, 511, 1024};
    for (const double rate : {44100.0, 48000.0, 96000.0}) {
        processor.prepareToPlay(rate, 1024);
        for (int cycle = 0; cycle < 3; ++cycle) {
            setParameter(processor, "aiEnabled", cycle == 1 ? 0.0f : 1.0f);
            setParameter(processor, "learnTarget", cycle == 2 ? 1.0f : 0.0f);
            setParameter(processor, "keyboardReduction", cycle == 0 ? 0.0f : 0.8f);
            setParameter(processor, "keyboardMode", static_cast<float>(cycle));
            setParameter(processor, "targetAmount", cycle == 1 ? 0.0f : 0.75f);
            for (const int blockSize : blockSizes) {
                juce::AudioBuffer<float> buffer(2, blockSize);
                for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
                    auto* samples = buffer.getWritePointer(channel);
                    for (int i = 0; i < blockSize; ++i) {
                        const double phase = 2.0 * juce::MathConstants<double>::pi
                            * 220.0 * static_cast<double>(i) / rate;
                        samples[i] = static_cast<float>(0.2 * std::sin(phase));
                    }
                }
                if (blockSize > 1)
                    buffer.getWritePointer(0)[0] = std::numeric_limits<float>::quiet_NaN();
                processor.processBlock(buffer, midi);
                for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
                    for (int i = 0; i < blockSize; ++i)
                        require(std::isfinite(buffer.getReadPointer(channel)[i]),
                                "runtime processing must keep output finite across toggles and block sizes");
            }
        }
        processor.releaseResources();
    }
}
void testMonoRuntimeProcessing() {
    SmartGateAudioProcessor processor;
    juce::MidiBuffer midi;
    constexpr int blockSizes[] = {1, 63, 256, 511, 1024};
    for (const double rate : {44100.0, 48000.0, 96000.0}) {
        processor.prepareToPlay(rate, 1024);
        for (const bool enableAi : {false, true}) {
            for (const bool enableKeyboard : {false, true}) {
                setParameter(processor, "aiEnabled", enableAi ? 1.0f : 0.0f);
                setParameter(processor, "keyboardReduction", enableKeyboard ? 0.7f : 0.0f);
                for (const int blockSize : blockSizes) {
                    juce::AudioBuffer<float> buffer(1, blockSize);
                    auto* samples = buffer.getWritePointer(0);
                    for (int i = 0; i < blockSize; ++i) {
                        const double phase = 2.0 * juce::MathConstants<double>::pi
                            * 330.0 * static_cast<double>(i) / rate;
                        samples[i] = static_cast<float>(0.15 * std::sin(phase));
                    }
                    if (blockSize > 1)
                        samples[0] = std::numeric_limits<float>::quiet_NaN();
                    processor.processBlock(buffer, midi);
                    for (int i = 0; i < blockSize; ++i)
                        require(std::isfinite(samples[i]),
                                "mono runtime processing must keep output finite across block sizes and analyzer states");
                }
            }
        }
        processor.releaseResources();
    }
}
void testAllParametersSurviveStateRoundTrip() {
    SmartGateAudioProcessor source;

    // Exercise every APVTS parameter, not only the threshold control. This
    // catches mismatched IDs and regressions when controls are added or renamed.
    setParameter(source, "threshold", -23.4f);
    setParameter(source, "attack", 12.3f);
    setParameter(source, "release", 345.0f);
    setParameter(source, "hold", 123.0f);
    setParameter(source, "hysteresis", 4.7f);
    setParameter(source, "adaptiveHold", 0.73f);
    setParameter(source, "range", -42.1f);
    setParameter(source, "aiEnabled", 0.0f);
    setParameter(source, "learnTarget", 1.0f);
    setParameter(source, "targetAmount", 0.82f);
    setParameter(source, "keyboardMode", 2.0f);
    setParameter(source, "keyboardReduction", 0.64f);

    juce::MemoryBlock saved;
    (void) saveXml(source, saved);

    SmartGateAudioProcessor restored;
    restored.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));

    const auto expect = [&](const char* id, float expected, float tolerance = 1.0e-4f) {
        float actualValue = std::numeric_limits<float>::quiet_NaN();
        for (auto* parameterBase : restored.getParameters()) {
            auto* parameterWithId =
                dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
            auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
            if (parameterWithId != nullptr && ranged != nullptr
                && parameterWithId->paramID == id) {
                actualValue = ranged->convertFrom0to1(parameterBase->getValue());
                break;
            }
        }
        require(std::isfinite(actualValue),
                "every persisted parameter must exist in the restored layout");
        if (std::abs(actualValue - expected) > tolerance)
            std::cerr << "Round-trip mismatch for " << id << ": expected "
                      << expected << ", got " << actualValue << '\\n';
        require(std::abs(actualValue - expected) <= tolerance,
                "every APVTS parameter must survive a complete save/load round trip");
    };

    expect("threshold", -23.4f);
    expect("attack", 12.3f);
    expect("release", 345.0f);
    expect("hold", 123.0f);
    expect("hysteresis", 4.7f);
    expect("adaptiveHold", 0.73f);
    expect("range", -42.1f);
    expect("aiEnabled", 0.0f);
    expect("learnTarget", 1.0f);
    expect("targetAmount", 0.82f);
    expect("keyboardMode", 2.0f);
    expect("keyboardReduction", 0.64f);
}

void testImmediateRoundTrip() {
    SmartGateAudioProcessor source;
    juce::MemoryBlock unused;
    auto sourceXml = saveXml(source, unused);
    addValidProfile(*sourceXml, 37);
    auto* threshold = parameter(*sourceXml, "threshold");
    require(threshold != nullptr, "threshold parameter must be serialized");
    threshold->setAttribute("value", -17.3);
    juce::MemoryBlock input;
    encode(source, *sourceXml, input);

    SmartGateAudioProcessor restored;
    restored.setStateInformation(input.getData(), static_cast<int>(input.getSize()));
    juce::MemoryBlock immediate;
    auto immediateXml = saveXml(restored, immediate);
    auto* restoredProfile = profile(*immediateXml);
    require(restoredProfile != nullptr, "immediate save must preserve pending profile");
    require(restoredProfile->getIntAttribute("count", -1) == 37,
            "immediate save must preserve profile count");
    require(std::abs(restoredProfile->getDoubleAttribute("mean0") + 1.0) < 1.0e-12,
            "immediate save must preserve profile statistics");
    auto* restoredThreshold = parameter(*immediateXml, "threshold");
    require(restoredThreshold != nullptr, "threshold must survive state restore");
    require(std::abs(restoredThreshold->getDoubleAttribute("value") + 17.3) < 1.0e-4,
            "threshold value must survive state restore");

    SmartGateAudioProcessor second;
    second.setStateInformation(immediate.getData(), static_cast<int>(immediate.getSize()));
    juce::MemoryBlock repeated;
    auto repeatedXml = saveXml(second, repeated);
    int count = 0;
    for (auto* child = repeatedXml->getFirstChildElement(); child != nullptr;
         child = child->getNextElement())
        if (child->hasTagName("TARGET_PROFILE")) ++count;
    require(count == 1, "repeated saves must contain exactly one target profile");
    require(profile(*repeatedXml)->getIntAttribute("count", -1) == 37,
            "repeated save/load must preserve target profile");
}

void testPreparedRestoreCanBeSavedBeforeAudioCallback() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 37);
    auto* threshold = parameter(*fixture, "threshold");
    require(threshold != nullptr, "prepared restore fixture must contain threshold");
    threshold->setAttribute("value", -21.7);
    juce::MemoryBlock encodedFixture;
    encode(fixtureProcessor, *fixture, encodedFixture);

    SmartGateAudioProcessor processor;
    processor.prepareToPlay(48000.0, 256);
    processor.setStateInformation(encodedFixture.getData(),
                                  static_cast<int>(encodedFixture.getSize()));

    // Hosts may save immediately after restoring a preset/project, before
    // the next processBlock() has had a chance to apply the pending profile.
    juce::MemoryBlock immediate;
    auto savedXml = saveXml(processor, immediate);
    auto* savedProfile = profile(*savedXml);
    require(savedProfile != nullptr && savedProfile->getIntAttribute("count", -1) == 37,
            "saving immediately after a prepared restore must preserve the pending profile");
    auto* savedThreshold = parameter(*savedXml, "threshold");
    require(savedThreshold != nullptr
                && std::abs(savedThreshold->getDoubleAttribute("value") + 21.7) < 1.0e-4,
            "saving immediately after a prepared restore must preserve restored parameters");
    processor.releaseResources();
}

void testLatestRestoreWinsBeforeAudioCallback() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 24);
    juce::MemoryBlock firstState;
    encode(fixtureProcessor, *fixture, firstState);

    juce::XmlElement newerFixture(*fixture);
    profile(newerFixture)->setAttribute("count", 37);
    profile(newerFixture)->setAttribute("mean0", -1.2);
    juce::MemoryBlock secondState;
    encode(fixtureProcessor, newerFixture, secondState);

    SmartGateAudioProcessor processor;
    processor.prepareToPlay(48000.0, 256);
    processor.setStateInformation(firstState.getData(), static_cast<int>(firstState.getSize()));
    processor.setStateInformation(secondState.getData(), static_cast<int>(secondState.getSize()));

    // No processBlock() occurs between the restores. The newest host state
    // must replace the older pending snapshot and survive an immediate save.
    juce::MemoryBlock saved;
    auto savedXml = saveXml(processor, saved);
    auto* savedProfile = profile(*savedXml);
    require(savedProfile != nullptr
                && savedProfile->getIntAttribute("count", -1) == 37,
            "the latest of multiple pending restores must win before audio processing");
    require(std::abs(savedProfile->getDoubleAttribute("mean0") + 1.2) < 1.0e-12,
            "the latest pending restore must preserve its own profile statistics");
    processor.releaseResources();
}

void testDuplicateTargetProfilesAreRejected() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 24);
    // A second, individually valid but conflicting snapshot makes the state
    // ambiguous. The loader must not silently select the first XML child.
    addValidProfile(*fixture, 37);
    auto* threshold = parameter(*fixture, "threshold");
    require(threshold != nullptr, "duplicate-profile fixture must contain threshold");
    threshold->setAttribute("value", -19.6);

    juce::MemoryBlock encoded;
    encode(fixtureProcessor, *fixture, encoded);

    SmartGateAudioProcessor restored;
    restored.setStateInformation(encoded.getData(), static_cast<int>(encoded.getSize()));

    juce::MemoryBlock saved;
    auto savedXml = saveXml(restored, saved);
    auto* savedProfile = profile(*savedXml);
    require(savedProfile != nullptr && savedProfile->getIntAttribute("count", -1) == 0,
            "duplicate target profile metadata must be rejected and cleared");
    auto* restoredThreshold = parameter(*savedXml, "threshold");
    require(restoredThreshold != nullptr
                && std::abs(restoredThreshold->getDoubleAttribute("value") + 19.6) < 1.0e-4,
            "rejecting duplicate profile metadata must not discard valid APVTS parameters");
}

void testInvalidProfileVariantsClearPreviousProfile() {
    SmartGateAudioProcessor processor;
    juce::MemoryBlock empty;
    auto fixture = saveXml(processor, empty);
    addValidProfile(*fixture, 24);

    const auto expectInvalidProfileCleared = [&](juce::XmlElement& invalid,
                                                  const char* message) {
        juce::MemoryBlock encoded;
        encode(processor, invalid, encoded);
        processor.setStateInformation(encoded.getData(),
                                      static_cast<int>(encoded.getSize()));
        juce::MemoryBlock saved;
        auto savedXml = saveXml(processor, saved);
        auto* savedProfile = profile(*savedXml);
        require(savedProfile != nullptr
                    && savedProfile->getIntAttribute("count", -1) == 0,
                message);
    };

    juce::XmlElement negativeVariance(*fixture);
    profile(negativeVariance)->setAttribute("m2_2", -0.01);
    expectInvalidProfileCleared(
        negativeVariance, "negative profile variance must clear previous training");

    juce::XmlElement invalidBandRatios(*fixture);
    profile(invalidBandRatios)->setAttribute("mean2", 0.8);
    expectInvalidProfileCleared(
        invalidBandRatios, "inconsistent spectral ratios must clear previous training");

    juce::XmlElement excessiveCount(*fixture);
    profile(excessiveCount)->setAttribute("count", 1000000001);
    expectInvalidProfileCleared(
        excessiveCount, "excessive restored frame counts must clear previous training");

    juce::XmlElement missingStatistic(*fixture);
    profile(missingStatistic)->removeAttribute("m2_4");
    expectInvalidProfileCleared(
        missingStatistic, "incomplete profile statistics must clear previous training");
}

void testLearnTargetRequestWaitsUntilAiIsEnabled() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 24);
    juce::MemoryBlock encodedFixture;
    encode(fixtureProcessor, *fixture, encodedFixture);

    SmartGateAudioProcessor processor;
    processor.setStateInformation(encodedFixture.getData(),
                                  static_cast<int>(encodedFixture.getSize()));
    processor.prepareToPlay(48000.0, 256);

    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(2, 1);
    setParameter(processor, "aiEnabled", 0.0f);
    setParameter(processor, "learnTarget", 1.0f);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock whileAiDisabled;
    auto preservedState = saveXml(processor, whileAiDisabled);
    require(profile(*preservedState) != nullptr
                && profile(*preservedState)->getIntAttribute("count", -1) == 24,
            "Learn Target requested with AI disabled must preserve the existing profile");

    setParameter(processor, "aiEnabled", 1.0f);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock afterAiEnabled;
    auto clearedState = saveXml(processor, afterAiEnabled);
    require(profile(*clearedState) != nullptr
                && profile(*clearedState)->getIntAttribute("count", -1) == 0,
            "deferred Learn Target request must clear the old profile when AI is re-enabled");
    processor.releaseResources();
}

void testExplicitLearningWinsOverPendingProfileRestore() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 37);
    juce::MemoryBlock encodedFixture;
    encode(fixtureProcessor, *fixture, encodedFixture);

    SmartGateAudioProcessor processor;
    processor.prepareToPlay(48000.0, 256);
    processor.setStateInformation(encodedFixture.getData(),
                                  static_cast<int>(encodedFixture.getSize()));

    // Both operations are pending for the next audio block: restore a saved
    // profile, then explicitly request a fresh Learn Target session.
    setParameter(processor, "learnTarget", 1.0f);
    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(2, 1);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock saved;
    auto savedXml = saveXml(processor, saved);
    auto* savedProfile = profile(*savedXml);
    require(savedProfile != nullptr && savedProfile->getIntAttribute("count", -1) == 0,
            "an explicit Learn Target request must override a profile restored in the same block");
    processor.releaseResources();
}

void testRestoreWithLearningEnabledAfterPreparePreservesProfile() {
    SmartGateAudioProcessor fixtureProcessor;
    setParameter(fixtureProcessor, "learnTarget", 1.0f);
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 37);
    juce::MemoryBlock encodedFixture;
    encode(fixtureProcessor, *fixture, encodedFixture);

    SmartGateAudioProcessor processor;
    processor.prepareToPlay(48000.0, 256);
    // Some hosts/preset flows restore state after the audio processor is prepared.
    processor.setStateInformation(encodedFixture.getData(),
                                  static_cast<int>(encodedFixture.getSize()));

    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(2, 1);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock saved;
    auto savedXml = saveXml(processor, saved);
    auto* savedProfile = profile(*savedXml);
    require(savedProfile != nullptr && savedProfile->getIntAttribute("count", -1) == 37,
            "restoring a session with Learn Target enabled after prepare must preserve its profile");
    processor.releaseResources();
}

void testLearnTargetCanRestartAfterRestoredEnabledSession() {
    SmartGateAudioProcessor fixtureProcessor;
    setParameter(fixtureProcessor, "learnTarget", 1.0f);
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);
    addValidProfile(*fixture, 37);
    juce::MemoryBlock encodedFixture;
    encode(fixtureProcessor, *fixture, encodedFixture);

    SmartGateAudioProcessor processor;
    processor.prepareToPlay(48000.0, 256);
    processor.setStateInformation(encodedFixture.getData(),
                                  static_cast<int>(encodedFixture.getSize()));

    juce::MidiBuffer midi;
    juce::AudioBuffer<float> buffer(2, 1);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock restored;
    auto restoredXml = saveXml(processor, restored);
    require(profile(*restoredXml) != nullptr
                && profile(*restoredXml)->getIntAttribute("count", -1) == 37,
            "restoring an enabled Learn Target session must preserve its profile");

    setParameter(processor, "learnTarget", 0.0f);
    processor.processBlock(buffer, midi);
    setParameter(processor, "learnTarget", 1.0f);
    processor.processBlock(buffer, midi);

    juce::MemoryBlock restarted;
    auto restartedXml = saveXml(processor, restarted);
    require(profile(*restartedXml) != nullptr
                && profile(*restartedXml)->getIntAttribute("count", -1) == 0,
            "turning Learn Target off and on after restore must start a fresh profile");
    processor.releaseResources();
}


void testLegacyPartialParameterStateUsesDefaults() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto legacy = saveXml(fixtureProcessor, fixtureData);

    auto* threshold = parameter(*legacy, "threshold");
    require(threshold != nullptr, "legacy fixture must contain threshold");
    threshold->setAttribute("value", -22.5);

    // Simulate a project saved before newer controls existed.
    for (auto* child = legacy->getFirstChildElement(); child != nullptr;) {
        auto* next = child->getNextElement();
        if (child->hasTagName("PARAM")
            && child->getStringAttribute("id") != "threshold")
            legacy->removeChildElement(child, true);
        child = next;
    }

    // Unknown future/foreign parameter IDs must not interfere with known IDs.
    auto* unknown = legacy->createNewChildElement("PARAM");
    unknown->setAttribute("id", "parameterFromAnotherVersion");
    unknown->setAttribute("value", 999.0);

    juce::MemoryBlock encoded;
    encode(fixtureProcessor, *legacy, encoded);

    SmartGateAudioProcessor restored;
    setParameter(restored, "threshold", -8.0f);
    restored.setStateInformation(encoded.getData(), static_cast<int>(encoded.getSize()));

    const auto value = [&](const char* id) {
        for (auto* parameterBase : restored.getParameters()) {
            auto* parameterWithId =
                dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
            auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
            if (parameterWithId != nullptr && ranged != nullptr
                && parameterWithId->paramID == id)
                return ranged->convertFrom0to1(parameterBase->getValue());
        }
        return std::numeric_limits<float>::quiet_NaN();
    };

    require(std::abs(value("threshold") - (-22.5f)) <= 1.0e-4f,
            "legacy state must restore parameters that are present");
    require(std::abs(value("release") - 120.0f) <= 1.0e-4f,
            "missing legacy parameters must retain their current layout defaults");
    require(std::abs(value("hold") - 80.0f) <= 1.0e-4f,
            "multiple missing legacy parameters must retain their defaults");

    // The host may save the legacy tree without materializing parameters
    // that were absent in the original project. That is valid: a subsequent
    // load must continue to obtain the current layout defaults.
    juce::MemoryBlock saved;
    (void) saveXml(restored, saved);

    SmartGateAudioProcessor roundTripped;
    roundTripped.setStateInformation(saved.getData(), static_cast<int>(saved.getSize()));
    const auto roundTripValue = [&](const char* id) {
        for (auto* parameterBase : roundTripped.getParameters()) {
            auto* parameterWithId =
                dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
            auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
            if (parameterWithId != nullptr && ranged != nullptr
                && parameterWithId->paramID == id)
                return ranged->convertFrom0to1(parameterBase->getValue());
        }
        return std::numeric_limits<float>::quiet_NaN();
    };
    require(std::abs(roundTripValue("threshold") - (-22.5f)) <= 1.0e-4f,
            "present legacy parameters must survive a save/load round trip");
    require(std::abs(roundTripValue("release") - 120.0f) <= 1.0e-4f
                && std::abs(roundTripValue("hold") - 80.0f) <= 1.0e-4f,
            "missing legacy parameters must continue to use layout defaults after a round trip");
}

void testDuplicateParameterIdsAreRejected() {
    SmartGateAudioProcessor fixtureProcessor;
    juce::MemoryBlock fixtureData;
    auto fixture = saveXml(fixtureProcessor, fixtureData);

    auto* threshold = parameter(*fixture, "threshold");
    require(threshold != nullptr, "duplicate-parameter fixture must contain threshold");
    threshold->setAttribute("value", -12.4);
    auto* duplicateThreshold = fixture->createNewChildElement("PARAM");
    duplicateThreshold->setAttribute("id", "threshold");
    duplicateThreshold->setAttribute("value", -55.0);

    auto* release = parameter(*fixture, "release");
    require(release != nullptr, "duplicate-parameter fixture must contain release");
    release->setAttribute("value", 345.0);

    juce::MemoryBlock encoded;
    encode(fixtureProcessor, *fixture, encoded);

    SmartGateAudioProcessor restored;
    restored.setStateInformation(encoded.getData(), static_cast<int>(encoded.getSize()));

    const auto value = [&](const char* id) {
        for (auto* parameterBase : restored.getParameters()) {
            auto* parameterWithId =
                dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
            auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
            if (parameterWithId != nullptr && ranged != nullptr
                && parameterWithId->paramID == id)
                return ranged->convertFrom0to1(parameterBase->getValue());
        }
        return std::numeric_limits<float>::quiet_NaN();
    };

    require(std::abs(value("threshold") - (-36.0f)) <= 1.0e-4f,
            "ambiguous duplicate parameter IDs must fall back to the parameter default");
    require(std::abs(value("release") - 345.0f) <= 1.0e-4f,
            "rejecting a duplicate parameter must preserve unrelated valid parameters");

    juce::MemoryBlock saved;
    auto savedXml = saveXml(restored, saved);
    int thresholdNodes = 0;
    for (auto* child = savedXml->getFirstChildElement(); child != nullptr;
         child = child->getNextElement()) {
        if (child->hasTagName("PARAM")
            && child->getStringAttribute("id") == "threshold")
            ++thresholdNodes;
    }
    require(thresholdNodes == 1,
            "saving after duplicate-parameter recovery must write one canonical parameter node");
}

void testMalformedAndNonFiniteParameterValuesAreRejected() {
    const char* invalidValues[] = {"nan", "inf", "-inf", "not-a-number", "1e9999"};
    for (const auto* invalidValue : invalidValues) {
        SmartGateAudioProcessor fixtureProcessor;
        juce::MemoryBlock fixtureData;
        auto fixture = saveXml(fixtureProcessor, fixtureData);

        auto* threshold = parameter(*fixture, "threshold");
        require(threshold != nullptr, "invalid-value fixture must contain threshold");
        threshold->setAttribute("value", invalidValue);
        auto* release = parameter(*fixture, "release");
        require(release != nullptr, "invalid-value fixture must contain release");
        release->setAttribute("value", 345.0);

        juce::MemoryBlock encoded;
        encode(fixtureProcessor, *fixture, encoded);

        SmartGateAudioProcessor restored;
        setParameter(restored, "threshold", -8.0f);
        restored.setStateInformation(encoded.getData(), static_cast<int>(encoded.getSize()));

        const auto value = [&](const char* id) {
            for (auto* parameterBase : restored.getParameters()) {
                auto* parameterWithId =
                    dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
                auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
                if (parameterWithId != nullptr && ranged != nullptr
                    && parameterWithId->paramID == id)
                    return ranged->convertFrom0to1(parameterBase->getValue());
            }
            return std::numeric_limits<float>::quiet_NaN();
        };

        require(std::abs(value("threshold") - (-36.0f)) <= 1.0e-4f,
                "malformed or non-finite parameter values must restore the declared default");
        require(std::abs(value("release") - 345.0f) <= 1.0e-4f,
                "rejecting one invalid parameter must preserve unrelated valid parameters");
    }
}

void testMalformedAndForeignRootStatesAreIgnored() {
    SmartGateAudioProcessor processor;
    setParameter(processor, "threshold", -18.0f);
    juce::MemoryBlock initialData;
    auto initialState = saveXml(processor, initialData);
    addValidProfile(*initialState, 24);
    encode(processor, *initialState, initialData);
    processor.setStateInformation(initialData.getData(),
                                  static_cast<int>(initialData.getSize()));

    juce::XmlElement foreignRoot("FOREIGN_STATE");
    juce::MemoryBlock foreignData;
    encode(processor, foreignRoot, foreignData);
    processor.setStateInformation(foreignData.getData(),
                                  static_cast<int>(foreignData.getSize()));

    float thresholdAfterForeignRoot = std::numeric_limits<float>::quiet_NaN();
    for (auto* parameterBase : processor.getParameters()) {
        auto* parameterWithId =
            dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
        auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
        if (parameterWithId != nullptr && ranged != nullptr
            && parameterWithId->paramID == "threshold") {
            thresholdAfterForeignRoot = ranged->convertFrom0to1(parameterBase->getValue());
            break;
        }
    }
    require(std::abs(thresholdAfterForeignRoot - (-18.0f)) <= 1.0e-4f,
            "state with a foreign XML root must leave current parameters unchanged");
    juce::MemoryBlock afterForeignRoot;
    auto foreignRootSaved = saveXml(processor, afterForeignRoot);
    require(profile(*foreignRootSaved) != nullptr
                && profile(*foreignRootSaved)->getIntAttribute("count", -1) == 24,
            "state with a foreign XML root must leave the pending target profile unchanged");

    const char malformedBytes[] = {
        static_cast<char>(0x13), static_cast<char>(0x37),
        static_cast<char>(0x00), static_cast<char>(0x7f),
        static_cast<char>(0x55), static_cast<char>(0xaa),
        static_cast<char>(0x21), static_cast<char>(0x42)
    };
    processor.setStateInformation(malformedBytes, static_cast<int>(sizeof(malformedBytes)));

    float thresholdAfterMalformedData = std::numeric_limits<float>::quiet_NaN();
    for (auto* parameterBase : processor.getParameters()) {
        auto* parameterWithId =
            dynamic_cast<juce::AudioProcessorParameterWithID*>(parameterBase);
        auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameterBase);
        if (parameterWithId != nullptr && ranged != nullptr
            && parameterWithId->paramID == "threshold") {
            thresholdAfterMalformedData = ranged->convertFrom0to1(parameterBase->getValue());
            break;
        }
    }
    require(std::abs(thresholdAfterMalformedData - (-18.0f)) <= 1.0e-4f,
            "malformed binary state must leave current parameters unchanged");
    juce::MemoryBlock afterMalformedData;
    auto malformedSaved = saveXml(processor, afterMalformedData);
    require(profile(*malformedSaved) != nullptr
                && profile(*malformedSaved)->getIntAttribute("count", -1) == 24,
            "malformed binary state must leave the pending target profile unchanged");
}

void testInvalidAndLegacyStatesClearProfile() {
    SmartGateAudioProcessor processor;
    juce::MemoryBlock empty;
    auto fixture = saveXml(processor, empty);
    addValidProfile(*fixture, 24);
    juce::MemoryBlock valid;
    encode(processor, *fixture, valid);
    processor.setStateInformation(valid.getData(), static_cast<int>(valid.getSize()));

    juce::XmlElement invalid(*fixture);
    profile(invalid)->setAttribute("mean0", 42.0);
    juce::MemoryBlock invalidData;
    encode(processor, invalid, invalidData);
    processor.setStateInformation(invalidData.getData(), static_cast<int>(invalidData.getSize()));
    juce::MemoryBlock afterInvalid;
    auto invalidSaved = saveXml(processor, afterInvalid);
    require(profile(*invalidSaved) != nullptr
            && profile(*invalidSaved)->getIntAttribute("count", -1) == 0,
            "invalid profile must serialize an explicit empty profile");

    juce::XmlElement legacy(*fixture);
    auto* oldProfile = profile(legacy);
    require(oldProfile != nullptr, "legacy fixture must start with a profile");
    legacy.removeChildElement(oldProfile, true);
    juce::MemoryBlock legacyData;
    encode(processor, legacy, legacyData);
    processor.setStateInformation(legacyData.getData(), static_cast<int>(legacyData.getSize()));
    juce::MemoryBlock afterLegacy;
    auto legacySaved = saveXml(processor, afterLegacy);
    require(profile(*legacySaved) != nullptr
            && profile(*legacySaved)->getIntAttribute("count", -1) == 0,
            "legacy state must serialize an explicit empty profile");
}

void testProcessorGateOpensClosesAndReopens() {
    SmartGateAudioProcessor processor;
    juce::MidiBuffer midi;
    constexpr double sampleRate = 48000.0;
    constexpr int blockSize = 256;
    processor.prepareToPlay(sampleRate, blockSize);

    setParameter(processor, "aiEnabled", 0.0f);
    setParameter(processor, "learnTarget", 0.0f);
    setParameter(processor, "keyboardReduction", 0.0f);
    setParameter(processor, "threshold", -36.0f);
    setParameter(processor, "attack", 5.0f);
    setParameter(processor, "release", 120.0f);
    setParameter(processor, "hold", 80.0f);
    setParameter(processor, "hysteresis", 3.0f);
    setParameter(processor, "range", -80.0f);

    const auto processTone = [&](float amplitude, int totalSamples) {
        float peak = 0.0f;
        int position = 0;
        while (position < totalSamples) {
            const int count = std::min(blockSize, totalSamples - position);
            juce::AudioBuffer<float> buffer(2, count);
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
                auto* samples = buffer.getWritePointer(channel);
                for (int i = 0; i < count; ++i) {
                    const double phase = 2.0 * juce::MathConstants<double>::pi
                        * 220.0 * static_cast<double>(position + i) / sampleRate;
                    samples[i] = amplitude * static_cast<float>(std::sin(phase));
                }
            }
            processor.processBlock(buffer, midi);
            for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
                for (int i = 0; i < count; ++i) {
                    const float output = buffer.getReadPointer(channel)[i];
                    require(std::isfinite(output),
                            "gate behavior test must keep every output sample finite");
                    peak = std::max(peak, std::abs(output));
                }
            position += count;
        }
        return peak;
    };

    const auto processSilence = [&](int totalSamples) {
        float finalPeak = 0.0f;
        int position = 0;
        while (position < totalSamples) {
            const int count = std::min(blockSize, totalSamples - position);
            juce::AudioBuffer<float> buffer(2, count);
            buffer.clear();
            processor.processBlock(buffer, midi);
            if (position + count == totalSamples) {
                for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
                    for (int i = 0; i < count; ++i)
                        finalPeak = std::max(finalPeak,
                            std::abs(buffer.getReadPointer(channel)[i]));
            }
            position += count;
        }
        return finalPeak;
    };

    (void) processSilence(static_cast<int>(sampleRate * 0.5));
    const float speechPeak = processTone(0.20f, static_cast<int>(sampleRate * 0.25));
    require(speechPeak > 0.15f,
            "conventional gate must open and pass a sustained above-threshold voice-like signal");

    // Wait long enough for the detector's 20 ms release smoothing to fall
    // below the closing threshold, but remain inside the configured 80 ms
    // Hold interval. The next quiet probe therefore tests Hold itself.
    (void) processSilence(static_cast<int>(sampleRate * 0.06));
    const float shortGapPeak = processTone(0.01f, static_cast<int>(sampleRate * 0.02));
    require(shortGapPeak > 0.0075f,
            "gate hold must preserve a quiet fragment immediately after a short pause");

    (void) processSilence(static_cast<int>(sampleRate * 1.0));
    // Probe with a non-zero signal below the opening threshold. Measuring
    // silence alone cannot prove that the gate closed because zero input
    // produces zero output even while the gate is fully open.
    const float closedPeak = processTone(0.01f, static_cast<int>(sampleRate * 0.05));
    require(closedPeak < 0.001f,
            "gate must attenuate a below-threshold probe after the hold/release interval expires");

    const float reopenedPeak = processTone(0.05f, static_cast<int>(sampleRate * 0.20));
    require(reopenedPeak > 0.035f,
            "gate must reopen for a subsequent quieter above-threshold voice-like signal");

    processor.releaseResources();
}

} // namespace


void testRepeatedPrepareWithoutRelease() {
    SmartGateAudioProcessor processor;
    juce::MidiBuffer midi;
    constexpr double rates[] = {48000.0, 44100.0, 96000.0};

    // Hosts can renegotiate the audio format while the same processor
    // instance is alive. prepareToPlay must stop/join the previous VAD
    // worker before replacing its model, even without an intervening release.
    for (const double rate : rates) {
        processor.prepareToPlay(rate, 512);
        juce::AudioBuffer<float> buffer(2, 257);
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            auto* samples = buffer.getWritePointer(channel);
            for (int i = 0; i < buffer.getNumSamples(); ++i) {
                const double phase = 2.0 * juce::MathConstants<double>::pi
                    * 440.0 * static_cast<double>(i) / rate;
                samples[i] = static_cast<float>(0.1 * std::sin(phase));
            }
        }
        processor.processBlock(buffer, midi);
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                require(std::isfinite(buffer.getReadPointer(channel)[i]),
                        "repeated prepare must keep audio output finite across sample-rate changes");
    }
    processor.releaseResources();
}

int main() {
    testHostMetadataAndAutomationParameters();
    testAllParametersSurviveStateRoundTrip();
    testImmediateRoundTrip();
    testPreparedRestoreCanBeSavedBeforeAudioCallback();
    testLatestRestoreWinsBeforeAudioCallback();
    testInvalidAndLegacyStatesClearProfile();
    testDuplicateParameterIdsAreRejected();
    testMalformedAndNonFiniteParameterValuesAreRejected();
    testMalformedAndForeignRootStatesAreIgnored();
    testLegacyPartialParameterStateUsesDefaults();
    testLearnTargetRequestWaitsUntilAiIsEnabled();
    testExplicitLearningWinsOverPendingProfileRestore();
    testRestoreWithLearningEnabledAfterPreparePreservesProfile();
    testLearnTargetCanRestartAfterRestoredEnabledSession();
    testInvalidProfileVariantsClearPreviousProfile();
    testDuplicateTargetProfilesAreRejected();
    testRuntimeBlockAndToggleTransitions();
    testProcessorGateOpensClosesAndReopens();
    testRepeatedPrepareWithoutRelease();
    testMonoRuntimeProcessing();
    std::cout << "All processor state integration tests passed.\n";
    return 0;
}
