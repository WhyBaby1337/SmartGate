#include "PluginProcessor.h"

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
            && parameterWithId->paramID.paramID == id) {
            parameterBase->setValueNotifyingHost(ranged->convertTo0to1(plainValue));
            return;
        }
    }
    fail("runtime test parameter must exist");
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
} // namespace

int main() {
    testImmediateRoundTrip();
    testInvalidAndLegacyStatesClearProfile();
    testRuntimeBlockAndToggleTransitions();
    std::cout << "All processor state integration tests passed.\n";
    return 0;
}
