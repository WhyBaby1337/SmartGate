#include "PluginProcessor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>

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
    auto* xml = juce::AudioProcessor::getXmlFromBinary(
        data.getData(), static_cast<int>(data.getSize()));
    require(xml != nullptr, "saved processor state must decode");
    return std::unique_ptr<juce::XmlElement>(xml);
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
    std::cout << "All processor state integration tests passed.\n";
    return 0;
}
