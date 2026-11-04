#include "PluginProcessor.h"
#include "GateEnvelope.h"

SmartGateAudioProcessor::SmartGateAudioProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters_(*this, nullptr, "PARAMETERS", {
          std::make_unique<juce::AudioParameterFloat>("threshold", "Threshold", -80.0f, 0.0f, -36.0f) })
{}
void SmartGateAudioProcessor::prepareToPlay(double, int) {}
void SmartGateAudioProcessor::releaseResources() {}
bool SmartGateAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto& in=layouts.getMainInputChannelSet(); const auto& out=layouts.getMainOutputChannelSet();
    return in==out && (in==juce::AudioChannelSet::mono() || in==juce::AudioChannelSet::stereo());
}
void SmartGateAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
    juce::ScopedNoDenormals noDenormals;
    const auto threshold=juce::Decibels::decibelsToGain(*parameters_.getRawParameterValue("threshold"));
    for (int s=0;s<buffer.getNumSamples();++s) {
        float peak=0.0f;
        for (int c=0;c<buffer.getNumChannels();++c) peak=juce::jmax(peak,std::abs(buffer.getSample(c,s)));
        const float gain=peak>=threshold ? 1.0f : 0.0f;
        for (int c=0;c<buffer.getNumChannels();++c) buffer.setSample(c,s,buffer.getSample(c,s)*gain);
    }
}
juce::AudioProcessorEditor* SmartGateAudioProcessor::createEditor() { return nullptr; }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SmartGateAudioProcessor(); }
