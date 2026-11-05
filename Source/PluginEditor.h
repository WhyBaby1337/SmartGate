#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <memory>

class SmartGateAudioProcessor;

class SmartGateAudioProcessorEditor final : public juce::AudioProcessorEditor {
public:
    explicit SmartGateAudioProcessorEditor(SmartGateAudioProcessor&);
    ~SmartGateAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void configureSlider(std::size_t index, const juce::String& label,
                         const juce::String& suffix);
    void layoutControl(std::size_t index, juce::Rectangle<int> area);

    SmartGateAudioProcessor& processor_;
    std::array<juce::Slider, 9> sliders_;
    std::array<juce::Label, 9> labels_;
    std::array<std::unique_ptr<SliderAttachment>, 9> sliderAttachments_;
    std::array<juce::String, 9> parameterNames_{
        "threshold", "attack", "release",
        "hold", "hysteresis", "range",
        "targetAmount", "keyboardReduction", "adaptiveHold"
    };
    std::array<juce::String, 9> displayNames_{
        "THRESHOLD", "ATTACK", "RELEASE",
        "HOLD", "HYSTERESIS", "RANGE",
        "TARGET INFLUENCE", "KEYBOARD REDUCTION", "ADAPTIVE HOLD"
    };

    juce::ToggleButton aiButton_{"AI ENGINE"};
    juce::ToggleButton learnButton_{"LEARN TARGET"};
    juce::ComboBox keyboardModeBox_;
    juce::Label keyboardModeLabel_;
    std::unique_ptr<ButtonAttachment> aiAttachment_;
    std::unique_ptr<ButtonAttachment> learnAttachment_;
    std::unique_ptr<ComboAttachment> keyboardModeAttachment_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SmartGateAudioProcessorEditor)
};
