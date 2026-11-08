#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <array>
#include <cstdint>
#include <memory>

class SmartGateAudioProcessor;

class SmartGateAudioProcessorEditor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit SmartGateAudioProcessorEditor(SmartGateAudioProcessor&);
    ~SmartGateAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    void configureSlider(std::size_t index, const juce::String& label,
                         const juce::String& suffix);
    void layoutControl(std::size_t index, juce::Rectangle<int> area);
    void drawMeter(juce::Graphics& g, juce::Rectangle<float> area,
                   const juce::String& label, float levelDb) const;
    void drawCpuMeter(juce::Graphics& g, juce::Rectangle<float> area) const;
    void chooseProfileExport();
    void chooseProfileImport();
    void confirmProfileReset();

    SmartGateAudioProcessor& processor_;
    float inputMeterDb_ = -100.0f;
    float outputMeterDb_ = -100.0f;
    float gateGain_ = 0.0f;
    float cpuLoadPercent_ = 0.0f;
    float cpuPeakPercent_ = 0.0f;
    std::uint64_t learnedFrames_ = 0;
    bool vadReady_ = false;
    std::array<juce::Slider, 10> sliders_;
    std::array<juce::Label, 10> labels_;
    std::array<std::unique_ptr<SliderAttachment>, 10> sliderAttachments_;
    std::array<juce::String, 10> parameterNames_{
        "threshold", "attack", "release",
        "hold", "hysteresis", "range",
        "targetAmount", "keyboardReduction", "adaptiveHold", "mouthNoiseReduction"
    };
    std::array<juce::String, 10> displayNames_{
        "THRESHOLD", "ATTACK", "RELEASE",
        "HOLD", "HYSTERESIS", "RANGE",
        "TARGET INFLUENCE", "KEYBOARD REDUCTION", "ADAPTIVE HOLD", "MOUTH CLICK REDUCTION"
    };

    juce::Slider voiceLevelingSlider_;
    juce::Label voiceLevelingLabel_;
    std::unique_ptr<SliderAttachment> voiceLevelingAttachment_;

    juce::ToggleButton aiButton_{"AI ENGINE"};
    juce::ToggleButton learnButton_{"START LEARNING"};
    juce::TextButton resetProfileButton_{"RESET PROFILE"};
    juce::ComboBox keyboardModeBox_;
    juce::Label keyboardModeLabel_;
    juce::TextButton exportProfileButton_{"EXPORT PROFILE"};
    juce::TextButton importProfileButton_{"IMPORT PROFILE"};
    juce::Label profileFileStatus_;
    std::unique_ptr<juce::FileChooser> profileFileChooser_;
    std::unique_ptr<ButtonAttachment> aiAttachment_;
    std::unique_ptr<ButtonAttachment> learnAttachment_;
    std::unique_ptr<ComboAttachment> keyboardModeAttachment_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SmartGateAudioProcessorEditor)
};
