#include "PluginEditor.h"
#include "PluginProcessor.h"

namespace {
const auto background = juce::Colour(0xff101318);
const auto panel = juce::Colour(0xff191f27);
const auto accent = juce::Colour(0xff65d6b0);
const auto muted = juce::Colour(0xff8b98a8);
}

SmartGateAudioProcessorEditor::SmartGateAudioProcessorEditor(
    SmartGateAudioProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor) {
    setSize(760, 560);
    setResizable(true, true);
    setResizeLimits(640, 500, 1100, 760);

    for (std::size_t i = 0; i < sliders_.size(); ++i) {
        const juce::String suffix = (i == 0 || i == 4 || i == 5)
            ? " dB" : (i >= 6 ? "" : " ms");
        configureSlider(i, displayNames_[i], suffix);
        sliderAttachments_[i] = std::make_unique<SliderAttachment>(
            processor_.parameters_, parameterNames_[i], sliders_[i]);
    }

    aiButton_.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    learnButton_.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    aiButton_.setColour(juce::ToggleButton::tickColourId, accent);
    learnButton_.setColour(juce::ToggleButton::tickColourId, accent);
    addAndMakeVisible(aiButton_);
    addAndMakeVisible(learnButton_);
    aiAttachment_ = std::make_unique<ButtonAttachment>(
        processor_.parameters_, "aiEnabled", aiButton_);
    learnAttachment_ = std::make_unique<ButtonAttachment>(
        processor_.parameters_, "learnTarget", learnButton_);

    keyboardModeBox_.addItemList({"Universal", "Mechanical", "Membrane"}, 1);
    keyboardModeBox_.setColour(juce::ComboBox::backgroundColourId, panel);
    keyboardModeBox_.setColour(juce::ComboBox::textColourId, juce::Colours::white);
    addAndMakeVisible(keyboardModeBox_);
    keyboardModeAttachment_ = std::make_unique<ComboAttachment>(
        processor_.parameters_, "keyboardMode", keyboardModeBox_);

    keyboardModeLabel_.setText("KEYBOARD TYPE", juce::dontSendNotification);
    keyboardModeLabel_.setColour(juce::Label::textColourId, muted);
    keyboardModeLabel_.setFont(juce::Font(juce::FontOptions(11.0f, juce::Font::bold)));
    addAndMakeVisible(keyboardModeLabel_);
}

void SmartGateAudioProcessorEditor::configureSlider(
    std::size_t index, const juce::String& label, const juce::String& suffix) {
    auto& slider = sliders_[index];
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 76, 20);
    slider.setTextValueSuffix(suffix);
    slider.setColour(juce::Slider::rotarySliderFillColourId, accent);
    slider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff35404d));
    slider.setColour(juce::Slider::thumbColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxBackgroundColourId, panel);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(slider);

    auto& title = labels_[index];
    title.setText(label, juce::dontSendNotification);
    title.setJustificationType(juce::Justification::centred);
    title.setColour(juce::Label::textColourId, muted);
    title.setFont(juce::Font(juce::FontOptions(10.5f, juce::Font::bold)));
    addAndMakeVisible(title);
}

void SmartGateAudioProcessorEditor::layoutControl(
    std::size_t index, juce::Rectangle<int> area) {
    labels_[index].setBounds(area.removeFromTop(20));
    sliders_[index].setBounds(area.reduced(4, 0));
}

void SmartGateAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(background);
    auto bounds = getLocalBounds().toFloat();

    g.setColour(panel);
    g.fillRoundedRectangle(bounds.reduced(12.0f), 14.0f);

    g.setColour(accent);
    g.fillRoundedRectangle(juce::Rectangle<float>(28.0f, 28.0f, 4.0f, 34.0f), 2.0f);

    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(25.0f, juce::Font::bold)));
    g.drawText("SMART GATE", 44, 25, 260, 30, juce::Justification::left);

    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(11.0f)));
    g.drawText("TARGET-AWARE NOISE CONTROL", 46, 55, 300, 18,
               juce::Justification::left);

    g.setColour(juce::Colour(0xff2a333e));
    g.drawHorizontalLine(91, 28.0f, getWidth() - 28.0f);
}

void SmartGateAudioProcessorEditor::resized() {
    auto area = getLocalBounds().reduced(28);
    area.removeFromTop(76);

    auto modeArea = area.removeFromTop(48);
    aiButton_.setBounds(modeArea.removeFromLeft(145).reduced(4, 7));
    learnButton_.setBounds(modeArea.removeFromLeft(170).reduced(4, 7));
    auto keyboardArea = modeArea.removeFromLeft(190);
    keyboardModeLabel_.setBounds(keyboardArea.removeFromTop(15));
    keyboardModeBox_.setBounds(keyboardArea.removeFromTop(28));

    area.removeFromTop(10);
    const int gap = 8;
    const int cellWidth = (area.getWidth() - 2 * gap) / 3;
    const int cellHeight = (area.getHeight() - 2 * gap) / 3;

    for (std::size_t i = 0; i < sliders_.size(); ++i) {
        const int row = static_cast<int>(i / 3);
        const int column = static_cast<int>(i % 3);
        auto cell = juce::Rectangle<int>(
            area.getX() + column * (cellWidth + gap),
            area.getY() + row * (cellHeight + gap),
            cellWidth, cellHeight);
        layoutControl(i, cell.reduced(4));
    }
}
