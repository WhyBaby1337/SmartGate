#include "PluginEditor.h"
#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>

namespace {
const auto background = juce::Colour(0xff101318);
const auto panel = juce::Colour(0xff191f27);
const auto accent = juce::Colour(0xff65d6b0);
const auto muted = juce::Colour(0xff8b98a8);
const auto meterTrack = juce::Colour(0xff29333e);
const auto warning = juce::Colour(0xffffc36b);
}

SmartGateAudioProcessorEditor::SmartGateAudioProcessorEditor(
    SmartGateAudioProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor) {
    setSize(800, 640);
    setResizable(true, true);
    setResizeLimits(680, 560, 1200, 900);

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

    timerCallback();
    startTimerHz(30);
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

void SmartGateAudioProcessorEditor::drawMeter(
    juce::Graphics& g, juce::Rectangle<float> area,
    const juce::String& label, float levelDb) const {
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.drawText(label, area.removeFromTop(15.0f), juce::Justification::left);
    auto valueArea = area.removeFromTop(17.0f);
    const auto valueText = levelDb <= -99.0f ? juce::String("-inf dB")
                                              : juce::String(levelDb, 1) + " dB";
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(11.0f)));
    g.drawText(valueText, valueArea, juce::Justification::right);
    g.setColour(meterTrack);
    g.fillRoundedRectangle(area, 4.0f);
    const float normalized = juce::jlimit(0.0f, 1.0f, (levelDb + 60.0f) / 60.0f);
    auto fill = area.withWidth(area.getWidth() * normalized);
    g.setColour(levelDb > -6.0f ? warning : accent);
    g.fillRoundedRectangle(fill, 4.0f);
}

void SmartGateAudioProcessorEditor::drawCpuMeter(
    juce::Graphics& g, juce::Rectangle<float> area) const {
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.drawText("PLUGIN CPU", area.removeFromTop(15.0f), juce::Justification::left);

    const auto valueArea = area.removeFromTop(17.0f);
    g.setColour(cpuPeakPercent_ >= 100.0f ? warning : juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(11.0f)));
    g.drawText(juce::String(cpuLoadPercent_, 1) + "%  PEAK "
                   + juce::String(cpuPeakPercent_, 1) + "%",
               valueArea, juce::Justification::right);

    g.setColour(meterTrack);
    g.fillRoundedRectangle(area, 4.0f);
    const float normalized = juce::jlimit(0.0f, 1.0f, cpuLoadPercent_ / 100.0f);
    auto fill = area.withWidth(area.getWidth() * normalized);
    g.setColour(cpuLoadPercent_ >= 100.0f ? warning
        : (cpuLoadPercent_ >= 75.0f ? warning : accent));
    g.fillRoundedRectangle(fill, 4.0f);

    // Peak marker remains visible even when the current average has fallen.
    const float peakX = area.getX()
        + area.getWidth() * juce::jlimit(0.0f, 1.0f, cpuPeakPercent_ / 100.0f);
    g.setColour(cpuPeakPercent_ >= 100.0f ? warning : juce::Colours::white);
    g.drawLine(peakX, area.getY() - 1.0f, peakX, area.getBottom() + 1.0f, 1.5f);
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

    auto meterArea = getLocalBounds().toFloat().reduced(30.0f, 0.0f);
    meterArea.setY(155.0f);
    meterArea.setHeight(48.0f);
    const float gap = 18.0f;
    const float meterWidth = (meterArea.getWidth() - 3.0f * gap) / 4.0f;
    drawMeter(g, meterArea.removeFromLeft(meterWidth), "INPUT", inputMeterDb_);
    meterArea.removeFromLeft(gap);
    drawMeter(g, meterArea.removeFromLeft(meterWidth), "OUTPUT", outputMeterDb_);
    meterArea.removeFromLeft(gap);

    auto gateArea = meterArea.removeFromLeft(meterWidth);
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.drawText("GATE", gateArea.removeFromTop(15.0f), juce::Justification::left);
    const auto gateText = gateGain_ > 0.95f ? "OPEN"
        : (gateGain_ > 0.02f ? "CLOSING" : "CLOSED");
    g.setColour(gateGain_ > 0.95f ? accent : juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(12.0f, juce::Font::bold)));
    g.drawText(gateText, gateArea.removeFromTop(17.0f), juce::Justification::left);
    meterArea.removeFromLeft(gap);
    drawCpuMeter(g, meterArea.removeFromLeft(meterWidth));

    auto status = getLocalBounds().reduced(30, 0);
    status.setY(207);
    status.setHeight(18);
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    g.setColour(vadReady_ ? accent : warning);
    g.drawText(vadReady_ ? "SILERO VAD: READY" : "SILERO VAD: NOT AVAILABLE",
               status.removeFromLeft(210), juce::Justification::left);
    g.setColour(muted);
    g.drawText("TARGET PROFILE: " + juce::String(static_cast<juce::int64>(learnedFrames_))
                   + " FRAMES", status, juce::Justification::left);
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

    area.removeFromTop(116);
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

void SmartGateAudioProcessorEditor::timerCallback() {
    const auto readDb = [](float value) noexcept {
        return std::isfinite(value) ? juce::jlimit(-100.0f, 0.0f, value) : -100.0f;
    };
    const float input = readDb(processor_.inputPeakDbForUi_.load(std::memory_order_relaxed));
    const float output = readDb(processor_.outputPeakDbForUi_.load(std::memory_order_relaxed));
    inputMeterDb_ = juce::jmax(input, inputMeterDb_ - 2.0f);
    outputMeterDb_ = juce::jmax(output, outputMeterDb_ - 2.0f);
    gateGain_ = juce::jlimit(0.0f, 1.0f,
        processor_.gateGainForUi_.load(std::memory_order_relaxed));
    learnedFrames_ = processor_.learnedFramesForUi_.load(std::memory_order_relaxed);
    vadReady_ = processor_.vadReadyForUi_.load(std::memory_order_relaxed);
    cpuLoadPercent_ = juce::jlimit(0.0f, 1000.0f,
        processor_.cpuLoadPercentForUi_.load(std::memory_order_relaxed));
    cpuPeakPercent_ = juce::jlimit(0.0f, 1000.0f,
        processor_.cpuPeakPercentForUi_.load(std::memory_order_relaxed));
    repaint();
}
