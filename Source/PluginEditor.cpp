#include "PluginEditor.h"
#include "PluginProcessor.h"

#include <algorithm>
#include <cmath>

namespace {
const auto background = juce::Colour(0xff101318);
const auto panel = juce::Colour(0xff191f27);
const auto panelRaised = juce::Colour(0xff202833);
const auto accent = juce::Colour(0xff65d6b0);
const auto muted = juce::Colour(0xff8b98a8);
const auto meterTrack = juce::Colour(0xff29333e);
const auto warning = juce::Colour(0xffffc36b);

void drawSectionTitle(juce::Graphics& g, juce::Rectangle<int> area,
                      const juce::String& title, const juce::String& detail) {
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(12.0f, juce::Font::bold)));
    g.drawText(title, area.removeFromLeft(190), juce::Justification::centredLeft);
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f)));
    g.drawText(detail, area, juce::Justification::centredLeft);
}
}

SmartGateAudioProcessorEditor::SmartGateAudioProcessorEditor(
    SmartGateAudioProcessor& processor)
    : AudioProcessorEditor(&processor), processor_(processor) {
    setSize(900, 760);
    setResizable(true, true);
    setResizeLimits(760, 680, 1200, 960);

    for (std::size_t i = 0; i < sliders_.size(); ++i) {
        const juce::String suffix = (i == 0 || i == 4 || i == 5)
            ? " dB" : (i >= 6 ? "" : " ms");
        configureSlider(i, displayNames_[i], suffix);
        sliderAttachments_[i] = std::make_unique<SliderAttachment>(
            processor_.parameters_, parameterNames_[i], sliders_[i]);
    }

    aiButton_.setTooltip("Enables Silero voice activity assistance and target-profile influence. Off keeps the conventional gate active.");
    learnButton_.setTooltip("Start or pause voice-profile learning. Pausing preserves all collected frames.");
    resetProfileButton_.setTooltip("Permanently clear the learned profile and start a fresh training session.");
    keyboardModeBox_.setTooltip("Choose the keyboard-noise profile that best matches your keyboard.");
    aiButton_.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    learnButton_.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    aiButton_.setColour(juce::ToggleButton::tickColourId, accent);
    learnButton_.setColour(juce::ToggleButton::tickColourId, accent);
    addAndMakeVisible(aiButton_);
    addAndMakeVisible(learnButton_);
    resetProfileButton_.setColour(juce::TextButton::buttonColourId, panelRaised);
    resetProfileButton_.setColour(juce::TextButton::textColourOffId, warning);
    addAndMakeVisible(resetProfileButton_);
    resetProfileButton_.onClick = [this] { confirmProfileReset(); };
    aiAttachment_ = std::make_unique<ButtonAttachment>(
        processor_.parameters_, "aiEnabled", aiButton_);
    learnAttachment_ = std::make_unique<ButtonAttachment>(
        processor_.parameters_, "learnTarget", learnButton_);

    keyboardModeBox_.addItemList({"Universal", "Mechanical", "Membrane"}, 1);
    keyboardModeBox_.setColour(juce::ComboBox::backgroundColourId, panelRaised);
    keyboardModeBox_.setColour(juce::ComboBox::textColourId, juce::Colours::white);
    keyboardModeBox_.setColour(juce::ComboBox::outlineColourId, meterTrack);
    addAndMakeVisible(keyboardModeBox_);
    keyboardModeAttachment_ = std::make_unique<ComboAttachment>(
        processor_.parameters_, "keyboardMode", keyboardModeBox_);

    keyboardModeLabel_.setText("KEYBOARD PROFILE", juce::dontSendNotification);
    keyboardModeLabel_.setColour(juce::Label::textColourId, muted);
    keyboardModeLabel_.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    addAndMakeVisible(keyboardModeLabel_);

    exportProfileButton_.setTooltip("Save the learned voice profile to a standalone .sgprofile backup file.");
    importProfileButton_.setTooltip("Load a previously exported .sgprofile backup. The current learned profile will be replaced.");
    for (auto* button : {&exportProfileButton_, &importProfileButton_}) {
        button->setColour(juce::TextButton::buttonColourId, panelRaised);
        button->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
        button->setColour(juce::TextButton::textColourOnId, juce::Colours::white);
        addAndMakeVisible(button);
    }
    exportProfileButton_.onClick = [this] { chooseProfileExport(); };
    importProfileButton_.onClick = [this] { chooseProfileImport(); };

    profileFileStatus_.setText("PROFILE BACKUP  ·  .sgprofile", juce::dontSendNotification);
    profileFileStatus_.setColour(juce::Label::textColourId, muted);
    profileFileStatus_.setFont(juce::Font(juce::FontOptions(9.0f)));
    profileFileStatus_.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(profileFileStatus_);

    timerCallback();
    startTimerHz(30);
}

void SmartGateAudioProcessorEditor::configureSlider(
    std::size_t index, const juce::String& label, const juce::String& suffix) {
    auto& slider = sliders_[index];
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 82, 20);
    slider.setTextValueSuffix(suffix);
    if (index >= 6) {
        slider.textFromValueFunction = [](double value) {
            return juce::String(juce::roundToInt(value * 100.0)) + " %";
        };
        slider.valueFromTextFunction = [](const juce::String& text) {
            auto numeric = text;
            numeric.retainCharacters("0123456789.-");
            return numeric.getDoubleValue() / 100.0;
        };
    }
    static const std::array<const char*, 10> tooltips{
        "Input level below which the gate closes. Start around the room-noise level.",
        "How quickly the gate opens after voice is detected.",
        "How smoothly the gate closes after the voice ends.",
        "Minimum time the gate stays open after the signal falls below threshold.",
        "Difference between opening and closing thresholds; helps prevent chatter.",
        "How much the signal is attenuated while the gate is closed.",
        "How strongly the learned target voice profile influences gating. Set to 0 to disable profile influence.",
        "Amount of keyboard-noise suppression. This works independently of the AI Engine toggle.",
        "Adds phrase-aware hold time so quiet consonants are less likely to be cut off.",
        "Conservatively attenuates brief high-frequency mouth clicks. It does not remove sustained breaths or nasal noise."
    };
    slider.setTooltip(tooltips[index]);
    slider.setColour(juce::Slider::rotarySliderFillColourId, accent);
    slider.setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff35404d));
    slider.setColour(juce::Slider::thumbColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxTextColourId, juce::Colours::white);
    slider.setColour(juce::Slider::textBoxBackgroundColourId, panelRaised);
    slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    addAndMakeVisible(slider);

    auto& title = labels_[index];
    title.setText(label, juce::dontSendNotification);
    title.setJustificationType(juce::Justification::centred);
    title.setColour(juce::Label::textColourId, muted);
    title.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    addAndMakeVisible(title);
}

void SmartGateAudioProcessorEditor::layoutControl(
    std::size_t index, juce::Rectangle<int> area) {
    labels_[index].setBounds(area.removeFromTop(19));
    sliders_[index].setBounds(area.reduced(3, 0));
}

void SmartGateAudioProcessorEditor::drawMeter(
    juce::Graphics& g, juce::Rectangle<float> area,
    const juce::String& label, float levelDb) const {
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.drawText(label, area.removeFromTop(15.0f), juce::Justification::left);
    auto valueArea = area.removeFromTop(18.0f);
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

    const auto valueArea = area.removeFromTop(18.0f);
    g.setColour(cpuPeakPercent_ >= 100.0f ? warning : juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(11.0f)));
    g.drawText(juce::String(cpuLoadPercent_, 1) + "%  /  PEAK "
                   + juce::String(cpuPeakPercent_, 1) + "%",
               valueArea, juce::Justification::right);

    g.setColour(meterTrack);
    g.fillRoundedRectangle(area, 4.0f);
    const float normalized = juce::jlimit(0.0f, 1.0f, cpuLoadPercent_ / 100.0f);
    auto fill = area.withWidth(area.getWidth() * normalized);
    g.setColour(cpuLoadPercent_ >= 75.0f ? warning : accent);
    g.fillRoundedRectangle(fill, 4.0f);

    const float peakX = area.getX()
        + area.getWidth() * juce::jlimit(0.0f, 1.0f, cpuPeakPercent_ / 100.0f);
    g.setColour(cpuPeakPercent_ >= 100.0f ? warning : juce::Colours::white);
    g.drawLine(peakX, area.getY() - 1.0f, peakX, area.getBottom() + 1.0f, 1.5f);
}

void SmartGateAudioProcessorEditor::paint(juce::Graphics& g) {
    g.fillAll(background);
    auto area = getLocalBounds().reduced(24);

    auto header = area.removeFromTop(66);
    g.setColour(accent);
    g.fillRoundedRectangle(juce::Rectangle<float>(
        static_cast<float>(header.getX()), static_cast<float>(header.getY() + 9),
        4.0f, 38.0f), 2.0f);
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(27.0f, juce::Font::bold)));
    g.drawText("SMART GATE", header.getX() + 16, header.getY() + 3, 280, 32,
               juce::Justification::left);
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(11.0f)));
    g.drawText("INTELLIGENT VOICE & NOISE CONTROL",
               header.getX() + 18, header.getY() + 35, 360, 18,
               juce::Justification::left);

    auto modeCard = area.removeFromTop(58);
    g.setColour(panel);
    g.fillRoundedRectangle(modeCard.toFloat(), 10.0f);

    auto meterCard = area.removeFromTop(78);
    g.setColour(panel);
    g.fillRoundedRectangle(meterCard.toFloat(), 10.0f);
    auto meters = meterCard.toFloat().reduced(14.0f, 10.0f);
    const float gap = 18.0f;
    const float meterWidth = (meters.getWidth() - 3.0f * gap) / 4.0f;
    drawMeter(g, meters.removeFromLeft(meterWidth), "INPUT", inputMeterDb_);
    meters.removeFromLeft(gap);
    drawMeter(g, meters.removeFromLeft(meterWidth), "OUTPUT", outputMeterDb_);
    meters.removeFromLeft(gap);

    auto gateArea = meters.removeFromLeft(meterWidth);
    g.setColour(muted);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.drawText("GATE", gateArea.removeFromTop(15.0f), juce::Justification::left);
    const auto gateText = gateGain_ > 0.95f ? "OPEN"
        : (gateGain_ > 0.02f ? "CLOSING" : "CLOSED");
    g.setColour(gateGain_ > 0.95f ? accent : juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(13.0f, juce::Font::bold)));
    g.drawText(gateText, gateArea.removeFromTop(18.0f), juce::Justification::left);
    meters.removeFromLeft(gap);
    drawCpuMeter(g, meters.removeFromLeft(meterWidth));

    auto status = area.removeFromTop(28);
    g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
    g.setColour(vadReady_ ? accent : warning);
    g.drawText(vadReady_ ? "●  VOICE AI READY" : "●  VOICE AI UNAVAILABLE",
               status.removeFromLeft(205), juce::Justification::centredLeft);
    g.setColour(muted);
    g.drawText("TARGET PROFILE  /  "
                   + juce::String(static_cast<juce::int64>(learnedFrames_)) + " FRAMES",
               status, juce::Justification::centredLeft);

    auto gatePanel = area.removeFromTop(static_cast<int>((area.getHeight() - 10) * 0.67f));
    auto intelligencePanel = area;
    g.setColour(panel);
    g.fillRoundedRectangle(gatePanel.toFloat(), 10.0f);
    g.fillRoundedRectangle(intelligencePanel.toFloat(), 10.0f);

    auto gateHeading = gatePanel.reduced(14, 8).removeFromTop(22);
    drawSectionTitle(g, gateHeading, "GATE CONTROLS",
                     "Shape how the gate opens, holds and closes");
    auto intelligenceHeading = intelligencePanel.reduced(14, 8).removeFromTop(22);
    drawSectionTitle(g, intelligenceHeading, "SMART PROCESSING",
                     "Target voice matching and keyboard suppression");

    if (!aiButton_.getToggleState()) {
        g.setColour(muted);
        g.setFont(juce::Font(juce::FontOptions(9.0f)));
        g.drawText("AI OFF  ·  conventional gate mode",
                   intelligencePanel.getX() + 14, intelligencePanel.getBottom() - 18,
                   intelligencePanel.getWidth() - 28, 12, juce::Justification::right);
    }
}

void SmartGateAudioProcessorEditor::resized() {
    auto area = getLocalBounds().reduced(24);
    area.removeFromTop(66);

    auto modeArea = area.removeFromTop(58).reduced(12, 5);
    aiButton_.setBounds(modeArea.removeFromLeft(125).reduced(4, 5));
    learnButton_.setBounds(modeArea.removeFromLeft(155).reduced(4, 5));
    resetProfileButton_.setBounds(modeArea.removeFromLeft(125).reduced(4, 5));
    auto keyboardArea = modeArea.removeFromLeft(205);
    keyboardModeLabel_.setBounds(keyboardArea.removeFromTop(15));
    keyboardModeBox_.setBounds(keyboardArea.removeFromTop(29));

    area.removeFromTop(78);
    area.removeFromTop(28);

    const int sectionGap = 10;
    auto gatePanel = area.removeFromTop(static_cast<int>((area.getHeight() - sectionGap) * 0.58f));
    auto intelligencePanel = area;
    gatePanel = gatePanel.reduced(12, 30);
    intelligencePanel = intelligencePanel.reduced(12, 8);

    const int columnGap = 8;
    const int cellWidth = (gatePanel.getWidth() - 2 * columnGap) / 3;
    const int rowGap = 3;
    const int cellHeight = (gatePanel.getHeight() - rowGap) / 2;
    for (std::size_t i = 0; i < 6; ++i) {
        const int row = static_cast<int>(i / 3);
        const int column = static_cast<int>(i % 3);
        auto cell = juce::Rectangle<int>(
            gatePanel.getX() + column * (cellWidth + columnGap),
            gatePanel.getY() + row * (cellHeight + rowGap),
            cellWidth, cellHeight);
        layoutControl(i, cell);
    }

    // Keep the section title, three full-size controls, and backup actions
    // in separate vertical bands. Previously the whole panel was reduced by
    // 30 px at both ends before reserving the buttons, leaving the smart
    // sliders only ~70 px tall and making their labels/readouts look cramped.
    auto smartContent = intelligencePanel.reduced(12, 8);
    smartContent.removeFromTop(38); // reserve a clear band below the section heading
    auto profileActions = smartContent.removeFromBottom(34);
    exportProfileButton_.setBounds(profileActions.removeFromLeft(142));
    profileActions.removeFromLeft(8);
    importProfileButton_.setBounds(profileActions.removeFromLeft(142));
    profileActions.removeFromLeft(12);
    profileFileStatus_.setBounds(profileActions);

    const int smartColumns = 4;
    const int smartCellWidth = (smartContent.getWidth()
        - (smartColumns - 1) * columnGap) / smartColumns;
    for (std::size_t i = 6; i < sliders_.size(); ++i) {
        const int column = static_cast<int>(i - 6);
        auto cell = juce::Rectangle<int>(
            smartContent.getX() + column * (smartCellWidth + columnGap),
            smartContent.getY(), smartCellWidth, smartContent.getHeight());
        layoutControl(i, cell);
    }
}

void SmartGateAudioProcessorEditor::confirmProfileReset() {
    juce::AlertWindow::showAsync(
        juce::MessageBoxOptions()
            .withIconType(juce::MessageBoxIconType::WarningIcon)
            .withTitle("Reset target profile?")
            .withMessage("This will permanently clear the collected FRAMES. "
                         "You can then start a completely new learning session.")
            .withButton("RESET")
            .withButton("CANCEL"),
        [this](int result) {
            if (result == 1) {
                processor_.requestTargetProfileReset();
                profileFileStatus_.setText("PROFILE RESET  ·  ready for new learning",
                                           juce::dontSendNotification);
                profileFileStatus_.setColour(juce::Label::textColourId, warning);
            }
        });
}

void SmartGateAudioProcessorEditor::chooseProfileExport() {
    profileFileChooser_ = std::make_unique<juce::FileChooser>(
        "Export Smart Gate target profile",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
            .getChildFile("SmartGate-Voice-Profile.sgprofile"),
        "*.sgprofile");
    profileFileChooser_->launchAsync(
        juce::FileBrowserComponent::saveMode
            | juce::FileBrowserComponent::canSelectFiles
            | juce::FileBrowserComponent::warnAboutOverwriting,
        [this](const juce::FileChooser& chooser) {
            auto file = chooser.getResult();
            if (file == juce::File{})
                return;
            if (file.getFileExtension().isEmpty())
                file = file.withFileExtension(".sgprofile");
            const bool saved = processor_.exportTargetProfile(file);
            profileFileStatus_.setText(saved
                ? "PROFILE EXPORTED  ·  " + file.getFileName()
                : "EXPORT FAILED  ·  no valid learned profile",
                juce::dontSendNotification);
            profileFileStatus_.setColour(juce::Label::textColourId,
                saved ? accent : warning);
        });
}

void SmartGateAudioProcessorEditor::chooseProfileImport() {
    profileFileChooser_ = std::make_unique<juce::FileChooser>(
        "Import Smart Gate target profile",
        juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
        "*.sgprofile");
    profileFileChooser_->launchAsync(
        juce::FileBrowserComponent::openMode
            | juce::FileBrowserComponent::canSelectFiles,
        [this](const juce::FileChooser& chooser) {
            const auto file = chooser.getResult();
            if (file == juce::File{})
                return;
            const bool loaded = processor_.importTargetProfile(file);
            profileFileStatus_.setText(loaded
                ? "PROFILE IMPORTED  ·  applied on next audio block"
                : "IMPORT FAILED  ·  invalid or unsupported file",
                juce::dontSendNotification);
            profileFileStatus_.setColour(juce::Label::textColourId,
                loaded ? accent : warning);
        });
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
    learnButton_.setButtonText(learnButton_.getToggleState()
        ? "PAUSE LEARNING"
        : (learnedFrames_ > 0 ? "RESUME LEARNING" : "START LEARNING"));
    vadReady_ = processor_.vadReadyForUi_.load(std::memory_order_relaxed);
    cpuLoadPercent_ = juce::jlimit(0.0f, 1000.0f,
        processor_.cpuLoadPercentForUi_.load(std::memory_order_relaxed));
    cpuPeakPercent_ = juce::jlimit(0.0f, 1000.0f,
        processor_.cpuPeakPercentForUi_.load(std::memory_order_relaxed));
    repaint();
}
