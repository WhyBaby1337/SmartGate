#include "PluginProcessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {
constexpr double sampleRate = 48000.0;
constexpr std::uint64_t measuredSeconds = 8;
constexpr std::uint64_t measuredSamples =
    static_cast<std::uint64_t>(sampleRate) * measuredSeconds;

void setParameter(SmartGateAudioProcessor& processor, const juce::String& id,
                  float plainValue) {
    for (auto* parameter : processor.getParameters()) {
        auto* withId = dynamic_cast<juce::AudioProcessorParameterWithID*>(parameter);
        auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameter);
        if (withId != nullptr && ranged != nullptr && withId->paramID == id) {
            parameter->setValueNotifyingHost(ranged->convertTo0to1(plainValue));
            return;
        }
    }
}

struct Result {
    double averagePercent = 0.0;
    double peakBlockPercent = 0.0;
};

Result runScenario(const std::string& name, int blockSize, bool aiEnabled,
                   float keyboardReduction, bool mouseEnabled,
                   float mouseReduction, bool mouthEnabled,
                   float mouthReduction) {
    SmartGateAudioProcessor processor;
    processor.prepareToPlay(sampleRate, blockSize);
    setParameter(processor, "aiEnabled", aiEnabled ? 1.0f : 0.0f);
    setParameter(processor, "keyboardReduction", keyboardReduction);
    setParameter(processor, "mouseClickSuppressionEnabled", mouseEnabled ? 1.0f : 0.0f);
    setParameter(processor, "mouseClickReduction", mouseReduction);
    setParameter(processor, "mouthClickSuppressionEnabled", mouthEnabled ? 1.0f : 0.0f);
    setParameter(processor, "mouthNoiseReduction", mouthReduction);

    juce::AudioBuffer<float> buffer(2, blockSize);
    juce::MidiBuffer midi;
    std::uint64_t samplePosition = 0;
    double elapsedSeconds = 0.0;
    double peakBlockPercent = 0.0;
    constexpr double pi = 3.14159265358979323846;

    // Warm up for two seconds independent of the selected host block size.
    const int warmupBlocks = static_cast<int>(2.0 * sampleRate / blockSize);
    for (int block = 0; block < warmupBlocks; ++block) {
        for (int channel = 0; channel < 2; ++channel) {
            auto* data = buffer.getWritePointer(channel);
            for (int i = 0; i < blockSize; ++i) {
                const double t = static_cast<double>(samplePosition + i) / sampleRate;
                const double envelope = 0.22 + 0.08 * std::sin(2.0 * pi * 2.1 * t);
                data[i] = static_cast<float>(envelope *
                    (0.70 * std::sin(2.0 * pi * 180.0 * t)
                   + 0.20 * std::sin(2.0 * pi * 1300.0 * t)
                   + 0.10 * std::sin(2.0 * pi * 5200.0 * t)));
            }
        }
        processor.processBlock(buffer, midi);
        samplePosition += blockSize;
    }

    samplePosition = 0;
    const std::uint64_t blockCount = measuredSamples / blockSize;
    for (std::uint64_t block = 0; block < blockCount; ++block) {
        for (int channel = 0; channel < 2; ++channel) {
            auto* data = buffer.getWritePointer(channel);
            for (int i = 0; i < blockSize; ++i) {
                const double t = static_cast<double>(samplePosition + i) / sampleRate;
                const double envelope = 0.22 + 0.08 * std::sin(2.0 * pi * 2.1 * t);
                data[i] = static_cast<float>(envelope *
                    (0.70 * std::sin(2.0 * pi * 180.0 * t)
                   + 0.20 * std::sin(2.0 * pi * 1300.0 * t)
                   + 0.10 * std::sin(2.0 * pi * 5200.0 * t)));
            }
        }

        const auto start = std::chrono::steady_clock::now();
        processor.processBlock(buffer, midi);
        const auto end = std::chrono::steady_clock::now();
        const double blockElapsed =
            std::chrono::duration<double>(end - start).count();
        elapsedSeconds += blockElapsed;
        const double blockBudget = static_cast<double>(blockSize) / sampleRate;
        peakBlockPercent = std::max(peakBlockPercent,
            100.0 * blockElapsed / blockBudget);
        samplePosition += blockSize;
    }

    processor.releaseResources();
    const double audioSeconds = static_cast<double>(measuredSamples) / sampleRate;
    Result result{100.0 * elapsedSeconds / audioSeconds, peakBlockPercent};
    std::cout << name << ": average=" << result.averagePercent
              << "% of one logical core, max block=" << result.peakBlockPercent
              << "% (48 kHz stereo, " << blockSize << " samples, " << measuredSeconds << " s)\n";
    return result;
}
}

int main() {
    std::cout << "SmartGate full AudioProcessor processBlock benchmark\n"
              << "Includes lookahead, gate, feature analysis and all transient DSP; "
              << "Silero worker inference is measured separately.\n";
    for (const int blockSize : {64, 128, 256}) {
        std::cout << "\nHost block size: " << blockSize << " samples\n";
        runScenario("Factory defaults", blockSize, true, 0.0f, true, 0.20f, true, 0.25f);
        runScenario("All transient suppressors active", blockSize, true, 0.65f, true, 0.20f, true, 0.25f);
        runScenario("All transient suppressors active, AI off", blockSize, false, 0.65f, true, 0.20f, true, 0.25f);
    }
    return 0;
}
