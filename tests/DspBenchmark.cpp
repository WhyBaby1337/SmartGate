#include "AdaptiveKeyboardDetector.h"
#include "AudioFeatures.h"
#include "GateEnvelope.h"
#include "TargetProfile.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {
constexpr double sampleRate = 48000.0;
constexpr std::size_t frameSize = 256;
constexpr std::size_t measuredSeconds = 8;
constexpr std::size_t sampleCount = static_cast<std::size_t>(sampleRate) * measuredSeconds;

struct Result {
    double elapsedMs = 0.0;
    double realtimePercent = 0.0;
};

template <bool targetEnabled, bool keyboardEnabled>
Result runScenario(const std::array<float, sampleCount>& input) {
    smartgate::GateEnvelope gate;
    smartgate::FeatureExtractor features;
    smartgate::AdaptiveKeyboardDetector keyboard;
    smartgate::TargetProfile profile;
    gate.prepare(sampleRate);
    features.prepare(sampleRate);
    keyboard.prepare(sampleRate);

    std::array<float, frameSize> frame{};
    std::size_t framePosition = 0;
    float confidence = 0.0f;
    volatile float outputSink = 0.0f;

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < sampleCount; ++i) {
        const float mono = input[i];
        if constexpr (targetEnabled || keyboardEnabled) {
            frame[framePosition++] = mono;
            if (framePosition == frameSize) {
                if constexpr (keyboardEnabled)
                    confidence = keyboard.processFrame(frame.data(), frame.size());
                if constexpr (targetEnabled) {
                    const auto value = features.process(frame.data(), frame.size());
                    if (value.isFinite()) {
                        profile.learn(value);
                        confidence = profile.isReady() ? profile.confidence(value) : 1.0f;
                    }
                }
                framePosition = 0;
            }
        }
        const float detector = std::abs(mono) * (1.0f - 0.25f * confidence);
        outputSink = outputSink + gate.process(detector);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    (void) outputSink;
    return {elapsed, elapsed / (static_cast<double>(measuredSeconds) * 1000.0) * 100.0};
}
}

int main() {
    std::array<float, sampleCount> input{};
    // Prepare deterministic speech-like test material outside the timed region.
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        const double envelope = 0.25 + 0.20 * std::sin(2.0 * 3.141592653589793 * 2.3 * t);
        input[i] = static_cast<float>(envelope
            * (0.65 * std::sin(2.0 * 3.141592653589793 * 180.0 * t)
             + 0.25 * std::sin(2.0 * 3.141592653589793 * 1100.0 * t)
             + 0.10 * std::sin(2.0 * 3.141592653589793 * 4200.0 * t)));
    }

    const auto gateOnly = runScenario<false, false>(input);
    const auto target = runScenario<true, false>(input);
    const auto targetAndKeyboard = runScenario<true, true>(input);

    std::cout << "SmartGate DSP microbenchmark (48 kHz mono analysis, 8 s simulated audio)\n"
              << "Gate only: " << gateOnly.elapsedMs << " ms, "
              << gateOnly.realtimePercent << "% of one logical core\n"
              << "Target features + profile: " << target.elapsedMs << " ms, "
              << target.realtimePercent << "% of one logical core\n"
              << "Target + keyboard classifier: " << targetAndKeyboard.elapsedMs << " ms, "
              << targetAndKeyboard.realtimePercent << "% of one logical core\n"
              << "Note: excludes JUCE host overhead and any neural VAD inference.\n";
    return 0;
}
