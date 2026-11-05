#include "SileroVadInference.h"
#include "BinaryData.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Inference = smartgate::SileroVadInference;
using Frame = std::array<float, Inference::frameSize>;

struct LatencySummary {
    double meanMicroseconds = 0.0;
    double p50Microseconds = 0.0;
    double p95Microseconds = 0.0;
    double maxMicroseconds = 0.0;
    float lastProbability = 0.0f;
};

LatencySummary measureCase(Inference& inference, const Frame& frame,
                           std::uint64_t generation, const char* label) {
    const auto successfulBefore = inference.successfulInferenceCount();
    // Each case starts a fresh recurrent stream, then warms that stream before
    // timing. This keeps silence, tonal speech-like input, and noise separate.
    for (int i = 0; i < 12; ++i) {
        const float probability = inference.infer(frame, generation);
        if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
            throw std::runtime_error("Silero returned an invalid warmup probability");
    }

    constexpr int measuredFrames = 128;
    std::vector<double> timings;
    timings.reserve(measuredFrames);
    double totalMicroseconds = 0.0;
    float lastProbability = 0.0f;
    for (int i = 0; i < measuredFrames; ++i) {
        const auto start = std::chrono::steady_clock::now();
        lastProbability = inference.infer(frame, generation);
        const auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        if (!std::isfinite(lastProbability)
            || lastProbability < 0.0f || lastProbability > 1.0f)
            throw std::runtime_error("Silero returned an invalid probability during timing");
        timings.push_back(elapsed);
        totalMicroseconds += elapsed;
    }

    std::sort(timings.begin(), timings.end());
    const auto percentile = [&timings](double fraction) {
        const auto index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(timings.size()))) - 1;
        return timings[std::min(index, timings.size() - 1)];
    };

    LatencySummary summary;
    summary.meanMicroseconds = totalMicroseconds / measuredFrames;
    summary.p50Microseconds = percentile(0.50);
    summary.p95Microseconds = percentile(0.95);
    summary.maxMicroseconds = timings.back();
    summary.lastProbability = lastProbability;

    if (inference.successfulInferenceCount() != successfulBefore + 12 + measuredFrames)
        throw std::runtime_error("Silero inference did not complete every ONNX Run() successfully");

    std::cout << "Silero ONNX latency [" << label << "] (128 warmed CPU frames): "
              << "mean=" << summary.meanMicroseconds
              << " us, p50=" << summary.p50Microseconds
              << " us, p95=" << summary.p95Microseconds
              << " us, max=" << summary.maxMicroseconds
              << " us, last_probability=" << summary.lastProbability
              << "; 32 ms frame budget=32000 us.\n";
    return summary;
}
}

int main() {
    try {
        Inference inference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));

        Frame silence{};
        measureCase(inference, silence, 1, "silence");

        Frame speechLike{};
        for (std::size_t i = 0; i < speechLike.size(); ++i) {
            const double t = static_cast<double>(i) / 16000.0;
            const double envelope = 0.75 + 0.25 * std::sin(
                2.0 * 3.14159265358979323846 * 3.0 * t);
            speechLike[i] = static_cast<float>(envelope * (
                0.12 * std::sin(2.0 * 3.14159265358979323846 * 180.0 * t)
              + 0.06 * std::sin(2.0 * 3.14159265358979323846 * 360.0 * t)
              + 0.025 * std::sin(2.0 * 3.14159265358979323846 * 720.0 * t)
              + 0.015 * std::sin(2.0 * 3.14159265358979323846 * 1200.0 * t)));
        }
        measureCase(inference, speechLike, 2, "speech-like-tones");

        Frame deterministicNoise{};
        std::uint32_t randomState = 0x13579BDFu;
        for (auto& sample : deterministicNoise) {
            randomState = randomState * 1664525u + 1013904223u;
            const auto centered = static_cast<std::int32_t>(randomState >> 8)
                                - static_cast<std::int32_t>(0x7FFFFFu);
            sample = static_cast<float>(centered / 8388608.0 * 0.08);
        }
        measureCase(inference, deterministicNoise, 3, "deterministic-noise");

        // A generation change must reset recurrent state and remain usable.
        const auto beforeGenerationReset = inference.successfulInferenceCount();
        const float afterReset = inference.infer(silence, 4);
        if (!std::isfinite(afterReset) || afterReset < 0.0f || afterReset > 1.0f)
            throw std::runtime_error("Silero failed after a generation reset");

        const float speechBand = inference.infer(speechLike, 4);
        if (!std::isfinite(speechBand) || speechBand < 0.0f || speechBand > 1.0f)
            throw std::runtime_error("Silero failed to process a finite speech-band frame");
        if (inference.successfulInferenceCount() != beforeGenerationReset + 2)
            throw std::runtime_error("Silero generation-reset frames did not complete ONNX inference");

        std::cout << "Silero ONNX smoke test passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SILERO TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
