#include "SileroVadInference.h"
#include "BinaryData.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        smartgate::SileroVadInference inference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        std::array<float, smartgate::SileroVadInference::frameSize> frame{};

        // Warm up the ONNX session before collecting timings. A 512-sample
        // Silero frame represents 32 ms of audio at 16 kHz.
        for (int i = 0; i < 12; ++i) {
            const float probability = inference.infer(frame, 1);
            if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
                throw std::runtime_error("Silero returned an invalid probability");
        }

        constexpr int measuredFrames = 128;
        std::vector<double> latencyMicroseconds;
        latencyMicroseconds.reserve(measuredFrames);
        double totalMicroseconds = 0.0;
        for (int i = 0; i < measuredFrames; ++i) {
            const auto start = std::chrono::steady_clock::now();
            const float probability = inference.infer(frame, 1);
            const auto elapsed = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
            if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
                throw std::runtime_error("Silero returned an invalid probability during timing");
            latencyMicroseconds.push_back(elapsed);
            totalMicroseconds += elapsed;
        }

        std::sort(latencyMicroseconds.begin(), latencyMicroseconds.end());
        const auto percentile = [&latencyMicroseconds](double fraction) {
            const auto index = static_cast<std::size_t>(
                std::ceil(fraction * static_cast<double>(latencyMicroseconds.size()))) - 1;
            return latencyMicroseconds[std::min(index, latencyMicroseconds.size() - 1)];
        };
        constexpr double frameBudgetMicroseconds = 32000.0;
        std::cout << "Silero ONNX inference latency (128 warmed CPU frames): "
                  << "mean=" << totalMicroseconds / measuredFrames
                  << " us, p50=" << percentile(0.50)
                  << " us, p95=" << percentile(0.95)
                  << " us, max=" << latencyMicroseconds.back()
                  << " us; 32 ms frame budget="
                  << frameBudgetMicroseconds << " us.\n";

        const float afterReset = inference.infer(frame, 2);
        if (!std::isfinite(afterReset) || afterReset < 0.0f || afterReset > 1.0f)
            throw std::runtime_error("Silero failed after a generation reset");

        for (std::size_t i = 0; i < frame.size(); ++i) {
            const double t = static_cast<double>(i) / 16000.0;
            frame[i] = static_cast<float>(0.2 * std::sin(
                2.0 * 3.141592653589793 * 220.0 * t));
        }
        const float voiceBand = inference.infer(frame, 2);
        if (!std::isfinite(voiceBand) || voiceBand < 0.0f || voiceBand > 1.0f)
            throw std::runtime_error("Silero failed to process a finite speech-band frame");

        std::cout << "Silero ONNX smoke test passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SILERO TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
