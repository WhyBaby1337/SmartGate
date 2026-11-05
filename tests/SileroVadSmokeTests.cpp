#include "SileroVadInference.h"
#include "BinaryData.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        smartgate::SileroVadInference inference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        std::array<float, smartgate::SileroVadInference::frameSize> frame{};
        for (int i = 0; i < 12; ++i) {
            const float probability = inference.infer(frame, 1);
            if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
                throw std::runtime_error("Silero returned an invalid probability");
        }

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
