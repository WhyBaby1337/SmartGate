#include "VadResampler.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void requireResampler(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runVadResamplerTests() {
    smartgate::VadResampler resampler;
    requireResampler(!resampler.prepare(8000.0, 16000.0),
                     "VAD resampler must reject unsupported upsampling");
    requireResampler(!resampler.prepare(0.0, 16000.0),
                     "VAD resampler must reject invalid rates");
    requireResampler(resampler.prepare(48000.0, 16000.0),
                     "48 kHz to 16 kHz preparation should succeed");

    int outputCount = 0;
    float output = 0.0f;
    for (int i = 0; i < 4800; ++i) {
        const float input = static_cast<float>(i) / 4800.0f;
        if (resampler.processSample(input, output)) {
            ++outputCount;
            requireResampler(std::isfinite(output), "Resampled output must be finite");
        }
    }
    requireResampler(outputCount == 1599,
                     "Streaming 48 kHz conversion should emit one sample per three input intervals");

    resampler.reset();
    outputCount = 0;
    // Feed in two conceptual host blocks without resetting the converter.
    for (int i = 0; i < 2205; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    for (int i = 2205; i < 4410; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    requireResampler(outputCount == 1599,
                     "44.1 kHz conversion must preserve phase across host block boundaries");

    resampler.reset();
    requireResampler(!resampler.processSample(0.5f, output),
                     "First sample after reset establishes interpolation history");
    (void) resampler.processSample(std::numeric_limits<float>::quiet_NaN(), output);
    (void) resampler.processSample(0.0f, output);
    requireResampler(resampler.processSample(0.0f, output) && std::isfinite(output),
                     "Non-finite input must be sanitized before interpolation");
}
