#include "VadResampler.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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

    // A failed re-prepare must invalidate the old configuration instead of
    // silently continuing to process at the previous sample rate.
    requireResampler(!resampler.prepare(8000.0, 16000.0) && !resampler.isPrepared(),
                     "A failed re-prepare must leave the converter unprepared");
    output = 123.0f;
    requireResampler(!resampler.processSample(0.25f, output) && output == 123.0f,
                     "An unprepared converter must not emit or modify output");

    requireResampler(resampler.prepare(44100.0, 16000.0),
                     "44.1 kHz to 16 kHz preparation should succeed");
    outputCount = 0;
    // Feed in two conceptual host blocks without resetting the converter.
    for (int i = 0; i < 2205; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    for (int i = 2205; i < 4410; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    if (outputCount != 1599)
        throw std::runtime_error("44.1 kHz conversion emitted " + std::to_string(outputCount)
                                 + " samples; expected 1599 across host block boundaries");

    resampler.reset();
    requireResampler(!resampler.processSample(0.5f, output),
                     "First sample after reset establishes interpolation history");
    (void) resampler.processSample(std::numeric_limits<float>::quiet_NaN(), output);
    (void) resampler.processSample(0.0f, output);
    requireResampler(resampler.processSample(0.0f, output) && std::isfinite(output),
                     "Non-finite input must be sanitized before interpolation");
}
