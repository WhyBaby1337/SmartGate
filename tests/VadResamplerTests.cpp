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

double measureRms(smartgate::VadResampler& resampler, double frequency,
                  double sampleRate, int sampleCount) {
    double sumSquares = 0.0;
    int outputCount = 0;
    float output = 0.0f;
    for (int i = 0; i < sampleCount; ++i) {
        const float input = static_cast<float>(
            std::sin(2.0 * 3.14159265358979323846 * frequency
                     * static_cast<double>(i) / sampleRate));
        if (resampler.processSample(input, output) && i > 200) {
            sumSquares += static_cast<double>(output) * output;
            ++outputCount;
        }
    }
    return outputCount > 0 ? std::sqrt(sumSquares / outputCount) : 0.0;
}
}

void runVadResamplerTests() {
    smartgate::VadResampler resampler;
    requireResampler(!resampler.prepare(8000.0, 16000.0),
                     "VAD resampler must reject unsupported upsampling");
    requireResampler(!resampler.prepare(0.0, 16000.0),
                     "VAD resampler must reject invalid rates");
    requireResampler(!resampler.prepare(192000.0, 16000.0),
                     "VAD resampler must reject rates beyond its real-time FIR design limit");
    requireResampler(!resampler.prepare(std::numeric_limits<double>::quiet_NaN(), 16000.0),
                     "VAD resampler must reject a non-finite input rate");
    requireResampler(!resampler.prepare(std::numeric_limits<double>::infinity(), 16000.0),
                     "VAD resampler must reject an infinite input rate");
    requireResampler(!resampler.prepare(48000.0, std::numeric_limits<double>::quiet_NaN()),
                     "VAD resampler must reject a non-finite output rate");
    requireResampler(!resampler.prepare(48000.0, 0.0),
                     "VAD resampler must reject a zero output rate");
    requireResampler(!resampler.prepare(48000.0, -1.0),
                     "VAD resampler must reject a negative output rate");
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
    requireResampler(outputCount == 1600,
                     "Streaming 48 kHz conversion should emit one sample per three input intervals");

    requireResampler(!resampler.prepare(8000.0, 16000.0) && !resampler.isPrepared(),
                     "A failed re-prepare must leave the converter unprepared");
    output = 123.0f;
    requireResampler(!resampler.processSample(0.25f, output) && output == 123.0f,
                     "An unprepared converter must not emit or modify output");

    requireResampler(resampler.prepare(44100.0, 16000.0),
                     "44.1 kHz to 16 kHz preparation should succeed");
    outputCount = 0;
    for (int i = 0; i < 2205; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    for (int i = 2205; i < 4410; ++i)
        if (resampler.processSample(std::sin(static_cast<float>(i) * 0.01f), output))
            ++outputCount;
    if (outputCount != 1600)
        throw std::runtime_error("44.1 kHz conversion emitted " + std::to_string(outputCount)
                                 + " samples; expected 1600 across host block boundaries");

    // Equal-rate conversion needs no interpolation history. In particular,
    // the first input sample must not be lost before the first VAD frame.
    requireResampler(resampler.prepare(16000.0, 16000.0),
                     "Equal-rate 16 kHz conversion should prepare");
    outputCount = 0;
    for (int i = 0; i < 512; ++i) {
        const float input = static_cast<float>(i) / 512.0f;
        requireResampler(resampler.processSample(input, output),
                         "Equal-rate conversion should emit every input sample");
        requireResampler(output == input,
                         "Equal-rate conversion must preserve samples exactly");
        ++outputCount;
    }
    requireResampler(outputCount == 512,
                     "One 512-sample input frame must produce exactly one 512-sample VAD frame");
    resampler.reset();
    requireResampler(resampler.processSample(0.375f, output) && output == 0.375f,
                     "Equal-rate conversion must emit the first sample after reset");

    requireResampler(resampler.processSample(std::numeric_limits<float>::quiet_NaN(), output)
                     && output == 0.0f,
                     "Equal-rate conversion must sanitize NaN input to silence");
    requireResampler(resampler.processSample(std::numeric_limits<float>::infinity(), output)
                     && output == 0.0f,
                     "Equal-rate conversion must sanitize infinite input to silence");
    requireResampler(resampler.processSample(-0.25f, output) && output == -0.25f,
                     "Non-finite input must not corrupt subsequent valid equal-rate samples");

    // The FIR must preserve speech-band content while rejecting frequencies
    // above the 8 kHz destination Nyquist limit before they can alias.
    requireResampler(resampler.prepare(48000.0, 16000.0),
                     "Resampler should re-prepare for frequency-response tests");
    const double passbandRms = measureRms(resampler, 1000.0, 48000.0, 48000);
    requireResampler(passbandRms > 0.60 && passbandRms < 0.80,
                     "Low-pass filter should preserve a 1 kHz passband tone");

    requireResampler(resampler.prepare(48000.0, 16000.0),
                     "Resampler should re-prepare before alias rejection test");
    const double stopbandRms = measureRms(resampler, 12000.0, 48000.0, 48000);
    requireResampler(stopbandRms < 0.025,
                     "Low-pass filter should strongly attenuate a 12 kHz tone before decimation");

    // 96 kHz is the maximum supported host rate. The longer FIR must still
    // preserve speech-band content and reject energy just above 8 kHz.
    requireResampler(resampler.prepare(96000.0, 16000.0),
                     "96 kHz input should remain supported by the expanded FIR");
    const double highRatePassband = measureRms(resampler, 1000.0, 96000.0, 96000);
    requireResampler(highRatePassband > 0.60 && highRatePassband < 0.80,
                     "96 kHz conversion should preserve a 1 kHz speech-band tone");
    requireResampler(resampler.prepare(96000.0, 16000.0),
                     "Resampler should re-prepare before 96 kHz alias rejection test");
    const double highRateStopband = measureRms(resampler, 9000.0, 96000.0, 96000);
    requireResampler(highRateStopband < 0.002,
                     "96 kHz conversion must strongly attenuate content just above the destination Nyquist limit");

    // Keep the sanitization assertion at 48 kHz so its expected output
    // cadence is independent of the preceding 96 kHz frequency-response test.
    requireResampler(resampler.prepare(48000.0, 16000.0),
                     "Resampler should prepare for non-finite input sanitization");
    resampler.reset();
    requireResampler(resampler.processSample(0.5f, output) && std::isfinite(output),
                     "First sample after reset must be emitted as the initial filtered output");
    requireResampler(std::abs(output - 0.5f) < 1.0e-4f,
                     "FIR startup must preserve the first sample instead of creating a zero-padding level dip");
    (void) resampler.processSample(std::numeric_limits<float>::quiet_NaN(), output);
    (void) resampler.processSample(0.0f, output);
    requireResampler(resampler.processSample(0.0f, output) && std::isfinite(output),
                     "Non-finite input must be sanitized before filtering");
}
