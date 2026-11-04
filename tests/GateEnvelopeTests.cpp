#include "GateEnvelope.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runGateEnvelopeTests() {
    smartgate::GateEnvelope envelope;
    envelope.prepare(48000.0);
    envelope.setThresholdDb(-40.0f);
    envelope.setTimes(2.0f, 80.0f);

    const float firstClosedSample = envelope.process(0.0f);
    require(firstClosedSample < 1.0f && firstClosedSample > 0.0f,
            "Closing must be smoothed rather than an instantaneous hard mute");

    // A rectified periodic waveform must not chatter at every zero crossing.
    envelope.reset();
    for (int i = 0; i < 1200; ++i) {
        const float sample = (i % 48) < 24 ? 0.08f : 0.0f;
        envelope.process(sample);
    }
    require(envelope.getDetectorLevel() > 0.04f,
            "Detector should follow a periodic waveform's amplitude envelope");
    require(envelope.getGain() > 0.95f,
            "Gate should remain open across waveform gaps");

    // Once opened above -40 dB, the gate should stay open in the -40 to
    // -46 dB hysteresis band instead of toggling around the opening point.
    envelope.reset();
    envelope.setThresholdDb(-40.0f);
    envelope.setHysteresisDb(6.0f);
    envelope.setTimes(1.0f, 1.0f);
    envelope.setHoldTime(0.0f);
    for (int i = 0; i < 1200; ++i)
        envelope.process(0.02f);
    for (int i = 0; i < 6000; ++i)
        envelope.process(0.007f);
    require(envelope.getGain() > 0.9f,
            "Hysteresis band should preserve an already-open gate");
    for (int i = 0; i < 30000; ++i)
        envelope.process(0.0f);
    require(envelope.getGain() < 0.01f,
            "Signal below the closing threshold should close the gate");

    envelope.reset();
    float previous = envelope.getGain();
    for (int i = 0; i < 2000; ++i) {
        const float gain = envelope.process(1.0f);
        require(gain >= previous, "Attack gain must be monotonic");
        previous = gain;
    }
    require(envelope.getGain() > 0.99f, "Above-threshold input should open envelope");

    previous = envelope.getGain();
    // Detector smoothing must first decay below threshold; then the 80 ms
    // gain release needs additional time to reach the 1% closed-state target.
    for (int i = 0; i < 30000; ++i) {
        const float gain = envelope.process(0.0f);
        require(gain <= previous, "Release gain must be monotonic");
        previous = gain;
    }
    require(envelope.getGain() < 0.01f, "Release should eventually close envelope");

    // Hold keeps the gate open briefly after the detector falls below
    // threshold, independently of the gain release time.
    envelope.reset();
    envelope.setThresholdDb(-40.0f);
    envelope.setTimes(1.0f, 1.0f);
    envelope.setHoldTime(100.0f);
    for (int i = 0; i < 1000; ++i)
        envelope.process(1.0f);
    for (int i = 0; i < 4500; ++i)
        envelope.process(0.0f);
    require(envelope.getGain() > 0.9f,
            "Hold should preserve an open gate after the detector drops below threshold");
    for (int i = 0; i < 7000; ++i)
        envelope.process(0.0f);
    require(envelope.getGain() < 0.1f,
            "Gate should close after the configured hold interval expires");
    envelope.setHoldTime(0.0f);

    // Adaptive Hold adds a bounded extension after sustained gate activity.
    // A short transient should receive less extension than a long open phrase.
    envelope.reset();
    envelope.setThresholdDb(-40.0f);
    envelope.setHysteresisDb(3.0f);
    envelope.setTimes(1.0f, 1.0f);
    envelope.setHoldTime(20.0f);
    envelope.setAdaptiveHoldAmount(1.0f);
    for (int i = 0; i < 48000; ++i)
        envelope.process(0.2f);
    for (int i = 0; i < 4700; ++i) {
        envelope.process(0.0f);
        // The processor updates parameters each block. Reapplying an unchanged
        // base Hold must preserve the adaptive extension already in progress.
        if ((i % 64) == 0)
            envelope.setHoldTime(20.0f);
    }
    require(envelope.getGain() > 0.9f,
            "Adaptive Hold should preserve a sustained phrase beyond base Hold");
    for (int i = 0; i < 12000; ++i)
        envelope.process(0.0f);
    require(envelope.getGain() < 0.1f,
            "Adaptive Hold must remain bounded and eventually close");

    // Range controls attenuation instead of forcing every closed state to
    // silence. A 12 dB range should settle close to -12 dB gain.
    envelope.reset();
    envelope.setRangeDb(-12.0f);
    envelope.setThresholdDb(-40.0f);
    envelope.setHysteresisDb(3.0f);
    envelope.setHoldTime(0.0f);
    envelope.setTimes(1.0f, 20.0f);
    for (int i = 0; i < 30000; ++i)
        envelope.process(0.0f);
    require(std::abs(envelope.getGain() - 0.2511886f) < 0.01f,
            "Closed gate gain should match the configured Range attenuation");

    envelope.setRangeDb(-80.0f);
    for (int i = 0; i < 30000; ++i)
        envelope.process(0.0f);
    require(envelope.getGain() < 0.0002f,
            "Maximum Range should provide near-mute attenuation");

    envelope.setThresholdDb(0.0f);
    require(std::isfinite(envelope.process(0.5f)),
            "Envelope output must remain finite");
    require(std::isfinite(envelope.process(std::numeric_limits<float>::quiet_NaN())),
            "NaN input must not poison the envelope");
    envelope.setTimes(std::numeric_limits<float>::quiet_NaN(), -1.0f);
    require(std::isfinite(envelope.process(0.0f)),
            "Invalid timing values must not poison the envelope");
}
