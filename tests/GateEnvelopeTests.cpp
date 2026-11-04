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

    envelope.setThresholdDb(0.0f);
    require(std::isfinite(envelope.process(0.5f)),
            "Envelope output must remain finite");
    require(std::isfinite(envelope.process(std::numeric_limits<float>::quiet_NaN())),
            "NaN input must not poison the envelope");
    envelope.setTimes(std::numeric_limits<float>::quiet_NaN(), -1.0f);
    require(std::isfinite(envelope.process(0.0f)),
            "Invalid timing values must not poison the envelope");
}
