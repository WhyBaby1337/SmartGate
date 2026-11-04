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

    envelope.reset();
    float previous = envelope.getGain();
    for (int i = 0; i < 2000; ++i) {
        const float gain = envelope.process(1.0f);
        require(gain >= previous, "Attack gain must be monotonic");
        previous = gain;
    }
    require(envelope.getGain() > 0.99f, "Above-threshold input should open envelope");

    previous = envelope.getGain();
    for (int i = 0; i < 20000; ++i) {
        const float gain = envelope.process(0.0f);
        require(gain <= previous, "Release gain must be monotonic");
        previous = gain;
    }
    require(envelope.getGain() < 0.01f, "Release should eventually close envelope");

    envelope.setThresholdDb(0.0f);
    require(std::isfinite(envelope.process(0.5f)),
            "Envelope output must remain finite");
    require(std::isfinite(envelope.process(std::numeric_limits<float>::quiet_NaN())),
            "NaN input must not poison the envelope");
    envelope.setTimes(std::numeric_limits<float>::quiet_NaN(), -1.0f);
    require(std::isfinite(envelope.process(0.0f)),
            "Invalid timing values must not poison the envelope");
}
