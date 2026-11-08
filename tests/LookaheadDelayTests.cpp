#include "LookaheadDelay.h"

#include <cmath>
#include <stdexcept>

void runLookaheadDelayTests() {
    smartgate::LookaheadDelay delay;
    if (!delay.prepare(48000.0, 30.0))
        throw std::runtime_error("lookahead prepare failed");
    if (delay.latencySamples() != 1440)
        throw std::runtime_error("30 ms at 48 kHz must be 1440 samples");

    float outL = 1.0f, outR = 1.0f;
    for (int i = 0; i < 1440; ++i) {
        delay.process(i == 0 ? 0.75f : 0.0f, i == 0 ? -0.5f : 0.0f, outL, outR);
        if (outL != 0.0f || outR != 0.0f)
            throw std::runtime_error("impulse escaped before full lookahead");
    }
    delay.process(0.0f, 0.0f, outL, outR);
    if (std::abs(outL - 0.75f) >= 1.0e-7f || std::abs(outR + 0.5f) >= 1.0e-7f)
        throw std::runtime_error("stereo impulse did not emerge at exact delay");

    delay.reset();
    delay.process(0.2f, -0.2f, outL, outR);
    if (outL != 0.0f || outR != 0.0f)
        throw std::runtime_error("reset did not clear buffered audio");

    if (!delay.prepare(48000.0, 0.0))
        throw std::runtime_error("zero-latency prepare failed");
    delay.process(0.3f, -0.4f, outL, outR);
    if (outL != 0.3f || outR != -0.4f)
        throw std::runtime_error("zero latency must be bit-transparent");
}
