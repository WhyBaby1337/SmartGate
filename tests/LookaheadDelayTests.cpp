#include "LookaheadDelay.h"

#include <cmath>
#include <stdexcept>
#include <iostream>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}

void runLookaheadDelayTests() {
    smartgate::LookaheadDelay delay;
    require(delay.prepare(48000.0, 30.0), "prepare succeeds");
    require(delay.latencySamples() == 1440, "30 ms at 48 kHz is 1440 samples");

    float outL = 1.0f, outR = 1.0f;
    for (int i = 0; i < 1440; ++i) {
        delay.process(i == 0 ? 0.75f : 0.0f, i == 0 ? -0.5f : 0.0f, outL, outR);
        require(outL == 0.0f && outR == 0.0f, "impulse remains delayed for full lookahead");
    }
    delay.process(0.0f, 0.0f, outL, outR);
    require(std::abs(outL - 0.75f) < 1.0e-7f, "left impulse emerges at exact delay");
    require(std::abs(outR + 0.5f) < 1.0e-7f, "right impulse emerges at exact delay");

    delay.reset();
    delay.process(0.2f, -0.2f, outL, outR);
    require(outL == 0.0f && outR == 0.0f, "reset clears buffered audio");

    require(delay.prepare(48000.0, 0.0), "zero-latency prepare succeeds");
    delay.process(0.3f, -0.4f, outL, outR);
    require(outL == 0.3f && outR == -0.4f, "zero latency is bit-transparent");
    std::cout << "LookaheadDelay tests passed\n";
    return 0;
}
