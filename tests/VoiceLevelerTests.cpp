#include "VoiceLeveler.h"

#include <cmath>
#include <stdexcept>

namespace {
void requireLeveler(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runVoiceLevelerTests() {
    smartgate::VoiceLeveler bypass;
    bypass.prepare(48000.0);
    for (int i = 0; i < 1000; ++i) {
        const float sample = 0.2f * std::sin(0.03f * static_cast<float>(i));
        requireLeveler(bypass.process(sample, 0.0f) == sample,
                       "Zero Voice Leveling must be bit-transparent");
    }

    smartgate::VoiceLeveler quiet;
    quiet.prepare(48000.0);
    float quietGain = 1.0f;
    for (int i = 0; i < 48000; ++i) {
        quiet.process(0.025f * std::sin(0.03f * static_cast<float>(i)), 1.0f);
        quietGain = quiet.gainForDiagnostics();
    }
    requireLeveler(quietGain > 1.5f && quietGain <= 3.9811f,
                   "Full Voice Leveling must lift quiet sustained speech within its gain cap");

    smartgate::VoiceLeveler loud;
    loud.prepare(48000.0);
    float loudGain = 1.0f;
    for (int i = 0; i < 48000; ++i) {
        loud.process(0.8f * std::sin(0.03f * static_cast<float>(i)), 1.0f);
        loudGain = loud.gainForDiagnostics();
    }
    requireLeveler(loudGain < 0.8f && loudGain >= 0.2511f,
                   "Full Voice Leveling must reduce loud sustained input within its attenuation cap");

    loud.reset();
    requireLeveler(loud.gainForDiagnostics() == 1.0f,
                   "Reset must restore unity gain");
}
