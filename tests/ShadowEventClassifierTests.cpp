#include "ShadowEventClassifier.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
void requireShadow(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
}

void runShadowEventClassifierTests() {
    smartgate::ShadowEventClassifier classifier;
    const smartgate::AudioFeatures steadyVoice{
        -0.8f, 0.12f, 0.20f, 0.48f, 0.32f
    };
    const auto first = classifier.process(steadyVoice, 0.92f, 0.05f);
    requireShadow(first.isFinite() && first.speech > 0.9f,
                  "Shadow classifier must preserve the supplied speech confidence");
    requireShadow(first.mouse == 0.0f,
                  "First frame must not invent a transient without history");

    const auto repeated = classifier.process(steadyVoice, 0.90f, 0.05f);
    requireShadow(repeated.mouse < 0.01f,
                  "Stable voiced frames must not become mouse candidates");

    const smartgate::AudioFeatures brightTransient{
        -0.25f, 0.42f, 0.04f, 0.16f, 0.80f
    };
    const auto transient = classifier.process(brightTransient, 0.88f, 0.05f);
    requireShadow(transient.isFinite() && transient.mouse > 0.05f,
                  "A sudden bright broadband event must raise the shadow mouse score");
    requireShadow(transient.speech > 0.8f,
                  "Mouse suspicion must not erase simultaneous speech evidence");

    const auto keyboard = classifier.process(brightTransient, 0.80f, 0.95f);
    requireShadow(keyboard.keyboard > 0.9f
                  && keyboard.mouse <= transient.mouse,
                  "Strong keyboard evidence must reduce ambiguous mouse attribution");

    const auto invalid = classifier.process(
        {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.2f, 0.3f, 0.5f},
        1.0f, 1.0f);
    requireShadow(invalid.isFinite()
                  && invalid.speech == 0.0f && invalid.keyboard == 0.0f
                  && invalid.mouse == 0.0f,
                  "Invalid feature frames must return a safe unknown result");

    classifier.reset();
    const auto afterReset = classifier.process(brightTransient, 0.0f, 0.0f);
    requireShadow(afterReset.mouse == 0.0f,
                  "Reset must clear transient history");
    const auto bounded = classifier.process(
        {-0.4f, 0.2f, 0.1f, 0.2f, 0.7f}, 4.0f, -2.0f);
    requireShadow(bounded.isFinite() && bounded.speech == 1.0f
                  && bounded.keyboard == 0.0f && bounded.mouse >= 0.0f
                  && bounded.mouse <= 1.0f && bounded.unknown >= 0.0f
                  && bounded.unknown <= 1.0f,
                  "All shadow scores must remain finite and bounded");
}
