#include "SileroVadInference.h"
#include "BinaryData.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using Inference = smartgate::SileroVadInference;
using Frame = std::array<float, Inference::frameSize>;

struct LatencySummary {
    double meanMicroseconds = 0.0;
    double p50Microseconds = 0.0;
    double p95Microseconds = 0.0;
    double maxMicroseconds = 0.0;
    float lastProbability = 0.0f;
};

LatencySummary measureCase(Inference& inference, const Frame& frame,
                           std::uint64_t generation, const char* label) {
    const auto successfulBefore = inference.successfulInferenceCount();
    // Each case starts a fresh recurrent stream, then warms that stream before
    // timing. This keeps silence, tonal speech-like input, and noise separate.
    for (int i = 0; i < 12; ++i) {
        const float probability = inference.infer(frame, generation);
        if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f)
            throw std::runtime_error("Silero returned an invalid warmup probability");
    }

    constexpr int measuredFrames = 128;
    std::vector<double> timings;
    timings.reserve(measuredFrames);
    double totalMicroseconds = 0.0;
    float lastProbability = 0.0f;
    for (int i = 0; i < measuredFrames; ++i) {
        const auto start = std::chrono::steady_clock::now();
        lastProbability = inference.infer(frame, generation);
        const auto elapsed = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        if (!std::isfinite(lastProbability)
            || lastProbability < 0.0f || lastProbability > 1.0f)
            throw std::runtime_error("Silero returned an invalid probability during timing");
        timings.push_back(elapsed);
        totalMicroseconds += elapsed;
    }

    std::sort(timings.begin(), timings.end());
    const auto percentile = [&timings](double fraction) {
        const auto index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(timings.size()))) - 1;
        return timings[std::min(index, timings.size() - 1)];
    };

    LatencySummary summary;
    summary.meanMicroseconds = totalMicroseconds / measuredFrames;
    summary.p50Microseconds = percentile(0.50);
    summary.p95Microseconds = percentile(0.95);
    summary.maxMicroseconds = timings.back();
    summary.lastProbability = lastProbability;

    if (inference.successfulInferenceCount() != successfulBefore + 12 + measuredFrames)
        throw std::runtime_error("Silero inference did not complete every ONNX Run() successfully");

    std::cout << "Silero ONNX latency [" << label << "] (128 warmed CPU frames): "
              << "mean=" << summary.meanMicroseconds
              << " us, p50=" << summary.p50Microseconds
              << " us, p95=" << summary.p95Microseconds
              << " us, max=" << summary.maxMicroseconds
              << " us, last_probability=" << summary.lastProbability
              << "; 32 ms frame budget=32000 us.\n";
    return summary;
}
}

int main() {
    try {
        Inference inference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));

        Frame silence{};
        measureCase(inference, silence, 1, "silence");

        Frame speechLike{};
        for (std::size_t i = 0; i < speechLike.size(); ++i) {
            const double t = static_cast<double>(i) / 16000.0;
            const double envelope = 0.75 + 0.25 * std::sin(
                2.0 * 3.14159265358979323846 * 3.0 * t);
            speechLike[i] = static_cast<float>(envelope * (
                0.12 * std::sin(2.0 * 3.14159265358979323846 * 180.0 * t)
              + 0.06 * std::sin(2.0 * 3.14159265358979323846 * 360.0 * t)
              + 0.025 * std::sin(2.0 * 3.14159265358979323846 * 720.0 * t)
              + 0.015 * std::sin(2.0 * 3.14159265358979323846 * 1200.0 * t)));
        }
        measureCase(inference, speechLike, 2, "speech-like-tones");

        Frame deterministicNoise{};
        std::uint32_t randomState = 0x13579BDFu;
        for (auto& sample : deterministicNoise) {
            randomState = randomState * 1664525u + 1013904223u;
            const auto centered = static_cast<std::int32_t>(randomState >> 8)
                                - static_cast<std::int32_t>(0x7FFFFFu);
            sample = static_cast<float>(centered / 8388608.0 * 0.08);
        }
        measureCase(inference, deterministicNoise, 3, "deterministic-noise");

        // A generation change must reset recurrent state and remain usable.
        const auto beforeGenerationReset = inference.successfulInferenceCount();
        const float afterReset = inference.infer(silence, 4);
        if (!std::isfinite(afterReset) || afterReset < 0.0f || afterReset > 1.0f)
            throw std::runtime_error("Silero failed after a generation reset");

        const float speechBand = inference.infer(speechLike, 4);
        if (!std::isfinite(speechBand) || speechBand < 0.0f || speechBand > 1.0f)
            throw std::runtime_error("Silero failed to process a finite speech-band frame");
        if (inference.successfulInferenceCount() != beforeGenerationReset + 2)
            throw std::runtime_error("Silero generation-reset frames did not complete ONNX inference");

        // Non-finite audio samples must be sanitized before reaching ONNX.
        // The model call must still complete, and the next valid frame must
        // continue to produce a finite result.
        Frame nonFiniteAudio{};
        for (std::size_t i = 0; i < nonFiniteAudio.size(); ++i) {
            if ((i % 3) == 0)
                nonFiniteAudio[i] = std::numeric_limits<float>::quiet_NaN();
            else if ((i % 3) == 1)
                nonFiniteAudio[i] = std::numeric_limits<float>::infinity();
            else
                nonFiniteAudio[i] = -std::numeric_limits<float>::infinity();
        }
        const auto beforeNonFiniteAudio = inference.successfulInferenceCount();
        const float afterNonFiniteAudio = inference.infer(nonFiniteAudio, 5);
        if (!std::isfinite(afterNonFiniteAudio)
            || afterNonFiniteAudio < 0.0f || afterNonFiniteAudio > 1.0f)
            throw std::runtime_error("Silero returned an invalid probability for non-finite audio");
        if (inference.successfulInferenceCount() != beforeNonFiniteAudio + 1)
            throw std::runtime_error("Silero did not complete ONNX inference for sanitized non-finite audio");

        const auto beforeRecoveryFrame = inference.successfulInferenceCount();
        const float afterRecoveryFrame = inference.infer(speechLike, 5);
        if (!std::isfinite(afterRecoveryFrame)
            || afterRecoveryFrame < 0.0f || afterRecoveryFrame > 1.0f)
            throw std::runtime_error("Silero failed to recover after non-finite audio");
        if (inference.successfulInferenceCount() != beforeRecoveryFrame + 1)
            throw std::runtime_error("Silero did not complete ONNX inference after non-finite audio");

        // Exactly 1.5 model-frame intervals is still contiguous enough to
        // preserve recurrent state. Compare against an instance without source
        // tracking so an accidental >= boundary cannot silently reset it.
        Inference boundaryInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        Inference continuousReference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        constexpr std::uint64_t discontinuityGeneration = 99;
        const float boundaryFirst = boundaryInference.infer(
            speechLike, discontinuityGeneration, 1000);
        const float referenceFirst = continuousReference.infer(
            speechLike, discontinuityGeneration);
        const float boundarySecond = boundaryInference.infer(
            speechLike, discontinuityGeneration, 1768); // 768 = 1.5 * 512
        const float referenceSecond = continuousReference.infer(
            speechLike, discontinuityGeneration);
        if (!std::isfinite(boundaryFirst) || !std::isfinite(referenceFirst)
            || !std::isfinite(boundarySecond) || !std::isfinite(referenceSecond)
            || std::abs(boundarySecond - referenceSecond) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must preserve recurrent state at the exact 1.5-frame discontinuity boundary");

        // One sample beyond the boundary must reset both recurrent state and
        // the 64-sample context. The first post-gap result must match a fresh
        // model instance processing that same frame.
        Inference freshInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        const float afterGap = boundaryInference.infer(
            speechLike, discontinuityGeneration, 2537); // 769 samples after 1768
        const float freshAfterGap = freshInference.infer(
            speechLike, discontinuityGeneration, 2537);
        if (!std::isfinite(afterGap) || !std::isfinite(freshAfterGap)
            || std::abs(afterGap - freshAfterGap) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must reset recurrent state and context after a source-position gap");

        // Repeated or backwards source positions are also discontinuities.
        Inference freshAfterBackwardsInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        const float afterBackwardsPosition = boundaryInference.infer(
            speechLike, discontinuityGeneration, 2000);
        const float freshAfterBackwardsPosition = freshAfterBackwardsInference.infer(
            speechLike, discontinuityGeneration, 2000);
        if (!std::isfinite(afterBackwardsPosition)
            || !std::isfinite(freshAfterBackwardsPosition)
            || std::abs(afterBackwardsPosition - freshAfterBackwardsPosition) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must reset recurrent state after a backwards source position");

        Inference freshAfterDuplicateInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize));
        const float afterDuplicatePosition = boundaryInference.infer(
            speechLike, discontinuityGeneration, 2000);
        const float freshAfterDuplicatePosition = freshAfterDuplicateInference.infer(
            speechLike, discontinuityGeneration, 2000);
        if (!std::isfinite(afterDuplicatePosition)
            || !std::isfinite(freshAfterDuplicatePosition)
            || std::abs(afterDuplicatePosition - freshAfterDuplicatePosition) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must reset recurrent state after a repeated source position");

        if (boundaryInference.successfulInferenceCount() != 5
            || continuousReference.successfulInferenceCount() != 2
            || freshInference.successfulInferenceCount() != 1
            || freshAfterBackwardsInference.successfulInferenceCount() != 1
            || freshAfterDuplicateInference.successfulInferenceCount() != 1)
            throw std::runtime_error(
                "Silero source-position discontinuity tests must complete all expected ONNX calls");

        // Fractional host rates need a host-rate-aware discontinuity threshold.
        // At 44.1 kHz one model frame spans 1411.2 host samples, so 2116
        // samples is still within 1.5 frames while 2117 must reset the state.
        Inference fractionalRateInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 44100.0);
        Inference fractionalRateReference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 44100.0);
        (void) fractionalRateInference.infer(speechLike, 120, 10000);
        (void) fractionalRateReference.infer(speechLike, 120);
        const float fractionalBoundary = fractionalRateInference.infer(
            speechLike, 120, 12116); // 2116 samples after the first frame
        const float fractionalReference = fractionalRateReference.infer(speechLike, 120);
        if (!std::isfinite(fractionalBoundary) || !std::isfinite(fractionalReference)
            || std::abs(fractionalBoundary - fractionalReference) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must preserve recurrent state within the fractional-rate 1.5-frame boundary");
        Inference fractionalGapFresh(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 44100.0);
        const float fractionalAfterGap = fractionalRateInference.infer(
            speechLike, 120, 14233); // 2117 samples after the previous frame
        const float fractionalFresh = fractionalGapFresh.infer(speechLike, 120, 14233);
        if (!std::isfinite(fractionalAfterGap) || !std::isfinite(fractionalFresh)
            || std::abs(fractionalAfterGap - fractionalFresh) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must reset recurrent state one sample beyond the fractional-rate boundary");

        // Repeat at the maximum supported host rate. A 96 kHz model frame
        // spans 3072 host samples; the exact 4608-sample boundary is retained.
        Inference highRateInference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 96000.0);
        Inference highRateReference(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 96000.0);
        (void) highRateInference.infer(speechLike, 121, 20000);
        (void) highRateReference.infer(speechLike, 121);
        const float highRateBoundary = highRateInference.infer(
            speechLike, 121, 24608); // exactly 4608 samples
        const float highRateContinuous = highRateReference.infer(speechLike, 121);
        if (!std::isfinite(highRateBoundary) || !std::isfinite(highRateContinuous)
            || std::abs(highRateBoundary - highRateContinuous) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must preserve recurrent state at the 96 kHz 1.5-frame boundary");
        Inference highRateGapFresh(BinaryData::silero_vad_onnx,
            static_cast<std::size_t>(BinaryData::silero_vad_onnxSize), 96000.0);
        const float highRateAfterGap = highRateInference.infer(
            speechLike, 121, 29217); // 4609 samples after the previous frame
        const float highRateFresh = highRateGapFresh.infer(speechLike, 121, 29217);
        if (!std::isfinite(highRateAfterGap) || !std::isfinite(highRateFresh)
            || std::abs(highRateAfterGap - highRateFresh) > 1.0e-6f)
            throw std::runtime_error(
                "Silero must reset recurrent state one sample beyond the 96 kHz boundary");

        std::cout << "Silero ONNX smoke test passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SILERO TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
