#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace smartgate {

// Streaming mono sample-rate converter for the VAD input path.
// A windowed-sinc FIR low-pass filter suppresses content above the
// destination Nyquist frequency before linear interpolation/decimation. The
// 95-tap design supports host rates through 96 kHz; higher rates are rejected
// rather than silently providing inadequate anti-alias attenuation.
// Coefficients are prepared off the audio thread; processing is fixed-cost,
// allocation-free, and retains state across arbitrary host block boundaries.
// Supports downsampling and equal-rate conversion.
class VadResampler {
public:
    static constexpr std::size_t filterTaps = 95;
    static constexpr double maximumInputRate = 96000.0;

    bool prepare(double inputRate, double outputRate = 16000.0) noexcept {
        prepared_ = false;
        inputRate_ = 0.0;
        outputRate_ = 16000.0;
        reset();

        if (!std::isfinite(inputRate) || !std::isfinite(outputRate)
            || inputRate < outputRate || inputRate > maximumInputRate
            || outputRate <= 0.0)
            return false;

        inputRate_ = inputRate;
        outputRate_ = outputRate;
        buildLowPass();
        prepared_ = true;
        return true;
    }

    void reset() noexcept {
        phase_ = 0.0;
        previous_ = 0.0f;
        hasPrevious_ = false;
        history_.fill(0.0f);
        historyPosition_ = 0;
        historyInitialized_ = false;
    }

    // Returns true when one output sample is ready. No allocation or locking.
    // Calls before a successful prepare() are harmless and produce no output.
    bool processSample(float input, float& output) noexcept {
        if (!prepared_)
            return false;

        const float current = filterSample(std::isfinite(input) ? input : 0.0f);

        // At equal rates there is no interpolation interval to establish:
        // every input sample is already one output sample. Emit the first
        // sample immediately instead of introducing a one-sample startup gap.
        if (inputRate_ == outputRate_) {
            output = current;
            return true;
        }

        if (!hasPrevious_) {
            previous_ = current;
            hasPrevious_ = true;
            // The FIR history is primed from the first real sample, so
            // startup has a steady boundary. Emit immediately to preserve
            // sample cadence instead of losing one output per reset.
            output = current;
            return true;
        }

        phase_ += outputRate_;
        if (phase_ >= inputRate_) {
            phase_ -= inputRate_;
            const double fraction = 1.0 - phase_ / outputRate_;
            output = previous_ + (current - previous_)
                * static_cast<float>(fraction);
            previous_ = current;
            return true;
        }

        previous_ = current;
        return false;
    }

    double inputRate() const noexcept { return inputRate_; }
    double outputRate() const noexcept { return outputRate_; }
    bool isPrepared() const noexcept { return prepared_; }

private:
    static double sinc(double value) noexcept {
        if (std::abs(value) < 1.0e-12)
            return 1.0;
        constexpr double pi = 3.1415926535897932384626433832795;
        const double angle = pi * value;
        return std::sin(angle) / angle;
    }

    void buildLowPass() noexcept {
        coefficients_.fill(0.0);
        if (inputRate_ == outputRate_) {
            coefficients_[filterTaps / 2] = 1.0;
            return;
        }

        const double cutoff = 0.45 * outputRate_ / inputRate_;
        constexpr double pi = 3.1415926535897932384626433832795;
        double coefficientSum = 0.0;
        for (std::size_t tap = 0; tap < filterTaps; ++tap) {
            const double offset = static_cast<double>(tap)
                - static_cast<double>(filterTaps / 2);
            const double ideal = 2.0 * cutoff * sinc(2.0 * cutoff * offset);
            const double window = 0.54 - 0.46
                * std::cos((2.0 * pi * static_cast<double>(tap))
                           / static_cast<double>(filterTaps - 1));
            coefficients_[tap] = ideal * window;
            coefficientSum += coefficients_[tap];
        }

        if (std::abs(coefficientSum) > 1.0e-12) {
            for (auto& coefficient : coefficients_)
                coefficient /= coefficientSum;
        }
    }

    float filterSample(float input) noexcept {
        if (inputRate_ == outputRate_)
            return input;

        // Extend the first real sample backwards to define a steady startup
        // boundary. Zero-padding would create an artificial level ramp in
        // the first VAD frame after prepare/reset.
        if (!historyInitialized_) {
            history_.fill(input);
            historyPosition_ = 0;
            historyInitialized_ = true;
        }

        history_[historyPosition_] = input;
        double filtered = 0.0;
        std::size_t position = historyPosition_;
        for (std::size_t tap = 0; tap < filterTaps; ++tap) {
            filtered += coefficients_[tap] * history_[position];
            position = position == 0 ? filterTaps - 1 : position - 1;
        }
        historyPosition_ = historyPosition_ + 1 == filterTaps
            ? 0 : historyPosition_ + 1;
        return static_cast<float>(filtered);
    }

    double inputRate_ = 0.0;
    double outputRate_ = 16000.0;
    double phase_ = 0.0;
    std::array<double, filterTaps> coefficients_{};
    std::array<float, filterTaps> history_{};
    std::size_t historyPosition_ = 0;
    bool historyInitialized_ = false;
    float previous_ = 0.0f;
    bool hasPrevious_ = false;
    bool prepared_ = false;
};

} // namespace smartgate
