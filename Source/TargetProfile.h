#pragma once

#include "AudioFeatures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace smartgate {

// Online target statistics using Welford's algorithm. The caller controls
// when frames are learned; this class never learns implicitly during detection.
class TargetProfile {
public:
    static constexpr std::uint64_t minimumFrames = 8;
    static constexpr std::uint64_t maximumRestoredFrames = 1000000000ULL;

    struct Snapshot {
        std::uint64_t count = 0;
        std::array<double, 5> mean{};
        std::array<double, 5> m2{};
    };

    Snapshot makeSnapshot() const noexcept {
        return {count_, mean_, m2_};
    }

    // Restore is intended for validated, non-real-time state loading. The
    // caller must ensure this object is not accessed concurrently.
    bool restore(const Snapshot& snapshot) noexcept {
        if (snapshot.count > maximumRestoredFrames)
            return false;

        for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
            if (!std::isfinite(snapshot.mean[i])
                || !std::isfinite(snapshot.m2[i])
                || snapshot.m2[i] < 0.0)
                return false;
        }

        count_ = snapshot.count;
        mean_ = snapshot.mean;
        m2_ = snapshot.m2;
        if (count_ == 0) {
            mean_.fill(0.0);
            m2_.fill(0.0);
        }
        return true;
    }

    void clear() noexcept {
        count_ = 0;
        mean_.fill(0.0);
        m2_.fill(0.0);
    }

    bool learn(const AudioFeatures& features) noexcept {
        if (!features.isFinite())
            return false;

        const auto values = toArray(features);
        ++count_;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double delta = static_cast<double>(values[i]) - mean_[i];
            mean_[i] += delta / static_cast<double>(count_);
            const double deltaAfter = static_cast<double>(values[i]) - mean_[i];
            m2_[i] += delta * deltaAfter;
        }
        return true;
    }

    bool isReady() const noexcept { return count_ >= minimumFrames; }
    std::uint64_t frameCount() const noexcept { return count_; }

    // Returns 0..1. This is feature similarity, not speaker identity.
    float confidence(const AudioFeatures& features) const noexcept {
        if (!isReady() || !features.isFinite())
            return 0.0f;

        // Floors prevent near-zero training variance from making the model
        // hypersensitive to harmless frame-to-frame fluctuations.
        constexpr std::array<double, 5> varianceFloors{
            0.08 * 0.08, 0.015 * 0.015, 0.04 * 0.04,
            0.04 * 0.04, 0.04 * 0.04
        };
        const auto values = toArray(features);
        double distance = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double variance = count_ > 1
                ? m2_[i] / static_cast<double>(count_ - 1) : 0.0;
            const double scale = std::sqrt(std::max(variance, varianceFloors[i]));
            const double z = (static_cast<double>(values[i]) - mean_[i]) / scale;
            distance += z * z;
        }

        // Mean squared normalized distance gives a smooth confidence curve.
        const double meanSquaredDistance = distance / static_cast<double>(values.size());
        return static_cast<float>(std::exp(-0.5 * meanSquaredDistance));
    }

private:
    static std::array<float, 5> toArray(const AudioFeatures& f) noexcept {
        return {f.logRms, f.zeroCrossingRate, f.lowBandRatio,
                f.midBandRatio, f.highBandRatio};
    }

    std::uint64_t count_ = 0;
    std::array<double, 5> mean_{};
    std::array<double, 5> m2_{};
};

} // namespace smartgate
