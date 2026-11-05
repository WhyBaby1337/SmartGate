#pragma once

#include "AudioFeatures.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <thread>

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

        // Only frames inside finishFeatureFrame()'s acquisition window can
        // enter a trained profile: log RMS is (-2.5, -0.03), while ZCR and
        // spectral energy ratios are all in [0, 1]. Reject finite but
        // impossible project data before it can poison subsequent Welford
        // updates (for example, a 1e308 mean causing M2 to overflow).
        if (snapshot.count > 0) {
            constexpr double logRmsMinimum = -2.5;
            constexpr double logRmsMaximum = -0.03;
            constexpr double featureMinimum = 0.0;
            constexpr double featureMaximum = 1.0;
            constexpr std::array<double, 5> minima{
                logRmsMinimum, featureMinimum, featureMinimum,
                featureMinimum, featureMinimum
            };
            constexpr std::array<double, 5> maxima{
                logRmsMaximum, featureMaximum, featureMaximum,
                featureMaximum, featureMaximum
            };

            for (std::size_t i = 0; i < snapshot.mean.size(); ++i) {
                const double range = maxima[i] - minima[i];
                const double maximumM2 = static_cast<double>(snapshot.count)
                    * range * range * 0.25;
                const double tolerance = 1.0e-9 * std::max(1.0, maximumM2);
                if (snapshot.mean[i] < minima[i]
                    || snapshot.mean[i] > maxima[i]
                    || snapshot.m2[i] > maximumM2 + tolerance)
                    return false;
            }

            const double ratioSum = snapshot.mean[2]
                + snapshot.mean[3] + snapshot.mean[4];
            if (std::abs(ratioSum - 1.0) > 1.0e-3)
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
        if (!features.isFinite() || count_ >= maximumRestoredFrames)
            return false;

        const auto values = toArray(features);
        // Keep the learning and restore contracts identical: every accepted
        // observation must be representable by a valid persisted profile.
        if (values[0] <= -2.5f || values[0] >= -0.03f)
            return false;
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i] < 0.0f || values[i] > 1.0f)
                return false;
        }
        const float ratioSum = values[2] + values[3] + values[4];
        if (!std::isfinite(ratioSum) || std::abs(ratioSum - 1.0f) > 1.0e-3f)
            return false;
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

class AtomicTargetProfileSnapshot {
public:
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "Requires lock-free 64-bit atomics");

    void publish(const TargetProfile::Snapshot& s) noexcept {
        sequence_.fetch_add(1, std::memory_order_acq_rel);
        count_.store(s.count, std::memory_order_relaxed);
        for (std::size_t i = 0; i < 5; ++i) {
            mean_[i].store(bits(s.mean[i]), std::memory_order_relaxed);
            m2_[i].store(bits(s.m2[i]), std::memory_order_relaxed);
        }
        sequence_.fetch_add(1, std::memory_order_release);
    }

    // Persistence runs off the audio thread. Retry while a published
    // snapshot is being updated so a transient seqlock collision cannot
    // silently omit the trained profile from a saved project. An unpublished
    // mailbox (sequence zero) represents the empty pre-prepare state.
    bool readForPersistence(TargetProfile::Snapshot& s,
                            std::uint64_t& version) const noexcept {
        for (;;) {
            if (read(s, version, 64))
                return true;
            if (sequence_.load(std::memory_order_acquire) == 0)
                return false;
            std::this_thread::yield();
        }
    }

    bool read(TargetProfile::Snapshot& s, std::uint64_t& version,
              int maximumAttempts = 8) const noexcept {
        // Audio-thread callers keep the short bounded default. Persistence
        // callers may request more retries because they are not real-time.
        for (int attempt = 0; attempt < std::max(0, maximumAttempts); ++attempt) {
            const auto first = sequence_.load(std::memory_order_acquire);
            if (first == 0 || (first & 1U)) continue;
            TargetProfile::Snapshot copy;
            copy.count = count_.load(std::memory_order_relaxed);
            for (std::size_t i = 0; i < 5; ++i) {
                copy.mean[i] = value(mean_[i].load(std::memory_order_relaxed));
                copy.m2[i] = value(m2_[i].load(std::memory_order_relaxed));
            }
            const auto last = sequence_.load(std::memory_order_acquire);
            if (first == last && !(last & 1U)) {
                s = copy;
                version = last;
                return true;
            }
        }
        return false;
    }

private:
    static std::uint64_t bits(double x) noexcept {
        std::uint64_t v;
        std::memcpy(&v, &x, sizeof(v));
        return v;
    }
    static double value(std::uint64_t v) noexcept {
        double x;
        std::memcpy(&x, &v, sizeof(x));
        return x;
    }
    std::atomic<std::uint64_t> sequence_{0}, count_{0};
    std::array<std::atomic<std::uint64_t>, 5> mean_{}, m2_{};
};

} // namespace smartgate
