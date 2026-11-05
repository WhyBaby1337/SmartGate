#pragma once

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace smartgate {

class SileroVadInference {
public:
    static constexpr std::size_t frameSize = 512;
    static constexpr std::size_t contextSize = 64;
    static constexpr std::size_t recurrentStateSize = 2 * 128;
    static constexpr std::size_t modelInputSize = contextSize + frameSize;

    SileroVadInference(const void* modelData, std::size_t modelSize)
        : env_(ORT_LOGGING_LEVEL_WARNING, "SmartGate-Silero"),
          memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        if (modelData == nullptr || modelSize == 0)
            throw Ort::Exception("Silero model data is empty", ORT_INVALID_ARGUMENT);

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        session_ = std::make_unique<Ort::Session>(env_, modelData, modelSize, options);

        if (session_->GetInputCount() != 3 || session_->GetOutputCount() != 2)
            throw Ort::Exception("Unexpected Silero ONNX input/output count", ORT_INVALID_GRAPH);
        resetState();
    }

    SileroVadInference(const SileroVadInference&) = delete;
    SileroVadInference& operator=(const SileroVadInference&) = delete;

    // One instance belongs to one inference worker. Never call from processBlock().
    float infer(const std::array<float, frameSize>& frame,
                std::uint64_t generation) noexcept {
        if (!session_)
            return 0.0f;

        if (!hasGeneration_ || generation != generation_) {
            resetState();
            generation_ = generation;
            hasGeneration_ = true;
        }

        std::copy(context_.begin(), context_.end(), input_.begin());
        for (std::size_t i = 0; i < frame.size(); ++i)
            input_[contextSize + i] = std::isfinite(frame[i]) ? frame[i] : 0.0f;

        try {
            constexpr std::array<std::int64_t, 2> inputShape{1, 576};
            constexpr std::array<std::int64_t, 3> stateShape{2, 1, 128};
            constexpr std::int64_t sampleRate = 16000;

            std::array<Ort::Value, 3> inputs{
                Ort::Value::CreateTensor<float>(memoryInfo_, input_.data(),
                    input_.size(), inputShape.data(), inputShape.size()),
                Ort::Value::CreateTensor<float>(memoryInfo_, recurrentState_.data(),
                    recurrentState_.size(), stateShape.data(), stateShape.size()),
                Ort::Value::CreateTensor<std::int64_t>(memoryInfo_,
                    const_cast<std::int64_t*>(&sampleRate), 1, nullptr, 0)
            };

            constexpr std::array<const char*, 3> inputNames{"input", "state", "sr"};
            constexpr std::array<const char*, 2> outputNames{"output", "state"};
            auto outputs = session_->Run(Ort::RunOptions{nullptr},
                inputNames.data(), inputs.data(), inputs.size(),
                outputNames.data(), outputNames.size());

            if (outputs.size() != 2
                || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() < 1
                || outputs[1].GetTensorTypeAndShapeInfo().GetElementCount()
                    != recurrentState_.size()) {
                resetState();
                return 0.0f;
            }

            const float rawProbability = outputs[0].GetTensorData<float>()[0];
            const float* nextState = outputs[1].GetTensorData<float>();
            std::copy(nextState, nextState + recurrentState_.size(),
                      recurrentState_.begin());
            std::copy(input_.end() - static_cast<std::ptrdiff_t>(contextSize),
                      input_.end(), context_.begin());

            return std::isfinite(rawProbability)
                ? std::max(0.0f, std::min(1.0f, rawProbability)) : 0.0f;
        } catch (...) {
            resetState();
            return 0.0f;
        }
    }

    void resetState() noexcept {
        context_.fill(0.0f);
        recurrentState_.fill(0.0f);
        input_.fill(0.0f);
    }

private:
    Ort::Env env_;
    Ort::MemoryInfo memoryInfo_;
    std::unique_ptr<Ort::Session> session_;
    std::array<float, contextSize> context_{};
    std::array<float, recurrentStateSize> recurrentState_{};
    std::array<float, modelInputSize> input_{};
    std::uint64_t generation_ = 0;
    bool hasGeneration_ = false;
};

} // namespace smartgate
