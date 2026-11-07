#pragma once

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace smartgate {

class SileroVadInference {
public:
    static constexpr std::size_t frameSize = 512;
    static constexpr std::size_t contextSize = 64;
    static constexpr std::size_t recurrentStateSize = 2 * 128;
    static constexpr std::size_t modelInputSize = contextSize + frameSize;
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "Audio-thread VAD failure diagnostics require lock-free 64-bit atomics");

    SileroVadInference(const void* modelData, std::size_t modelSize,
                       double hostSampleRate = 16000.0)
        {
        // Resolve the runtime beside the module that owns this code. A host
        // process may already have loaded a different onnxruntime.dll; using
        // that module by basename can expose an incompatible C API version.
        initializeOrtApi();
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SmartGate-Silero");
        memoryInfo_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        expectedSourceFrameSamples_ = std::isfinite(hostSampleRate)
            ? std::max(512.0, hostSampleRate * static_cast<double>(frameSize) / 16000.0)
            : static_cast<double>(frameSize);
        if (modelData == nullptr || modelSize == 0)
            throw Ort::Exception("Silero model data is empty", ORT_INVALID_ARGUMENT);

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        session_ = std::make_unique<Ort::Session>(*env_, modelData, modelSize, options);

        if (session_->GetInputCount() != 3 || session_->GetOutputCount() != 2)
            throw Ort::Exception("Unexpected Silero ONNX input/output count", ORT_INVALID_GRAPH);

        // Validate the actual graph contract at startup. Silero v5 names its
        // recurrent-state output "stateN" (not "state"); requesting the wrong
        // output name makes every Run() throw and can otherwise look like a
        // valid stream of zero speech probabilities.
        constexpr std::array<const char*, 3> expectedInputs{"input", "state", "sr"};
        constexpr std::array<const char*, 2> expectedOutputs{"output", "stateN"};
        Ort::AllocatorWithDefaultOptions allocator;
        const auto hasNames = [this, &allocator](std::size_t count,
                                           const auto& expected,
                                           bool inputs) {
            for (const auto* requiredName : expected) {
                bool found = false;
                for (std::size_t index = 0; index < count; ++index) {
                    auto actualName = inputs
                        ? session_->GetInputNameAllocated(index, allocator)
                        : session_->GetOutputNameAllocated(index, allocator);
                    if (actualName && std::strcmp(actualName.get(), requiredName) == 0) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    return false;
            }
            return true;
        };
        if (!hasNames(session_->GetInputCount(), expectedInputs, true)
            || !hasNames(session_->GetOutputCount(), expectedOutputs, false))
            throw Ort::Exception("Silero ONNX input/output names do not match the expected v5 contract",
                                 ORT_INVALID_GRAPH);
        resetState();
    }

    SileroVadInference(const SileroVadInference&) = delete;
    SileroVadInference& operator=(const SileroVadInference&) = delete;

    // One instance belongs to one inference worker. Never call from processBlock().
    float infer(const std::array<float, frameSize>& frame,
                std::uint64_t generation,
                std::uint64_t sourceSamplePosition = ~std::uint64_t{0}) noexcept {
        if (!session_) {
            inferenceFailureCount_.fetch_add(1, std::memory_order_relaxed);
            return 0.0f;
        }

        const bool hasSourcePosition = sourceSamplePosition != ~std::uint64_t{0};
        const bool sourceDiscontinuity = hasSourcePosition && hasSourcePosition_
            && (sourceSamplePosition <= lastSourceSamplePosition_
                || static_cast<double>(sourceSamplePosition - lastSourceSamplePosition_)
                    > expectedSourceFrameSamples_ * 1.5);
        if (!hasGeneration_ || generation != generation_ || sourceDiscontinuity) {
            resetState();
            generation_ = generation;
            hasGeneration_ = true;
        }
        if (hasSourcePosition) {
            lastSourceSamplePosition_ = sourceSamplePosition;
            hasSourcePosition_ = true;
        }

        std::copy(context_.begin(), context_.end(), input_.begin());
        for (std::size_t i = 0; i < frame.size(); ++i)
            input_[contextSize + i] = std::isfinite(frame[i]) ? frame[i] : 0.0f;

        try {
            constexpr std::array<std::int64_t, 2> inputShape{1, 576};
            constexpr std::array<std::int64_t, 3> stateShape{2, 1, 128};
            std::int64_t sampleRate = 16000;
            constexpr std::array<std::int64_t, 1> sampleRateShape{1};

            std::array<Ort::Value, 3> inputs{
                Ort::Value::CreateTensor<float>(*memoryInfo_, input_.data(),
                    input_.size(), inputShape.data(), inputShape.size()),
                Ort::Value::CreateTensor<float>(*memoryInfo_, recurrentState_.data(),
                    recurrentState_.size(), stateShape.data(), stateShape.size()),
                Ort::Value::CreateTensor<std::int64_t>(memoryInfo_,
                    &sampleRate, 1, sampleRateShape.data(), sampleRateShape.size())
            };

            constexpr std::array<const char*, 3> inputNames{"input", "state", "sr"};
            constexpr std::array<const char*, 2> outputNames{"output", "stateN"};
            auto outputs = session_->Run(Ort::RunOptions{nullptr},
                inputNames.data(), inputs.data(), inputs.size(),
                outputNames.data(), outputNames.size());

            if (outputs.size() != 2) {
                recordInferenceFailure();
                resetState();
                return 0.0f;
            }

            const auto probabilityInfo = outputs[0].GetTensorTypeAndShapeInfo();
            const auto stateInfo = outputs[1].GetTensorTypeAndShapeInfo();
            if (probabilityInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                || probabilityInfo.GetElementCount() < 1
                || stateInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                || stateInfo.GetElementCount() != recurrentState_.size()) {
                recordInferenceFailure();
                resetState();
                return 0.0f;
            }

            const float rawProbability = outputs[0].GetTensorData<float>()[0];
            const float* nextState = outputs[1].GetTensorData<float>();
            if (!std::isfinite(rawProbability)) {
                recordInferenceFailure();
                resetState();
                return 0.0f;
            }

            for (std::size_t i = 0; i < recurrentState_.size(); ++i) {
                if (!std::isfinite(nextState[i])) {
                    recordInferenceFailure();
                    resetState();
                    return 0.0f;
                }
            }

            std::copy(nextState, nextState + recurrentState_.size(),
                      recurrentState_.begin());
            std::copy(input_.end() - static_cast<std::ptrdiff_t>(contextSize),
                      input_.end(), context_.begin());
            ++successfulInferenceCount_;

            return std::max(0.0f, std::min(1.0f, rawProbability));
        } catch (...) {
            recordInferenceFailure();
            resetState();
            return 0.0f;
        }
    }

    // Counts failed/invalid ONNX evaluations; safe to inspect from the audio thread.
    std::uint64_t inferenceFailureCount() const noexcept {
        return inferenceFailureCount_.load(std::memory_order_relaxed);
    }

    // Counts only completed ONNX Run() calls with validated probability and state.
    // The inference instance is worker-thread confined; no atomic is needed.
    std::uint64_t successfulInferenceCount() const noexcept {
        return successfulInferenceCount_;
    }

    void resetState() noexcept {
        context_.fill(0.0f);
        recurrentState_.fill(0.0f);
        input_.fill(0.0f);
    }

private:
    static void initializeOrtApi() {
        static std::once_flag initialized;
        static HMODULE runtimeModule = nullptr;
        std::call_once(initialized, [] {
            HMODULE ownerModule = nullptr;
            if (!GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                        | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(&moduleAnchor_),
                    &ownerModule))
                throw std::runtime_error("Could not locate the Smart Gate module for ONNX Runtime loading");

            std::array<wchar_t, 32768> modulePath{};
            const DWORD pathLength = GetModuleFileNameW(
                ownerModule, modulePath.data(), static_cast<DWORD>(modulePath.size()));
            if (pathLength == 0 || pathLength >= modulePath.size())
                throw std::runtime_error("Could not resolve the Smart Gate module path");

            std::wstring runtimePath(modulePath.data(), pathLength);
            const auto separator = runtimePath.find_last_of(L"\\/");
            if (separator == std::wstring::npos)
                throw std::runtime_error("Smart Gate module path has no directory");
            runtimePath.resize(separator + 1);
            runtimePath += L"onnxruntime.dll";

            // An absolute path plus LOAD_WITH_ALTERED_SEARCH_PATH also makes
            // dependent ONNX Runtime DLLs resolve from this private directory.
            runtimeModule = LoadLibraryExW(
                runtimePath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (runtimeModule == nullptr)
                throw std::runtime_error("Could not load the bundled ONNX Runtime beside Smart Gate");

            using GetApiBaseFunction = const OrtApiBase* (ORT_API_CALL*)();
            const auto getApiBase = reinterpret_cast<GetApiBaseFunction>(
                GetProcAddress(runtimeModule, "OrtGetApiBase"));
            if (getApiBase == nullptr)
                throw std::runtime_error("Bundled ONNX Runtime does not export OrtGetApiBase");

            const OrtApiBase* apiBase = getApiBase();
            const OrtApi* api = apiBase != nullptr ? apiBase->GetApi(ORT_API_VERSION) : nullptr;
            if (api == nullptr)
                throw std::runtime_error("Bundled ONNX Runtime does not support the requested C API version");

            Ort::InitApi(api);
        });
    }

    inline static int moduleAnchor_ = 0;

    void recordInferenceFailure() noexcept {
        inferenceFailureCount_.fetch_add(1, std::memory_order_relaxed);
    }

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;
    std::unique_ptr<Ort::Session> session_;
    std::array<float, contextSize> context_{};
    std::array<float, recurrentStateSize> recurrentState_{};
    std::array<float, modelInputSize> input_{};
    double expectedSourceFrameSamples_ = static_cast<double>(frameSize);
    std::uint64_t generation_ = 0;
    std::uint64_t lastSourceSamplePosition_ = 0;
    bool hasGeneration_ = false;
    bool hasSourcePosition_ = false;
    std::uint64_t successfulInferenceCount_ = 0;
    std::atomic<std::uint64_t> inferenceFailureCount_{0};
};

} // namespace smartgate
