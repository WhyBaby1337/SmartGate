cmake_minimum_required(VERSION 3.24)
project(SmartGate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(Threads REQUIRED)

add_library(smartgate_dsp INTERFACE)
target_include_directories(smartgate_dsp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Source)

add_executable(SmartGateDspTests
    tests/main.cpp
    tests/GateEnvelopeTests.cpp
    tests/TargetProfileTests.cpp
    tests/AdaptiveKeyboardDetectorTests.cpp
    tests/VadFrameQueueTests.cpp
    tests/VadResamplerTests.cpp
    tests/VadFrameAssemblerTests.cpp
    tests/VadInferenceWorkerTests.cpp)
target_link_libraries(SmartGateDspTests PRIVATE smartgate_dsp Threads::Threads)
add_test(NAME SmartGateDspTests COMMAND SmartGateDspTests)

add_executable(SmartGateDspBenchmark tests/DspBenchmark.cpp)
target_link_libraries(SmartGateDspBenchmark PRIVATE smartgate_dsp)

option(SMARTGATE_BUILD_PLUGIN "Build the JUCE VST3 plugin" ON)
option(SMARTGATE_ENABLE_SILERO "Fetch and build the embedded Silero VAD engine" ${WIN32})

if(SMARTGATE_BUILD_PLUGIN)
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.12
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)

    if(SMARTGATE_ENABLE_SILERO)
        if(NOT WIN32)
            message(FATAL_ERROR "The pinned ONNX Runtime binary package is currently configured for Windows x64 only.")
        endif()

        FetchContent_Declare(onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.30.1/onnxruntime-win-x64-1.30.1.zip
            URL_HASH SHA256=9e70d829a7c2014c6f6d2889cd3578733c2e5bbe919e186b5f2038d59176ab13
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_GetProperties(onnxruntime)
        if(NOT onnxruntime_POPULATED)
            FetchContent_Populate(onnxruntime)
        endif()

        set(SMARTGATE_ORT_ROOT "${onnxruntime_SOURCE_DIR}")
        if(NOT EXISTS "${SMARTGATE_ORT_ROOT}/include/onnxruntime_cxx_api.h")
            set(SMARTGATE_ORT_ROOT "${onnxruntime_SOURCE_DIR}/onnxruntime-win-x64-1.30.1")
        endif()
        if(NOT EXISTS "${SMARTGATE_ORT_ROOT}/include/onnxruntime_cxx_api.h"
           OR NOT EXISTS "${SMARTGATE_ORT_ROOT}/lib/onnxruntime.lib"
           OR NOT EXISTS "${SMARTGATE_ORT_ROOT}/lib/onnxruntime.dll")
            message(FATAL_ERROR "The pinned ONNX Runtime archive does not contain the expected Windows x64 SDK layout.")
        endif()

        add_library(SmartGateONNXRuntime SHARED IMPORTED GLOBAL)
        set_target_properties(SmartGateONNXRuntime PROPERTIES
            IMPORTED_IMPLIB "${SMARTGATE_ORT_ROOT}/lib/onnxruntime.lib"
            IMPORTED_LOCATION "${SMARTGATE_ORT_ROOT}/lib/onnxruntime.dll")

        set(SILERO_MODEL_PATH "${CMAKE_BINARY_DIR}/_deps/silero_vad.onnx")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
        if(NOT EXISTS "${SILERO_MODEL_PATH}")
            file(DOWNLOAD
                https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx
                "${SILERO_MODEL_PATH}"
                TLS_VERIFY ON
                SHOW_PROGRESS
                STATUS SILERO_DOWNLOAD_STATUS)
            list(GET SILERO_DOWNLOAD_STATUS 0 SILERO_DOWNLOAD_CODE)
            if(NOT SILERO_DOWNLOAD_CODE EQUAL 0)
                list(GET SILERO_DOWNLOAD_STATUS 1 SILERO_DOWNLOAD_MESSAGE)
                message(FATAL_ERROR "Failed to download the pinned Silero VAD model: ${SILERO_DOWNLOAD_MESSAGE}")
            endif()
        endif()

        file(SIZE "${SILERO_MODEL_PATH}" SILERO_MODEL_SIZE)
        if(NOT SILERO_MODEL_SIZE EQUAL 2327524)
            message(FATAL_ERROR "Silero model size mismatch: expected 2327524 bytes, got ${SILERO_MODEL_SIZE}.")
        endif()
        execute_process(
            COMMAND git hash-object "${SILERO_MODEL_PATH}"
            RESULT_VARIABLE SILERO_HASH_RESULT
            OUTPUT_VARIABLE SILERO_MODEL_BLOB_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT SILERO_HASH_RESULT EQUAL 0
           OR NOT SILERO_MODEL_BLOB_HASH STREQUAL "80c5592ef1f4c9ede3e357bbd02eb863358a6a9d")
            message(FATAL_ERROR "Pinned Silero model Git blob hash verification failed.")
        endif()

        juce_add_binary_data(SileroVADModel SOURCES "${SILERO_MODEL_PATH}")
    endif()

    juce_add_plugin(SmartGate
        COMPANY_NAME "SmartGate Audio"
        BUNDLE_ID "audio.smartgate.plugin"
        PLUGIN_MANUFACTURER_CODE SmGt
        PLUGIN_CODE SmG1
        FORMATS VST3
        PRODUCT_NAME "Smart Gate"
        DESCRIPTION "Target-aware noise gate"
        IS_SYNTH FALSE NEEDS_MIDI_INPUT FALSE NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE COPY_PLUGIN_AFTER_BUILD FALSE)

    target_sources(SmartGate PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/SileroVadInference.h)
    target_compile_definitions(SmartGate PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)
    target_link_libraries(SmartGate PRIVATE smartgate_dsp juce::juce_audio_utils juce::juce_dsp
        PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags
               juce::juce_recommended_warning_flags)

    if(SMARTGATE_ENABLE_SILERO)
        target_include_directories(SmartGate PRIVATE "${SMARTGATE_ORT_ROOT}/include")
        target_compile_definitions(SmartGate PRIVATE SMARTGATE_HAS_SILERO=1)
        target_link_libraries(SmartGate PRIVATE SileroVADModel SmartGateONNXRuntime)
        add_custom_command(TARGET SmartGate POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SmartGateONNXRuntime>"
                "$<TARGET_FILE_DIR:SmartGate>")

        add_executable(SmartGateSileroTests tests/SileroVadSmokeTests.cpp)
        target_include_directories(SmartGateSileroTests PRIVATE "${SMARTGATE_ORT_ROOT}/include")
        target_link_libraries(SmartGateSileroTests PRIVATE SileroVADModel SmartGateONNXRuntime)
        add_custom_command(TARGET SmartGateSileroTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SmartGateONNXRuntime>"
                "$<TARGET_FILE_DIR:SmartGateSileroTests>")
        add_test(NAME SmartGateSileroTests COMMAND SmartGateSileroTests)
    endif()
endif()
