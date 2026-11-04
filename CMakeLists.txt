cmake_minimum_required(VERSION 3.24)
project(SmartGate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_library(smartgate_dsp INTERFACE)
target_include_directories(smartgate_dsp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/Source)

add_executable(SmartGateDspTests
    tests/main.cpp
    tests/GateEnvelopeTests.cpp
    tests/TargetProfileTests.cpp
    tests/AdaptiveKeyboardDetectorTests.cpp)
target_link_libraries(SmartGateDspTests PRIVATE smartgate_dsp)
add_test(NAME SmartGateDspTests COMMAND SmartGateDspTests)

add_executable(SmartGateDspBenchmark tests/DspBenchmark.cpp)
target_link_libraries(SmartGateDspBenchmark PRIVATE smartgate_dsp)

option(SMARTGATE_BUILD_PLUGIN "Build the JUCE VST3 plugin" ON)
if(SMARTGATE_BUILD_PLUGIN)
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.12
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(JUCE)

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

    target_sources(SmartGate PRIVATE Source/PluginProcessor.cpp Source/PluginProcessor.h)
    # This is a new VST3-only product; it has no released VST2 identity to replace.
    target_compile_definitions(SmartGate PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)
    target_link_libraries(SmartGate PRIVATE smartgate_dsp juce::juce_audio_utils juce::juce_dsp
        PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags
               juce::juce_recommended_warning_flags)
endif()
