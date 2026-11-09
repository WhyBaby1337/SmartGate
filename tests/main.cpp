#include <exception>
#include <iostream>

void runGateEnvelopeTests();
void runTargetProfileTests();
void runAdaptiveKeyboardDetectorTests();
void runShadowEventClassifierTests();
void runVadFrameQueueTests();
void runVadResamplerTests();
void runVadFrameAssemblerTests();
void runVadInferenceWorkerTests();
void runLookaheadDelayTests();
void runVoiceLevelerTests();

int main() {
    try {
        runGateEnvelopeTests();
        runTargetProfileTests();
        runAdaptiveKeyboardDetectorTests();
        runShadowEventClassifierTests();
        runVadFrameQueueTests();
        runVadResamplerTests();
        runVadFrameAssemblerTests();
        runVadInferenceWorkerTests();
        runLookaheadDelayTests();
        runVoiceLevelerTests();
        std::cout << "All SmartGate DSP tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
