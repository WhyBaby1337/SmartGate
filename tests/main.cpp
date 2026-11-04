#include <exception>
#include <iostream>

void runGateEnvelopeTests();
void runTargetProfileTests();
void runAdaptiveKeyboardDetectorTests();

int main() {
    try {
        runGateEnvelopeTests();
        runTargetProfileTests();
        runAdaptiveKeyboardDetectorTests();
        std::cout << "All SmartGate DSP tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
