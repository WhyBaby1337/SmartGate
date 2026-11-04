#include "GateEnvelope.h"
#include <cmath>
#include <stdexcept>
namespace { void require(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); } }
void runGateEnvelopeTests() {
    smartgate::GateEnvelope e; e.prepare(48000.0); e.setThresholdDb(-40.0f); e.setTimes(2.0f,80.0f);
    require(e.process(0.0f)<1.0f,"Below-threshold input should close envelope");
    e.reset(); for(int i=0;i<2000;++i) e.process(1.0f);
    require(e.getGain()>0.99f,"Above-threshold input should open envelope");
    for(int i=0;i<20000;++i) e.process(0.0f);
    require(e.getGain()<0.01f,"Release should eventually close envelope");
    e.setThresholdDb(0.0f);
    require(std::isfinite(e.process(0.5f)),"Envelope output must remain finite");
}
