// VortexGPGPU.h
#pragma once
#include <sst/core/component.h>
#include <sst/core/interfaces/stdMem.h>
#include <memory>
#include <string>
#include "vortex_simulator.h"  // wrapper around SimX

namespace SST {
namespace Vortex {

class VortexGPGPU : public SST::Component {
public:
    VortexGPGPU(SST::ComponentId_t id, SST::Params& params);
    ~VortexGPGPU() override;

    void setup() override;
    void finish() override;

private:
    bool clockTick(SST::Cycle_t cycle);
    void handleMemResp(SST::Interfaces::StandardMem::Request* req);

    // static pointer used by lambda in vx_register_submit()
    static VortexGPGPU* instance_;

    std::unique_ptr<vortex::VortexSimulator> sim_;
    SST::Interfaces::StandardMem* memIface_;
};

} // namespace Vortex
} // namespace SST
