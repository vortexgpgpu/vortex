// vortex_gpgpu.h
#pragma once
#include <sst/core/component.h>
#include <sst/core/interfaces/stdMem.h>
#include <memory>
#include <string>
#include "vortex_simulator.h"  // wrapper around SimX
#include <unordered_map>

namespace SST {
namespace Vortex {

class VortexGPGPU : public SST::Component {
public:
    VortexGPGPU(SST::ComponentId_t id, SST::Params& params);
    ~VortexGPGPU() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

    // Register with SST
    SST_ELI_REGISTER_COMPONENT(
        VortexGPGPU,
        "vortex",           // Element library name
        "VortexGPGPU",      // Component name
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Vortex GPGPU Simulator",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"program", "Path to the kernel or ELF to load", ""},
    )

    // for future usage with SST memory
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memIface", "StandardMem interface to memory hierarchy", "SST::Interfaces::StandardMem"}
    )

private:

    bool clockTick(SST::Cycle_t cycle);

    // Phase 3 SST integration: optional memHierarchy interface. If the
    // SST script connects a SubComponent to the "memIface" slot, every
    // MemReq accepted by the local DRAM model is mirrored to it as a
    // StandardMem::Read or Write event. Responses from SST are
    // acknowledged but not used (data path stays local).
    void handleMemRsp(SST::Interfaces::StandardMem::Request* req);

    std::unique_ptr<vortex::VortexSimulator> sim_;
    SST::Interfaces::StandardMem* memIface_ = nullptr;
};

} // namespace Vortex
} // namespace SST
