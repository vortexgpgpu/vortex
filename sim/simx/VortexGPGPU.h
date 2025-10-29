// VortexGPGPU.h
#pragma once
#include <sst/core/component.h>
//#include <sst/core/interfaces/stdMem.h>
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

    void setup() override;
    void finish() override;

    // Register with SST
    SST_ELI_REGISTER_COMPONENT(
        VortexGPGPU,
        "vortex",           // element library name
        "VortexGPGPU",      // component name
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Vortex GPGPU Simulator",
        COMPONENT_CATEGORY_PROCESSOR
    )
    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "Clock frequency", "1GHz"},
        {"program", "Path to the kernel or ELF to load (defaults to built-in test image)", ""},
        {"launch_bytes", "Size in bytes of the default launch descriptor", "64"}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memIface", "StandardMem interface to memory hierarchy", "SST::Interfaces::StandardMem"}
    )

private:

    bool clockTick(SST::Cycle_t cycle);

    std::unique_ptr<vortex::VortexSimulator> sim_;

    //uint64_t launch_desc_addr_ = 0; // required only when launch descriptor is required

    #ifdef USE_SST_MEM
    void handleMemResp(SST::Interfaces::StandardMem::Request* req);
    
    // static pointer used by lambda in vx_register_submit()
    static VortexGPGPU* instance_;

    SST::Interfaces::StandardMem* memIface_;
    std::unordered_map<SST::Interfaces::StandardMem::Request::id_t, uint64_t> tag_by_id;
    //#else
    SST::Interfaces::StandardMem* memIface_ = nullptr;
    #endif
};

} // namespace Vortex
} // namespace SST
