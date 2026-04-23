#include <sst/core/sst_config.h>
#include "vortex_gpgpu.h"
#include <cstdlib>
#include <vector>
#include <utility>
#include <unordered_map>

using namespace SST;
using namespace SST::Vortex;

VortexGPGPU::VortexGPGPU(ComponentId_t id, Params &params)
    : Component(id),
      sim_(std::make_unique<vortex::VortexSimulator>()) {

    std::cout << "VortexGPGPU Component: Initializing Vortex GPGPU simulator\n";

    // Parameter: clock frequency (default 1GHz)
    std::string clockfreq = params.find<std::string>("clock", "1GHz");

    // Parameter: program path
    std::string kernel = params.find<std::string>("program", "");

    // Register our clock handler with SST
    registerClock(clockfreq,
                  new SST::Clock::Handler<VortexGPGPU>(this, &VortexGPGPU::clockTick));

    // Load the kernel image
    if (!sim_->init(kernel)) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexSimulator init failed\n");
    }
    else{
        std::cout << "VortexGPGPU Component: loaded kernel: " << kernel << std::endl;
    }

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

VortexGPGPU::~VortexGPGPU() = default;

void VortexGPGPU::setup() {}
void VortexGPGPU::finish() {}

// Advance the GPU execution one cycle based on SST clock handler callback
bool VortexGPGPU::clockTick(SST::Cycle_t cycle) {
    bool running = sim_->cycle();
    if (!running) {
        primaryComponentOKToEndSim();
        std::cout << "VortexGPGPU Component: simulation finished\n";
        return true;
    }
    return false;
}
