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

    // Register our clock handler with SST. Capture the TimeConverter so
    // any SubComponents (e.g. memIface) can share the same time domain.
    SST::TimeConverter* tc = registerClock(clockfreq,
                  new SST::Clock::Handler<VortexGPGPU>(this, &VortexGPGPU::clockTick));

    // Load the kernel image
    if (!sim_->init(kernel)) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexSimulator init failed\n");
    }
    else{
        std::cout << "VortexGPGPU Component: loaded kernel: " << kernel << std::endl;
    }

    // Phase 3: optional memHierarchy interface. Loads if the SST script
    // connects a SubComponent to the "memIface" slot; nullptr otherwise.
    memIface_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memIface", SST::ComponentInfo::SHARE_NONE, tc,
        new SST::Interfaces::StandardMem::Handler<VortexGPGPU>(this, &VortexGPGPU::handleMemRsp));
    if (memIface_) {
        std::cout << "VortexGPGPU Component: memIface attached — mirroring memory requests to SST memHierarchy\n";
        sim_->set_sst_mem_iface(memIface_);
    }

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

VortexGPGPU::~VortexGPGPU() = default;

// Phase 3 SST: forward init() phases to memIface so memHierarchy can
// discover destinations and exchange address-range info before the clock
// starts ticking. Without this, the first memIface->send() crashes with
// "MemLink cannot find a destination".
void VortexGPGPU::init(unsigned int phase) {
    if (memIface_) {
        memIface_->init(phase);
    }
}

void VortexGPGPU::setup() {
    if (memIface_) {
        memIface_->setup();
    }
}
void VortexGPGPU::finish() {
    if (memIface_) {
        memIface_->finish();
    }
}

void VortexGPGPU::handleMemRsp(SST::Interfaces::StandardMem::Request* req) {
    // Phase 3: SST sent us a Read/Write response. Local data path already
    // handled the request — this is timing-only acknowledgement. Just
    // free the request object.
    delete req;
}

// Advance the GPU execution one cycle based on SST clock handler callback
bool VortexGPGPU::clockTick(SST::Cycle_t cycle) {
    (void)cycle;  // SST passes the global cycle count; we don't use it.
    bool running = sim_->cycle();
    if (!running) {
        primaryComponentOKToEndSim();
        std::cout << "VortexGPGPU Component: simulation finished\n";
        return true;
    }
    return false;
}
