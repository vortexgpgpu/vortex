#include <sst/core/sst_config.h>
#include "VortexGPGPU.h"
#ifdef USE_SST_MEM
#include "mem_backend_sst.h" // needed for vx_register_submit and vx_on_mem_complete
#endif
#include <cstdlib>
#include <vector>
#include <utility>
#include <unordered_map>

using namespace SST;
using namespace SST::Vortex;
#ifdef USE_SST_MEM
using SST::Interfaces::StandardMem;
#endif

#ifdef USE_SST_MEM
VortexGPGPU *VortexGPGPU::instance_ = nullptr;
#endif

VortexGPGPU::VortexGPGPU(ComponentId_t id, Params &params)
    : Component(id),
      sim_(std::make_unique<vortex::VortexSimulator>()) {

    std::cout << "VortexGPGPU: initializing Vortex GPGPU simulator\n";

    // Parameter: clock frequency (default 1GHz)
    std::string clockfreq = params.find<std::string>("clock", "1GHz");

    // Parameter: program path
    std::string kernel = params.find<std::string>("program", "/nethome/jsubburayan3/vortex/build/tests/kernel/hello/hello.bin");

    //const uint32_t launch_bytes = params.find<uint32_t>("launch_bytes", kDefaultLaunchBytes); // required when launch descriptor is used

#ifdef USE_SST_MEM
    // Create StandardMem interface; auto-bind to port name "memIface"
    memIface_ = loadUserSubComponent<StandardMem>(
        "memIface", ComponentInfo::SHARE_NONE,
        registerClock(clockfreq,
                      new SST::Clock::Handler<VortexGPGPU>(this, &VortexGPGPU::clockTick)),
        new StandardMem::Handler<VortexGPGPU>(this, &VortexGPGPU::handleMemResp));

    if (!memIface_) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexGPGPU: failed to load memIface StandardMem port\n");
    }
#else
    // No SST memory: just register our clock handler
    registerClock(clockfreq,
                  new SST::Clock::Handler<VortexGPGPU>(this, &VortexGPGPU::clockTick));
#endif

#ifdef USE_SST_MEM
    // Register callback so SimX can submit memory to SST
    instance_ = this;
    // Track app-specific tags by StandardMem request-id
        // (e.g., inside your instance_ type)

        vx_register_submit(+[](uint64_t addr, bool write, uint32_t size, uint64_t tag) {

        StandardMem::Request* req = nullptr;

        if (write) {
            std::vector<uint8_t> zeros(static_cast<size_t>(size), 0);
            // posted=false so we get a WriteResp
            req = new StandardMem::Write(static_cast<StandardMem::Addr>(addr),
                                static_cast<uint64_t>(size),
                                std::move(zeros),
                                /*posted=*/false);
        } else {
            req = new StandardMem::Read(static_cast<StandardMem::Addr>(addr),
                            static_cast<uint64_t>(size));
        }

        // Use the StandardMem-assigned ID to correlate responses
        const auto id = req->getID();
        instance_->tag_by_id.emplace(id, tag);

        instance_->memIface_->send(req);
    });
#endif

    // Load the kernel image
    if (!sim_->init(kernel)) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexSimulator init failed\n");
    }
    else{
        std::cout << "VortexGPGPU: loaded kernel: " << kernel << std::endl;
    }

    // needed when launch descriptor is used
    /*
    if (!sim_->allocateMemory(launch_bytes, 64, true, true, &launch_desc_addr_)) {
        SST::Output out;
        out.fatal(CALL_INFO, -1,
                  "VortexGPGPU: unable to allocate launch descriptor (%u bytes)\n",
                  launch_bytes);
    }
    std::vector<uint8_t> launch_payload(launch_bytes, 0);
    sim_->writeMemory(launch_desc_addr_, launch_payload.data(), launch_payload.size());
    sim_->setStartupArg(launch_desc_addr_);
    */

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

VortexGPGPU::~VortexGPGPU() = default;

void VortexGPGPU::setup() {}
void VortexGPGPU::finish() {}

bool VortexGPGPU::clockTick(SST::Cycle_t cycle) {
    // Advance the GPU one cycle
    //std::cout << "VortexGPGPU: clockTick came from SST " << std::endl;
    bool running = sim_->cycle();
    //std::cout << "VortexGPGPU cycle returned: " << running << std::endl;
    if (!running) {
        primaryComponentOKToEndSim();
        std::cout << "VortexGPGPU: simulation finished\n";
        return true;
    }
    //std::cout << "VortexGPGPU clockTick returns false " << std::endl;
    return false;
}

#ifdef USE_SST_MEM
void VortexGPGPU::handleMemResp(StandardMem::Request *req) {
    // Inform SimX that this request has completed
    const auto id = req->getID();
    const auto it = tag_by_id.find(id);
    if (it == tag_by_id.end()) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexGPGPU: received response with unknown ID %lu\n", id);
    }
    else{
        vx_on_mem_complete(it->second);
        tag_by_id.erase(it);
    }
    delete req;
}
#endif
