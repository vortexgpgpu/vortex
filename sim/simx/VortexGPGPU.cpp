#include <sst/core/sst_config.h>
#include "VortexGPGPU.h"
#include "mem_backend_sst.h"   // needed for vx_register_submit and vx_on_mem_complete
#include <vector>
#include <utility>
#include <unordered_map>

using namespace SST;
using namespace SST::Vortex;
using SST::Interfaces::StandardMem;

VortexGPGPU *VortexGPGPU::instance_ = nullptr;

VortexGPGPU::VortexGPGPU(ComponentId_t id, Params &params)
    : Component(id),
      sim_(std::make_unique<vortex::VortexSimulator>()),
      memIface_(nullptr) {

    // Parameter: clock frequency (default 1GHz)
    std::string clockfreq = params.find<std::string>("clock", "1GHz");
    // Parameter: program path
    std::string kernel = params.find<std::string>("program", "");

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


    // Load the kernel or ELF
    if (!sim_->init(kernel)) {
        SST::Output out;
        out.fatal(CALL_INFO, -1, "VortexSimulator init failed\n");
    }

    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

VortexGPGPU::~VortexGPGPU() = default;

void VortexGPGPU::setup() {}
void VortexGPGPU::finish() {}

bool VortexGPGPU::clockTick(SST::Cycle_t) {
    // Advance the GPU one cycle
    bool running = sim_->cycle();
    if (!running) {
        primaryComponentOKToEndSim();
        return false;
    }
    return true;
}

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
