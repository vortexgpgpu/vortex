#include <sst/core/sst_config.h>
#include "VortexGPGPU.h"
#include "mem_backend_sst.h"   // needed for vx_register_submit and vx_on_mem_complete

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

    // Register callback so SimX can submit memory to SST
    instance_ = this;
    vx_register_submit(+[](uint64_t addr, bool write, uint32_t size, uint64_t tag) {
        if (write) {
            std::vector<uint8_t> zero(size, 0);
            auto *req = new StandardMem::Write(addr, zero);
            req->setDst(tag);
            instance_->memIface_->send(req);
        } else {
            auto *req = new StandardMem::Read(addr, size);
            req->setDst(tag);
            instance_->memIface_->send(req);
        }
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
    vx_on_mem_complete(req->getDst());
    delete req;
}

// Register with SST
SST_ELI_REGISTER_COMPONENT(
    VortexGPGPU,
    "vortex",           // element library name
    "VortexGPGPU",      // component name
    SST_ELI_ELEMENT_VERSION(1,0,0),
    "Headless Vortex GPGPU Simulator",
    COMPONENT_CATEGORY_PROCESSOR
)
SST_ELI_DOCUMENT_PARAMS(
    {"clock", "Clock frequency", "1GHz"},
    {"program", "Path to the kernel or ELF to load", ""}
)
SST_ELI_DOCUMENT_PORTS(
    {"memIface", "StandardMem port to connect to the SST memory hierarchy", {}}
)
