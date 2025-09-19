#include "vortex_simulator.h"
#include <fstream>
#include <vector>
#include <string>
#include "simobject.h"
#include "dcrs.h"
#include <VX_config.h>
#include <VX_types.h>

namespace vortex {

// Utility to extract file extension
static std::string getFileExt(const std::string& filename) {
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

VortexSimulator::VortexSimulator()
: arch_(NUM_THREADS, NUM_WARPS, NUM_CORES)
, ram_(0, MEM_PAGE_SIZE)
, proc_(std::make_unique<Processor>(arch_))
, halted_(true) {}

bool VortexSimulator::init(const std::string& kernelPath) {
    proc_->attach_ram(&ram_);

    // Load the kernel image if provided
    // Load the kernel image if provided
    if (!kernelPath.empty()) {
        std::string ext = getFileExt(kernelPath);
        if (ext == "bin") {
            // Load raw binary at STARTUP_ADDR
            ram_.loadBinImage(kernelPath.c_str(), STARTUP_ADDR);
        } else if (ext == "hex") {
            // Load Intel-hex
            ram_.loadHexImage(kernelPath.c_str());
        } else {
            return false; // unsupported format
        }
    }

    // Program base DCRs (match main.cpp behavior)
    const uint64_t startup = STARTUP_ADDR;
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, startup & 0xffffffffu);

    #if (XLEN == 64)
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, startup >> 32);
    #endif
    proc_->dcr_write(VX_DCR_BASE_MPM_CLASS, 0);

    halted_ = false;
    return true;
}

bool VortexSimulator::cycle() {
if (halted_) return false;
// Advance one cycle through the processor interface
bool running = proc_->cycle(); 
halted_ = !running;
return running;
}

bool VortexSimulator::isHalted() const {
    return halted_;
}

} // namespace vortex
