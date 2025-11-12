#include "vortex_simulator.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "simobject.h"
#include "dcrs.h"
#include <VX_config.h>
#include <VX_types.h>
#include "util.h"

namespace vortex {

VortexSimulator::VortexSimulator()
: arch_(NUM_THREADS, NUM_WARPS, NUM_CORES)
, ram_(0, MEM_PAGE_SIZE)
, proc_(std::make_unique<Processor>(arch_))
, halted_(true) {}

bool VortexSimulator::init(const std::string& kernelPath) {
    proc_->attach_ram(&ram_);

    // setup base DCRs
    const uint64_t startup_addr(STARTUP_ADDR);
    proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR0, startup_addr & 0xffffffff);
    #if (XLEN == 64)
        proc_->dcr_write(VX_DCR_BASE_STARTUP_ADDR1, startup_addr >> 32);
    #endif
    proc_->dcr_write(VX_DCR_BASE_MPM_CLASS, 0);

    // load program/kernel
    {
      std::string program_ext(fileExtension(kernelPath.c_str()));
      if (program_ext == "bin") {
        std::cout << "vortex_simulator: Loading binary image: " << kernelPath << " with startup address: 0x" << std::hex << startup_addr << std::dec << std::endl;
        ram_.loadBinImage(kernelPath.c_str(), startup_addr);
      } else if (program_ext == "hex") {
        std::cout << "vortex_simulator: Loading hex image: " << kernelPath << std::endl;
        ram_.loadHexImage(kernelPath.c_str());
      } else {
        std::cerr << "Error: only *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }

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
