#include "vortex_simulator.h"
#include <iostream>
#include <string>
#include <util.h>
#include <VX_config.h>
#include <VX_types.h>

namespace vortex {

VortexSimulator::VortexSimulator()
: ram_(0, MEM_PAGE_SIZE)
, proc_(std::make_unique<Processor>())
, halted_(true) {}

bool VortexSimulator::init(const std::string& kernelPath) {
    proc_->attach_ram(&ram_);

    // Prime KMU DCRs the same way main.cpp does — the KMU needs the
    // startup address and a 1×1×1 grid/block to launch a single CTA.
    const uint64_t startup_addr(STARTUP_ADDR);
    proc_->dcr_write(VX_DCR_KMU_STARTUP_ADDR0, startup_addr & 0xffffffff);
#if (XLEN == 64)
    proc_->dcr_write(VX_DCR_KMU_STARTUP_ADDR1, startup_addr >> 32);
#endif
    proc_->dcr_write(VX_DCR_KMU_STARTUP_ARG0, 0);
    proc_->dcr_write(VX_DCR_KMU_STARTUP_ARG1, 0);
    proc_->dcr_write(VX_DCR_KMU_GRID_DIM_X,   1);
    proc_->dcr_write(VX_DCR_KMU_GRID_DIM_Y,   1);
    proc_->dcr_write(VX_DCR_KMU_GRID_DIM_Z,   1);
    proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_X,  1);
    proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_Y,  1);
    proc_->dcr_write(VX_DCR_KMU_BLOCK_DIM_Z,  1);
    proc_->dcr_write(VX_DCR_KMU_LMEM_SIZE,    0);
    proc_->dcr_write(VX_DCR_KMU_BLOCK_SIZE,   1);
    proc_->dcr_write(VX_DCR_KMU_WARP_STEP_X,  NUM_THREADS);
    proc_->dcr_write(VX_DCR_KMU_WARP_STEP_Y,  0);
    proc_->dcr_write(VX_DCR_KMU_WARP_STEP_Z,  0);

    {
      std::string program_ext(fileExtension(kernelPath.c_str()));
      if (program_ext == "vxbin") {
        std::cout << "vortex_simulator: Loading vxbin image: " << kernelPath << std::endl;
        ram_.loadVxImage(kernelPath.c_str());
      } else if (program_ext == "bin") {
        std::cout << "vortex_simulator: Loading binary image: " << kernelPath << std::endl;
        ram_.loadBinImage(kernelPath.c_str(), startup_addr);
      } else if (program_ext == "hex") {
        std::cout << "vortex_simulator: Loading hex image: " << kernelPath << std::endl;
        ram_.loadHexImage(kernelPath.c_str());
      } else {
        std::cerr << "Error: only *.vxbin, *.bin or *.hex images supported." << std::endl;
        return false;
      }
    }

    halted_ = false;
    return true;
}

bool VortexSimulator::cycle() {
    if (halted_) return false;
    bool running = proc_->cycle();
    halted_ = !running;
    return running;
}

bool VortexSimulator::isHalted() const {
    return halted_;
}

} // namespace vortex
