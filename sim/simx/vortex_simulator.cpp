#include "vortex_simulator.h"
#include <fstream>
#include <vector>
#include <string>
#include "simobject.h"
#include "dcrs.h"

namespace vortex {

// Fallback macro definitions in case they are not provided by VX_config.h
#ifndef NUM_CLUSTERS
#define NUM_CLUSTERS 1
#endif
#ifndef NUM_CORES
#define NUM_CORES 1
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 1
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif
#ifndef RAM_PAGE_SIZE
#define RAM_PAGE_SIZE 4096
#endif
#ifndef STARTUP_ADDR
#define STARTUP_ADDR 0x0
#endif

static std::string getFileExt(const std::string& filename) {
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

VortexSimulator::VortexSimulator() : halted_(true) {}

bool VortexSimulator::init(const std::string& kernelPath) {
    // Initialize the architecture from macros or fallbacks
    arch_.num_clusters = NUM_CLUSTERS;
    arch_.num_cores    = NUM_CORES;
    arch_.num_warps    = NUM_WARPS;
    arch_.num_threads  = NUM_THREADS;
    arch_.global_mem_size = 1ULL << 30; // 1 GiB of global memory

    ram_ = RAM(arch_.global_mem_size, RAM_PAGE_SIZE);
    proc_ = std::make_unique<Processor>(arch_);
    proc_->attach_ram(&ram_);

    // Load a kernel binary if provided
    if (!kernelPath.empty()) {
        std::string ext = getFileExt(kernelPath);
        if (ext == "bin") {
            std::ifstream in(kernelPath, std::ios::binary);
            if (!in.good()) return false;
            std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
            ram_.loadBinImage(data.data(), data.size(), 0x0);
        } else if (ext == "hex") {
            std::ifstream in(kernelPath);
            if (!in.good()) return false;
            std::vector<uint8_t> bytes;
            std::string byteStr;
            while (in >> byteStr) {
                uint8_t val = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
                bytes.push_back(val);
            }
            ram_.loadBinImage(bytes.data(), bytes.size(), 0x0);
        } else {
            return false;
        }
    }

    // Write start address to DCRs for each cluster
    for (uint32_t cid = 0; cid < arch_.num_clusters; ++cid) {
        proc_->impl_->dcr_write(cid, DCR_LSU_BASE, STARTUP_ADDR);
        proc_->impl_->dcr_write(cid, DCR_HALT, 0);
    }

    halted_ = false;
    return true;
}

bool VortexSimulator::cycle() {
    if (halted_) return false;
    SimPlatform::instance().tick();
    bool anyRunning = false;
    for (auto cluster : proc_->impl_->clusters_) {
        if (cluster->running()) {
            anyRunning = true;
            break;
        }
    }
    halted_ = !anyRunning;
    return !halted_;
}

bool VortexSimulator::isHalted() const {
    return halted_;
}

} // namespace vortex