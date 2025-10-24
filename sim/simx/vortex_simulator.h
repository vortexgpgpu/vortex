// vortex_simulator.h
#pragma once

#include "processor.h"  // for Processor, RAM
#include "arch.h"       // for Arch
#include "constants.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace vortex {

struct KernelImageInfo {
    uint64_t base_addr = 0;
    uint64_t size_bytes = 0;
};

/**
 * A wrapper class used by the SST integration to drive the Vortex GPU
 * one cycle at a time.  It encapsulates the architecture definition,
 * memory subsystem, and processor instance.
 */
class VortexSimulator {
public:
    VortexSimulator();

    /**
     * Initializes the simulator.  If @p kernelPath is non-empty, the
     * kernel image at the given path will be loaded into memory.
     * Returns false if the image format is not supported.
     */
    bool init(const std::string& kernelPath);

    // changes to substitute for run-time wrt memory setup
    const KernelImageInfo& kernelImage() const { return kernel_image_; }
    bool allocateMemory(uint64_t size, uint64_t alignment, bool readable, bool writable, uint64_t* addr_out);
    bool reserveMemory(uint64_t addr, uint64_t size, bool readable, bool writable);
    void setMemoryPermissions(uint64_t addr, uint64_t size, bool readable, bool writable);
    void writeMemory(uint64_t addr, const void* data, uint64_t size);

    RAM& ram() { return ram_; }
    const RAM& ram() const { return ram_; }

    void setStartupArg(uint64_t arg_addr);

    /**
     * Advances the simulation by one cycle.  Returns false once the
     * simulation has completed (i.e. all clusters are halted).
     */
    bool cycle();

    /** Returns true if the simulation has finished. */
    bool isHalted() const;

private:
    static constexpr uint64_t kGlobalMemSize = (XLEN == 64) ? 0x200000000ull : 0x100000000ull;
    static constexpr uint64_t kAllocBaseAddr = USER_BASE_ADDR;
    static constexpr uint64_t kDefaultAlignment = 64ull;

    static uint64_t alignUp(uint64_t value, uint64_t alignment);
    static uint64_t alignDown(uint64_t value, uint64_t alignment);
    static uint64_t normalizeAlignment(uint64_t alignment);

    std::optional<KernelImageInfo> loadKernelImage(const std::string& path);

    Arch arch_;
    RAM ram_;
    std::unique_ptr<Processor> proc_;
    KernelImageInfo kernel_image_;
    uint64_t next_alloc_addr_;
    bool halted_;
};

} // namespace vortex
