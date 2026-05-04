// vortex_simulator.h
//
// Thin wrapper that owns a v3 Processor + RAM and exposes a single-cycle
// step() entry point for the SST integration in vortex_gpgpu.cpp.
//
// Differences from upstream PR #298:
//   - No `Arch` member (v3 deleted the class; sizing is via macros).
//   - `Processor()` constructor takes no args on v3.
//   - DCR layout uses VX_DCR_KMU_* (KMU dispatch) rather than
//     VX_DCR_BASE_*; the KMU also needs grid/block dims set up before
//     the first cycle so warps actually launch.

#pragma once

#include "processor.h"
#include "constants.h"
#include <cstdint>
#include <memory>
#include <string>

namespace SST { namespace Interfaces { class StandardMem; } }

namespace vortex {

class VortexSimulator {
public:
    VortexSimulator();

    // Loads the kernel image at @p kernelPath and primes the KMU DCRs.
    // Returns false if the image extension is not vxbin/bin/hex.
    bool init(const std::string& kernelPath);

    // Advances the simulation by one cycle. Returns false once nothing
    // is running (program has completed and channels are drained).
    bool cycle();

    bool isHalted() const;

    // Phase 3 SST integration: register the SST memHierarchy interface so
    // every accepted memory request is mirrored to it (timing-only). Pass
    // nullptr to disable. Called by VortexGPGPU after init().
    void set_sst_mem_iface(SST::Interfaces::StandardMem* iface);

private:
    RAM ram_;
    std::unique_ptr<Processor> proc_;
    bool halted_;
};

} // namespace vortex
