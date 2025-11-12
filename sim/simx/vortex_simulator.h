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

    /**
     * Advances the simulation by one cycle.  Returns false once the
     * simulation has completed (i.e. all clusters are halted).
     */
    bool cycle();

    /** Returns true if the simulation has finished. */
    bool isHalted() const;

private:

    Arch arch_;
    RAM ram_;
    std::unique_ptr<Processor> proc_;
    bool halted_;
};

} // namespace vortex
