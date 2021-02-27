
#pragma once

namespace vortex {

  struct trace_inst_t {
  	// Warp step
    bool       valid;
    unsigned   PC;

    // Core scheduler
    int        wid;

    // Encoder
    int        irs1;
    int        irs2;    
    int        ird;

    // Floating-point
    int        frs1;
    int        frs2;
    int        frs3;
    int        frd;

    // Vector extension
    int        vrs1;
    int        vrs2;
    int        vrd;

    // Instruction execute
    bool       is_lw;
    bool       is_sw;
    unsigned * mem_addresses;

    // dmem interface
    unsigned long mem_stall_cycles;
    unsigned long fetch_stall_cycles;

    // Instruction execute
    bool stall_warp;
    bool wspawn;

    bool stalled;
  };
}