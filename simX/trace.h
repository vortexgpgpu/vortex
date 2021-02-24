
#pragma once

namespace vortex {

  struct trace_inst_t {
  	// Warp step
    bool       valid_inst;
    unsigned   pc;

    // Core scheduler
    int        wid;

    // Encoder
    int        rs1;
    int        rs2;
    int        rs3;
    int        rd;

    //Encoder
    int        vs1;
    int        vs2;
    int        vd;

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