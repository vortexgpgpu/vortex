
#pragma once

namespace Harp {

  typedef struct
  {
  	// Warp step
    bool       valid_inst;
    unsigned   pc;

    // Core scheduler
    int        wid;

    // Encoder
    int        rs1;
    int        rs2;
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
    int        mem_stall_cycles;
    int        fetch_stall_cycles;

    // Instruction execute
    bool stall_warp;
    bool wspawn;

    bool stalled;
  } trace_inst_t;

}