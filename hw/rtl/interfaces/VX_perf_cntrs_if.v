`ifndef VX_PERF_CNTRS_IF
`define VX_PERF_CNTRS_IF

`include "VX_define.vh"

interface VX_perf_cntrs_if ();

    wire [63:0] total_cycles;
    wire [63:0] total_instrs;

endinterface

`endif