`ifndef VX_PERF_PIPELINE_IF
`define VX_PERF_PIPELINE_IF

`include "VX_define.vh"

interface VX_perf_pipeline_if ();
    // from pipeline
    wire [63:0]     icache_stalls;
    wire [63:0]     ibuffer_stalls;
    // from issue
    wire [63:0]     scoreboard_stalls;
    // from execute
    wire [63:0]     lsu_stalls;
    wire [63:0]     csr_stalls;
    wire [63:0]     alu_stalls;
    wire [63:0]     gpu_stalls;
`ifdef EXT_M_ENABLE
    wire [63:0]     mul_stalls;
`endif
`ifdef EXT_F_ENABLE
    wire [63:0]     fpu_stalls;
`endif
endinterface

`endif