`ifndef VX_PERF_PIPELINE_STALL_IF
`define VX_PERF_PIPELINE_STALL_IF

`include "VX_define.vh"

interface VX_perf_pipeline_stall_if ();
    // from pipeline
    wire [63:0]     icache_stall;
    wire [63:0]     ibuffer_stall;
    // from issue
    wire [63:0]     scoreboard_stall;
    // from execute
    wire [63:0]     lsu_stall;
    wire [63:0]     csr_stall;
    wire [63:0]     alu_stall;
    wire [63:0]     gpu_stall;
    `ifdef EXT_M_ENABLE
        wire [63:0]     mul_stall;
    `endif
    `ifdef EXT_F_ENABLE
        wire [63:0]     fpu_stall;
    `endif
endinterface

`endif