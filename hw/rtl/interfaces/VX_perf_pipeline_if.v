`ifndef VX_PERF_PIPELINE_IF
`define VX_PERF_PIPELINE_IF

`include "VX_define.vh"

interface VX_perf_pipeline_if ();
    wire [43:0]     ibf_stalls;
    wire [43:0]     scb_stalls;
    wire [43:0]     lsu_stalls;
    wire [43:0]     csr_stalls;
    wire [43:0]     alu_stalls;
    wire [43:0]     gpu_stalls;
`ifdef EXT_F_ENABLE
    wire [43:0]     fpu_stalls;
`endif
endinterface

`endif