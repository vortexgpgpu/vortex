`ifndef VX_PERF_PIPELINE_IF
`define VX_PERF_PIPELINE_IF

`include "VX_define.vh"

interface VX_perf_pipeline_if ();
    wire [`PERF_CTR_BITS-1:0]     ibf_stalls;
    wire [`PERF_CTR_BITS-1:0]     scb_stalls;
    wire [`PERF_CTR_BITS-1:0]     lsu_stalls;
    wire [`PERF_CTR_BITS-1:0]     csr_stalls;
    wire [`PERF_CTR_BITS-1:0]     alu_stalls;
    wire [`PERF_CTR_BITS-1:0]     gpu_stalls;
`ifdef EXT_F_ENABLE
    wire [`PERF_CTR_BITS-1:0]     fpu_stalls;
`endif
endinterface

`endif