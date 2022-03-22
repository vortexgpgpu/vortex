`ifndef VX_PERF_PIPELINE_IF
`define VX_PERF_PIPELINE_IF

`include "VX_define.vh"

interface VX_perf_pipeline_if ();
    wire [`PERF_CTR_BITS-1:0]   loads;
    wire [`PERF_CTR_BITS-1:0]   stores;
    wire [`PERF_CTR_BITS-1:0]   branches;
    
    wire [`PERF_CTR_BITS-1:0]   ibf_stalls;
    wire [`PERF_CTR_BITS-1:0]   scb_stalls;
    wire [`PERF_CTR_BITS-1:0]   lsu_stalls;
    wire [`PERF_CTR_BITS-1:0]   csr_stalls;
    wire [`PERF_CTR_BITS-1:0]   alu_stalls;
`ifdef EXT_F_ENABLE
    wire [`PERF_CTR_BITS-1:0]   fpu_stalls;
`endif
    wire [`PERF_CTR_BITS-1:0]   gpu_stalls;

    modport decode (
        output loads,
        output stores,
        output branches
    );

    modport issue (
        output ibf_stalls,
        output scb_stalls,
        output lsu_stalls,
        output csr_stalls,
        output alu_stalls,
    `ifdef EXT_F_ENABLE
        output fpu_stalls,
    `endif
        output gpu_stalls
    );    

    modport slave (
        input loads,
        input stores,
        input branches,
        input ibf_stalls,
        input scb_stalls,
        input lsu_stalls,
        input csr_stalls,
        input alu_stalls,
    `ifdef EXT_F_ENABLE
        input fpu_stalls,
    `endif
        input gpu_stalls
    );

endinterface

`endif
