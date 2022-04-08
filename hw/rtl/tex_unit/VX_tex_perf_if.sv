`ifndef VX_TEX_PERF_IF
`define VX_TEX_PERF_IF

`include "VX_define.vh"

import VX_tex_types::*;

interface VX_tex_perf_if ();

    wire [`PERF_CTR_BITS-1:0] mem_reads;
    wire [`PERF_CTR_BITS-1:0] mem_latency;

    modport master (
        output mem_reads,
        output mem_latency
    );

    modport slave (
        input mem_reads,
        input mem_latency
    );

endinterface

`endif
