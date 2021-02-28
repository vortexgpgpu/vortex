`ifndef VX_PERF_CACHE_IF
`define VX_PERF_CACHE_IF

`include "VX_define.vh"

interface VX_perf_cache_if ();

    wire [43:0] reads;
    wire [43:0] writes;
    wire [43:0] read_misses;
    wire [43:0] write_misses;
    wire [43:0] bank_stalls;
    wire [43:0] mshr_stalls;
    wire [43:0] pipe_stalls;
    wire [43:0] crsp_stalls;

endinterface

`endif