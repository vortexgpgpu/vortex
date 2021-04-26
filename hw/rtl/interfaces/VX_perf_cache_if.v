`ifndef VX_PERF_CACHE_IF
`define VX_PERF_CACHE_IF

`include "VX_define.vh"

interface VX_perf_cache_if ();

    wire [`PERF_CTR_BITS-1:0] reads;
    wire [`PERF_CTR_BITS-1:0] writes;
    wire [`PERF_CTR_BITS-1:0] read_misses;
    wire [`PERF_CTR_BITS-1:0] write_misses;
    wire [`PERF_CTR_BITS-1:0] bank_stalls;
    wire [`PERF_CTR_BITS-1:0] mshr_stalls;
    wire [`PERF_CTR_BITS-1:0] pipe_stalls;
    wire [`PERF_CTR_BITS-1:0] crsp_stalls;

endinterface

`endif