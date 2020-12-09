`ifndef VX_PERF_CACHE_IF
`define VX_PERF_CACHE_IF

`include "VX_define.vh"

interface VX_perf_cache_if ();

    wire [63:0] reads;
    wire [63:0] writes;    
    wire [63:0] read_misses;
    wire [63:0] write_misses;
    wire [63:0] evictions;
    wire [63:0] mshr_stalls;
    wire [63:0] crsp_stalls;
    wire [63:0] dreq_stalls;
    wire [63:0] pipe_stalls;    

endinterface

`endif