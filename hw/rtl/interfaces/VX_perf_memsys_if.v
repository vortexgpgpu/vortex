`ifndef VX_PERF_MEMSYS_IF
`define VX_PERF_MEMSYS_IF

`include "VX_define.vh"

interface VX_perf_memsys_if ();

    wire [63:0] icache_reads;
    wire [63:0] icache_read_misses;
    wire [63:0] icache_mshr_stalls;
    wire [63:0] icache_crsp_stalls;
    wire [63:0] icache_dreq_stalls;
    wire [63:0] icache_pipe_stalls;

    wire [63:0] dcache_reads;
    wire [63:0] dcache_writes;
    wire [63:0] dcache_read_misses;
    wire [63:0] dcache_write_misses;
    wire [63:0] dcache_evictions;
    wire [63:0] dcache_mshr_stalls;
    wire [63:0] dcache_crsp_stalls;
    wire [63:0] dcache_dreq_stalls;
    wire [63:0] dcache_pipe_stalls;
    
    wire [63:0] dram_latency;
    wire [63:0] dram_requests;
    wire [63:0] dram_responses;

endinterface

`endif