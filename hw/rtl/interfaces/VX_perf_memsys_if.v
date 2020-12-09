`ifndef VX_PERF_MEMSYS_IF
`define VX_PERF_MEMSYS_IF

`include "VX_define.vh"

interface VX_perf_memsys_if ();

    VX_perf_cache_if dcache_if;
    VX_perf_cache_if icache_if;
    
    wire [63:0] dram_latency;
    wire [63:0] dram_requests;
    wire [63:0] dram_responses;

endinterface

`endif