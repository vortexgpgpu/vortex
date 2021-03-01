`ifndef VX_PERF_MEMSYS_IF
`define VX_PERF_MEMSYS_IF

`include "VX_define.vh"

interface VX_perf_memsys_if ();

    wire [43:0] icache_reads;
    wire [43:0] icache_read_misses;
    wire [43:0] icache_pipe_stalls;
    wire [43:0] icache_crsp_stalls;

    wire [43:0] dcache_reads;
    wire [43:0] dcache_writes;    
    wire [43:0] dcache_read_misses;
    wire [43:0] dcache_write_misses;
    wire [43:0] dcache_bank_stalls;
    wire [43:0] dcache_mshr_stalls;
    wire [43:0] dcache_pipe_stalls;
    wire [43:0] dcache_crsp_stalls;

    wire [43:0] smem_reads;
    wire [43:0] smem_writes;
    wire [43:0] smem_bank_stalls;
    
    wire [43:0] dram_reads;
    wire [43:0] dram_writes;
    wire [43:0] dram_stalls;    
    wire [43:0] dram_latency;

endinterface

`endif