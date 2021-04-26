`ifndef VX_PERF_MEMSYS_IF
`define VX_PERF_MEMSYS_IF

`include "VX_define.vh"

interface VX_perf_memsys_if ();

    wire [`PERF_CTR_BITS-1:0] icache_reads;
    wire [`PERF_CTR_BITS-1:0] icache_read_misses;
    wire [`PERF_CTR_BITS-1:0] icache_pipe_stalls;
    wire [`PERF_CTR_BITS-1:0] icache_crsp_stalls;

    wire [`PERF_CTR_BITS-1:0] dcache_reads;
    wire [`PERF_CTR_BITS-1:0] dcache_writes;    
    wire [`PERF_CTR_BITS-1:0] dcache_read_misses;
    wire [`PERF_CTR_BITS-1:0] dcache_write_misses;
    wire [`PERF_CTR_BITS-1:0] dcache_bank_stalls;
    wire [`PERF_CTR_BITS-1:0] dcache_mshr_stalls;
    wire [`PERF_CTR_BITS-1:0] dcache_pipe_stalls;
    wire [`PERF_CTR_BITS-1:0] dcache_crsp_stalls;

    wire [`PERF_CTR_BITS-1:0] smem_reads;
    wire [`PERF_CTR_BITS-1:0] smem_writes;
    wire [`PERF_CTR_BITS-1:0] smem_bank_stalls;
    
    wire [`PERF_CTR_BITS-1:0] mem_reads;
    wire [`PERF_CTR_BITS-1:0] mem_writes;
    wire [`PERF_CTR_BITS-1:0] mem_stalls;    
    wire [`PERF_CTR_BITS-1:0] mem_latency;

endinterface

`endif