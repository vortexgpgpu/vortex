`include "VX_define.vh"

interface VX_mem_perf_if ();

    wire [`PERF_CTR_BITS-1:0] icache_reads;
    wire [`PERF_CTR_BITS-1:0] icache_read_misses;

    wire [`PERF_CTR_BITS-1:0] dcache_reads;
    wire [`PERF_CTR_BITS-1:0] dcache_writes;
    wire [`PERF_CTR_BITS-1:0] dcache_read_misses;
    wire [`PERF_CTR_BITS-1:0] dcache_write_misses;
    wire [`PERF_CTR_BITS-1:0] dcache_bank_stalls;
    wire [`PERF_CTR_BITS-1:0] dcache_mshr_stalls;
    
    wire [`PERF_CTR_BITS-1:0] smem_reads;
    wire [`PERF_CTR_BITS-1:0] smem_writes;
    wire [`PERF_CTR_BITS-1:0] smem_bank_stalls;

    wire [`PERF_CTR_BITS-1:0] l2cache_reads;
    wire [`PERF_CTR_BITS-1:0] l2cache_writes;
    wire [`PERF_CTR_BITS-1:0] l2cache_read_misses;
    wire [`PERF_CTR_BITS-1:0] l2cache_write_misses;
    wire [`PERF_CTR_BITS-1:0] l2cache_bank_stalls;
    wire [`PERF_CTR_BITS-1:0] l2cache_mshr_stalls;

    wire [`PERF_CTR_BITS-1:0] l3cache_reads;
    wire [`PERF_CTR_BITS-1:0] l3cache_writes;
    wire [`PERF_CTR_BITS-1:0] l3cache_read_misses;
    wire [`PERF_CTR_BITS-1:0] l3cache_write_misses;
    wire [`PERF_CTR_BITS-1:0] l3cache_bank_stalls;
    wire [`PERF_CTR_BITS-1:0] l3cache_mshr_stalls;

    wire [`PERF_CTR_BITS-1:0] mem_reads;
    wire [`PERF_CTR_BITS-1:0] mem_writes;
    wire [`PERF_CTR_BITS-1:0] mem_latency;

    modport master (
        output icache_reads,
        output icache_read_misses,

        output dcache_reads,
        output dcache_writes,
        output dcache_read_misses,
        output dcache_write_misses,
        output dcache_bank_stalls,
        output dcache_mshr_stalls,

        output smem_reads,
        output smem_writes,
        output smem_bank_stalls,

        output l2cache_reads,
        output l2cache_writes,
        output l2cache_read_misses,
        output l2cache_write_misses,
        output l2cache_bank_stalls,
        output l2cache_mshr_stalls,

        output l3cache_reads,
        output l3cache_writes,
        output l3cache_read_misses,
        output l3cache_write_misses,
        output l3cache_bank_stalls,
        output l3cache_mshr_stalls,

        output mem_reads,
        output mem_writes,
        output mem_latency
    );

    modport slave (
        input icache_reads,
        input icache_read_misses,

        input dcache_reads,
        input dcache_writes,
        input dcache_read_misses,
        input dcache_write_misses,
        input dcache_bank_stalls,
        input dcache_mshr_stalls,

        input smem_reads,
        input smem_writes,
        input smem_bank_stalls,

        input l2cache_reads,
        input l2cache_writes,
        input l2cache_read_misses,
        input l2cache_write_misses,
        input l2cache_bank_stalls,
        input l2cache_mshr_stalls,

        input l3cache_reads,
        input l3cache_writes,
        input l3cache_read_misses,
        input l3cache_write_misses,
        input l3cache_bank_stalls,
        input l3cache_mshr_stalls,
        
        input mem_reads,
        input mem_writes,
        input mem_latency
    );

endinterface
