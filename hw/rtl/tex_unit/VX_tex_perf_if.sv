`include "VX_define.vh"

interface VX_tex_perf_if ();

    wire [`PERF_CTR_BITS-1:0] mem_reads;
    wire [`PERF_CTR_BITS-1:0] mem_latency;
    wire [`PERF_CTR_BITS-1:0] stall_cycles;

    modport master (
        output mem_reads,
        output mem_latency,
        output stall_cycles
    );

    modport slave (
        input mem_reads,
        input mem_latency,
        input stall_cycles
    );

endinterface
