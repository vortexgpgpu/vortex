`include "VX_define.vh"

interface VX_sched_csr_if ();

    wire [`PERF_CTR_BITS-1:0] cycles;

    modport master (
        output cycles
    );

    modport slave (
        input  cycles
    );

endinterface
