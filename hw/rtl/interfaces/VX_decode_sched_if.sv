`include "VX_define.vh"

interface VX_decode_sched_if ();

    wire                    valid;
    wire                    is_wstall;
    wire [`UP(`NW_BITS)-1:0] wid;

    modport master (
        output valid,
        output is_wstall,
        output wid
    );

    modport slave (
        input valid,
        input is_wstall,
        input wid
    );

endinterface
