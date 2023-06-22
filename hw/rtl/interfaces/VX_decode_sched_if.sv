`include "VX_define.vh"

interface VX_decode_sched_if ();

    wire                    valid;
    wire                    is_wstall;
    wire                    is_join;
    wire [`UP(`NW_BITS)-1:0] wid;

    modport master (
        output valid,
        output is_wstall,
        output is_join,
        output wid
    );

    modport slave (
        input valid,
        input is_wstall,
        input is_join,
        input wid
    );

endinterface
