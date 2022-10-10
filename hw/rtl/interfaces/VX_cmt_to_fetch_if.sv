`include "VX_define.vh"

interface VX_cmt_to_fetch_if ();

    wire valid;
    wire [`EX_UNITS_BITS-1:0] committed;

    modport master (
        output valid,
        output committed
    );

    modport slave (
        input valid,
        input committed
    );

endinterface
