`include "VX_define.vh"

interface VX_join_if ();

    wire                    valid;
    wire [`UP(`NW_BITS)-1:0] wid;

    modport master (
        output valid,    
        output wid
    );

    modport slave (
        input valid,   
        input wid
    );

endinterface
