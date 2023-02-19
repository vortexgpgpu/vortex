`include "VX_define.vh"

interface VX_branch_ctl_if ();

    wire                    valid;    
    wire [`UP(`NW_BITS)-1:0] wid;    
    wire                    taken;
    wire [`XLEN-1:0]        dest;

    modport master (
        output valid,    
        output wid,
        output taken,
        output dest
    );

    modport slave (
        input valid,   
        input wid,
        input taken,
        input dest
    );

endinterface
