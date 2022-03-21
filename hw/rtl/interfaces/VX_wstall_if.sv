`ifndef VX_WSTALL_IF
`define VX_WSTALL_IF

`include "VX_define.vh"

interface VX_wstall_if();

    wire                valid;    
    wire [`NW_BITS-1:0] wid;
    wire                stalled;

    modport master (
        output valid,
        output wid,
        output stalled
    );

    modport slave (
        input valid,
        input wid,
        input stalled
    );

endinterface

`endif
