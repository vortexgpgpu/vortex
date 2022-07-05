`include "VX_define.vh"

interface VX_dcr_write_if ();

    wire                          valid;
    wire [`VX_DCR_ADDR_WIDTH-1:0] addr;
    wire [`VX_DCR_DATA_WIDTH-1:0] data;

    modport master (
        output valid,
        output addr,
        output data
    );

    modport slave (
        input  valid,
        input  addr,
        input  data
    );

endinterface
