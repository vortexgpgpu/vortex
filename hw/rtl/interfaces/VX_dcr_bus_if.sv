`include "VX_define.vh"

interface VX_dcr_bus_if ();

    wire                          write_valid;
    wire [`VX_DCR_ADDR_WIDTH-1:0] write_addr;
    wire [`VX_DCR_DATA_WIDTH-1:0] write_data;

    modport master (
        output write_valid,
        output write_addr,
        output write_data
    );

    modport slave (
        input  write_valid,
        input  write_addr,
        input  write_data
    );

endinterface
