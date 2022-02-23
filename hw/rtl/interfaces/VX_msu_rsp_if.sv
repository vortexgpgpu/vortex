`ifndef VX_MSU_RSP_IF
`define VX_MSU_RSP_IF

`include "VX_define.vh"

interface VX_msu_req_if #(
    parameter NUM_REQS = 4,
    parameter WORD_SIZE = 4,
    parameter TAG_WIDTH = 32
)();

    wire                        valid;
    wire [NUM_REQS-1:0]         mask;
    wire [NUM_REQS-1:0][31:0]   data;
    wire [TAG_WIDTH-1:0]        tag;
    wire                        ready;

    modport master (
        output valid,
        output mask,
        output data,
        output tag,
        input ready
    );

    modport slave (
        input valid,
        input mask,
        input data,
        input tag,
        output ready
    );

endinterface

`endif