`ifndef VX_MSU_REQ_IF
`define VX_MSU_REQ_IF

`include "VX_define.vh"

interface VX_msu_req_if #(
    parameter NUM_REQS = 4,
    parameter WORD_SIZE = 4,
    parameter TAG_WIDTH = 32
)();

    wire                        valid;
    wire                        rw;
    wire [NUM_REQS-1:0]         mask;
    wire [WORD_SIZE-1:0]        byteen;
    wire [NUM_REQS-1:0][31:0]   addr;
    wire [NUM_REQS-1:0][31:0]   data;
    wire [TAG_WIDTH-1:0]        tag;
    wire                        ready;

    modport master (
        output valid,
        output rw,
        output mask,
        output byteen,
        output addr,
        output data,
        output tag,
        input ready
    );

    modport slave (
        input valid,
        input rw,
        input mask,
        input byteen,
        input addr,
        input data,
        input tag,
        output ready
    );

endinterface

`endif