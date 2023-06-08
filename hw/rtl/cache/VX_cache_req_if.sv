`include "VX_cache_define.vh"

interface VX_cache_req_if #(
    parameter NUM_REQS   = 1,
    parameter WORD_SIZE  = 1,
    parameter TAG_WIDTH  = 1,    
    parameter ADDR_WIDTH = `XLEN - `CLOG2(WORD_SIZE),
    parameter DATA_WIDTH = WORD_SIZE * 8
) ();

    wire [NUM_REQS-1:0]                 valid;
    wire [NUM_REQS-1:0]                 rw;
    wire [NUM_REQS-1:0][WORD_SIZE-1:0]  byteen;
    wire [NUM_REQS-1:0][ADDR_WIDTH-1:0] addr;
    wire [NUM_REQS-1:0][DATA_WIDTH-1:0] data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  tag;
    wire [NUM_REQS-1:0]                 ready;

    modport master (
        output valid,
        output rw,
        output byteen,
        output addr,
        output data,        
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  rw,
        input  byteen,
        input  addr,
        input  data,
        input  tag,
        output ready
    );

endinterface
