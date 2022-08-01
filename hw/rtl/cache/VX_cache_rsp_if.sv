`include "VX_cache_define.vh"

interface VX_cache_rsp_if #(
    parameter NUM_REQS  = 1,
    parameter WORD_SIZE = 1,
    parameter TAG_WIDTH = 1
) ();

    wire [NUM_REQS-1:0]                 valid;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0] data;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]  tag;
    wire [NUM_REQS-1:0]                 ready;

    modport master (
        output valid,
        output data,        
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  data,
        input  tag,
        output ready
    );

endinterface
