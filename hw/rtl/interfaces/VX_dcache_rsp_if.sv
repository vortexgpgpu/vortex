`ifndef VX_DCACHE_RSP_IF
`define VX_DCACHE_RSP_IF

`include "../cache/VX_cache_define.vh"

interface VX_dcache_rsp_if #(
    parameter NUM_REQS  = 1,
    parameter WORD_SIZE = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                                    valid;
    wire [NUM_REQS-1:0]                     tmask;
    wire [NUM_REQS-1:0][`WORD_WIDTH-1:0]    data;
    wire [TAG_WIDTH-1:0]                    tag;
    wire                                    ready;

    modport master (
        output valid,
        output tmask,
        output data,        
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  tmask,
        input  data,
        input  tag,
        output ready
    );

endinterface

`endif
