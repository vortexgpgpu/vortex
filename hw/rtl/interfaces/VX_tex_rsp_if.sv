`ifndef VX_TEX_RSP_IF
`define VX_TEX_RSP_IF

`include "VX_define.vh"

interface VX_tex_rsp_if ();

    wire                    valid;   
    wire [`NW_BITS-1:0]     wid;
    wire [`NUM_THREADS-1:0] tmask;    
    wire [31:0]             PC;    
    wire [`NR_BITS-1:0]     rd;
    wire                    wb;
    wire [`NUM_THREADS-1:0][31:0] data;
    wire                    ready;

    modport master (
        output valid,
        output wid,
        output tmask,
        output PC,
        output rd,
        output wb,
        output data,
        input  ready
    );

    modport slave (
        input  valid,
        input  wid,
        input  tmask,
        input  PC,
        input  rd,
        input  wb,
        input  data,
        output ready
    );

endinterface

`endif
 
 
