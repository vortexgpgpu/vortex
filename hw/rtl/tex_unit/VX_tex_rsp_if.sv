`include "VX_tex_define.vh"

interface VX_tex_rsp_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                        valid;
    wire [NUM_LANES-1:0][31:0]  texels;
    wire [TAG_WIDTH-1:0]        tag; 
    wire                        ready;

    modport master (
        output valid,
        output texels,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  texels,
        input  tag,
        output ready
    );

endinterface
