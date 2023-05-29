`include "VX_tex_define.vh"

interface VX_tex_req_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                            valid;
    wire [NUM_LANES-1:0]            mask;
    wire [1:0][NUM_LANES-1:0][31:0] coords;
    wire [NUM_LANES-1:0][`TEX_LOD_BITS-1:0] lod;
    wire [`TEX_STAGE_BITS-1:0]      stage;
    wire [TAG_WIDTH-1:0]            tag;    
    wire                            ready;

    modport master (
        output valid,
        output mask,
        output coords,
        output lod,
        output stage,
        output tag,
        input  ready
    );

    modport slave (
        input  valid,
        input  mask,
        input  coords,
        input  lod,
        input  stage,
        input  tag,
        output ready
    );

endinterface
