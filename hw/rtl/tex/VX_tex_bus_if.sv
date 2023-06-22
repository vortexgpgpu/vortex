`include "VX_tex_define.vh"

interface VX_tex_bus_if #(
    parameter NUM_LANES = 1,
    parameter TAG_WIDTH = 1
) ();

    wire                            req_valid;
    wire [NUM_LANES-1:0]            req_mask;
    wire [1:0][NUM_LANES-1:0][31:0] req_coords;
    wire [NUM_LANES-1:0][`TEX_LOD_BITS-1:0] req_lod;
    wire [`TEX_STAGE_BITS-1:0]      req_stage;
    wire [TAG_WIDTH-1:0]            req_tag;    
    wire                            req_ready;

    wire                            rsp_valid;
    wire [NUM_LANES-1:0][31:0]      rsp_texels;
    wire [TAG_WIDTH-1:0]            rsp_tag; 
    wire                            rsp_ready;

    modport master (
        output req_valid,
        output req_mask,
        output req_coords,
        output req_lod,
        output req_stage,
        output req_tag,
        input  req_ready,

        input  rsp_valid,
        input  rsp_texels,
        input  rsp_tag,
        output rsp_ready
    );

    modport slave (
        input  req_valid,
        input  req_mask,
        input  req_coords,
        input  req_lod,
        input  req_stage,
        input  req_tag,
        output req_ready,

        output rsp_valid,
        output rsp_texels,
        output rsp_tag,
        input  rsp_ready
    );

endinterface
