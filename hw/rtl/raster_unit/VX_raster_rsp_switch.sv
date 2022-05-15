`include "VX_raster_define.vh"

module VX_raster_rsp_switch #(  
    parameter CLUSTER_ID = 0,
    parameter OUTPUT_QUADS = 4
) (
    input wire valid,
    input wire empty,
    // Quad data
    input wire        [`RASTER_DIM_BITS-1:0]             x_loc   [OUTPUT_QUADS-1:0],
    input wire        [`RASTER_DIM_BITS-1:0]             y_loc   [OUTPUT_QUADS-1:0],
    input wire        [3:0]                              masks   [OUTPUT_QUADS-1:0],
    input wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  bcoords [OUTPUT_QUADS-1:0][2:0][3:0],
    input wire        [`RASTER_PRIMITIVE_DATA_BITS-1:0]  pid     [OUTPUT_QUADS-1:0],

    // Output
    VX_raster_req_if.master  raster_req_if
);

    raster_stamp_t [OUTPUT_QUADS-1:0]   stamps;
    for (genvar i = 0; i < OUTPUT_QUADS; ++i) begin
        always_comb begin
            stamps[i].pos_x    = x_loc[i][`RASTER_DIM_BITS-2:0];
            stamps[i].pos_y    = y_loc[i][`RASTER_DIM_BITS-2:0];
            stamps[i].mask     = masks[i];
            stamps[i].pid      = pid[i][`RASTER_PID_BITS-1:0];
        end
    end

    // Assign for bcoords array type transformation
    for (genvar i = 0; i < OUTPUT_QUADS; ++i) begin
        for (genvar j = 0; j < 4; ++j) begin
            assign stamps[i].bcoord_x[j] = bcoords[i][0][j];
            assign stamps[i].bcoord_y[j] = bcoords[i][1][j];
            assign stamps[i].bcoord_z[j] = bcoords[i][2][j];
        end
    end

    assign raster_req_if.empty  = empty;
    assign raster_req_if.stamps = stamps;
    assign raster_req_if.valid  = valid;

endmodule
