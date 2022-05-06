`include "VX_raster_define.vh"

module VX_raster_rsp_switch #(  
    parameter CLUSTER_ID = 0,
    parameter RASTER_QUAD_OUTPUT_RATE = 4
) (
    input valid,
    input empty,
    // Quad data
    input        [`RASTER_DIM_BITS-1:0]             x_loc   [RASTER_QUAD_OUTPUT_RATE-1:0],
    input        [`RASTER_DIM_BITS-1:0]             y_loc   [RASTER_QUAD_OUTPUT_RATE-1:0],
    input        [3:0]                              masks   [RASTER_QUAD_OUTPUT_RATE-1:0],
    input signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]  bcoords [RASTER_QUAD_OUTPUT_RATE-1:0][2:0][3:0],
    input        [`RASTER_PRIMITIVE_DATA_BITS-1:0]  pid     [RASTER_QUAD_OUTPUT_RATE-1:0],

    // Output
    VX_raster_req_if.master  raster_req_if
);

    raster_stamp_t [RASTER_QUAD_OUTPUT_RATE-1:0]   stamps;
    for (genvar i = 0; i < RASTER_QUAD_OUTPUT_RATE; ++i) begin
        always_comb begin
            stamps[i].pos_x    = x_loc[i][`RASTER_DIM_BITS-2:0];
            stamps[i].pos_y    = y_loc[i][`RASTER_DIM_BITS-2:0];
            stamps[i].mask     = masks[i];
            stamps[i].pid      = pid[i][`RASTER_PID_BITS-1:0];
        end
    end

    // Assign for bcoords array type transformation
    for (genvar i = 0; i < RASTER_QUAD_OUTPUT_RATE; ++i) begin
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
