// Module to evaluate the edge function

`include "VX_raster_define.vh"

module VX_raster_edge_function #(
    parameter MUL_LATENCY = 3
) (
    input wire clk,
    input wire reset,

    input wire         [`RASTER_DIM_BITS-1:0]            x_loc,
    input wire         [`RASTER_DIM_BITS-1:0]            y_loc,
    input wire signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0] edges[2:0][2:0],

    output wire signed [`RASTER_PRIMITIVE_DATA_BITS-1:0] result[2:0]
);
    `UNUSED_VAR (reset)

    for (genvar i = 0; i < 3; ++i) begin
        wire signed [2*`RASTER_PRIMITIVE_DATA_BITS-1:0] prod_x;
        wire signed [2*`RASTER_PRIMITIVE_DATA_BITS-1:0] prod_y;

        VX_multiplier #(
            .WIDTHA  (`RASTER_PRIMITIVE_DATA_BITS),
            .WIDTHB  (`RASTER_DIM_BITS),
            .WIDTHP  (2 * `RASTER_PRIMITIVE_DATA_BITS),
            .SIGNED  (1),
            .LATENCY (MUL_LATENCY)
        ) x_multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (edges[i][0]),
            .datab  (x_loc),
            .result (prod_x)
        );

        VX_multiplier #(
            .WIDTHA  (`RASTER_PRIMITIVE_DATA_BITS),
            .WIDTHB  (`RASTER_DIM_BITS),
            .WIDTHP  (2 * `RASTER_PRIMITIVE_DATA_BITS),
            .SIGNED  (1),
            .LATENCY (MUL_LATENCY)
        ) y_multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (edges[i][1]),
            .datab  (y_loc),
            .result (prod_y)
        );

        `UNUSED_VAR (prod_x)
        `UNUSED_VAR (prod_y)

        assign result[i] = prod_x[`RASTER_PRIMITIVE_DATA_BITS-1:0] 
                         + prod_y[`RASTER_PRIMITIVE_DATA_BITS-1:0] 
                         + edges[i][2];
    end

endmodule
