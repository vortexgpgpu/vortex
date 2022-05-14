// Module to evaluate the edge function

`include "VX_raster_define.vh"

module VX_raster_edge_functions #(
    parameter MUL_LATENCY = 3
) (
    input logic clk,
    input logic         [`RASTER_DIM_BITS-1:0]              x_loc, y_loc,
    input logic signed  [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edges[2:0][2:0],
    output logic signed [`RASTER_PRIMITIVE_DATA_BITS-1:0]   edge_func_val[2:0]
);
    for (genvar i = 0; i < 3; ++i) begin
        logic signed [2*`RASTER_PRIMITIVE_DATA_BITS-1:0] mul_val1, mul_val2;
        VX_multiplier #(
            .WIDTHA  (`RASTER_PRIMITIVE_DATA_BITS),
            .WIDTHB  (`RASTER_DIM_BITS),
            .WIDTHP  (2*`RASTER_PRIMITIVE_DATA_BITS),
            .SIGNED  (1),
            .LATENCY (MUL_LATENCY)
        ) x_multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (edges[i][0]),
            .datab  (x_loc),
            .result (mul_val1)
        );

        VX_multiplier #(
            .WIDTHA  (`RASTER_PRIMITIVE_DATA_BITS),
            .WIDTHB  (`RASTER_DIM_BITS),
            .WIDTHP  (2*`RASTER_PRIMITIVE_DATA_BITS),
            .SIGNED  (1),
            .LATENCY (MUL_LATENCY)
        ) y_multiplier (
            .clk    (clk),
            .enable (1'b1),
            .dataa  (edges[i][1]),
            .datab  (y_loc),
            .result (mul_val2)
        );

        `UNUSED_VAR (mul_val1)
        `UNUSED_VAR (mul_val2)

        always_comb begin
            edge_func_val[i] = mul_val1[`RASTER_PRIMITIVE_DATA_BITS-1:0] + mul_val2[`RASTER_PRIMITIVE_DATA_BITS-1:0] + edges[i][2];
        end
    end

endmodule
