// Module to evaluate the edge function

`include "VX_raster_define.vh"

module VX_raster_edge #(
    parameter LATENCY = 3
) (
    input wire clk,
    input wire reset,

    input wire                        enable,
    input wire [`RASTER_DIM_BITS-1:0] xloc,
    input wire [`RASTER_DIM_BITS-1:0] yloc,
    input wire [2:0][2:0][`RASTER_DATA_BITS-1:0] edges,

    output wire [2:0][`RASTER_DATA_BITS-1:0] result
);
    localparam PROD_WIDTH = `RASTER_DATA_BITS + 1;

    `UNUSED_VAR (reset)

    `STATIC_ASSERT((LATENCY >= `LATENCY_IMUL), ("invalid parameter"))

    wire [2:0][PROD_WIDTH-1:0] prod_x;
    wire [2:0][PROD_WIDTH-1:0] prod_y;
    wire [2:0][`RASTER_DATA_BITS-1:0] edge_c, edge_c_s;
    
    wire [2:0][`RASTER_DATA_BITS-1:0] result_s;

    for (genvar i = 0; i < 3; ++i) begin
        VX_multiplier #(
            .A_WIDTH (`RASTER_DATA_BITS),
            .B_WIDTH (`RASTER_DIM_BITS),
            .R_WIDTH (PROD_WIDTH),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL)
        ) x_multiplier (
            .clk    (clk),
            .enable (enable),
            .dataa  (edges[i][0]),
            .datab  (xloc),
            .result (prod_x[i])
        );

        VX_multiplier #(
            .A_WIDTH (`RASTER_DATA_BITS),
            .B_WIDTH (`RASTER_DIM_BITS),
            .R_WIDTH (PROD_WIDTH),
            .SIGNED  (1),
            .LATENCY (`LATENCY_IMUL)
        ) y_multiplier (
            .clk    (clk),
            .enable (enable),
            .dataa  (edges[i][1]),
            .datab  (yloc),
            .result (prod_y[i])
        );

        assign edge_c[i] = edges[i][2];
    end

    VX_shift_register #(
        .DATAW (3 * `RASTER_DATA_BITS),
        .DEPTH (LATENCY)
    ) shift_reg1 (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  ({edge_c}),
        .data_out ({edge_c_s})
    );

    for (genvar i = 0; i < 3; ++i) begin
        wire [PROD_WIDTH-1:0] sum = prod_x[i] + prod_y[i] + PROD_WIDTH'(edge_c_s[i]);
        `UNUSED_VAR (sum)
        assign result_s[i] = sum[`RASTER_DATA_BITS-1:0];
    end

    VX_shift_register #(
        .DATAW (3 * `RASTER_DATA_BITS),
        .DEPTH (LATENCY - `LATENCY_IMUL)
    ) shift_reg2 (
        .clk      (clk),
        `UNUSED_PIN (reset),
        .enable   (enable),
        .data_in  (result_s),
        .data_out (result)
    );

endmodule
