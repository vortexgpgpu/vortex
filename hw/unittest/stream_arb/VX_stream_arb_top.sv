// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_platform.vh"

// VX_stream_arb shapes under test, each in its own self-checking harness.
// Over 1.5x MAX_FANOUT requests an arbiter becomes a tree of slices; the
// shapes cover full slices, a last slice of one input or output, one smaller
// than the other side, one not a whole number of rows, and a join that is a
// tree itself.
module VX_stream_arb_top #(
    parameter NUM_CFGS = 21
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        drain,
    output wire        idle,
    output wire [NUM_CFGS-1:0] busy,
    output wire [31:0] delivered,
    output wire [31:0] errors
);
    //                                            input select                                    | output select              | flat
    localparam int CFG_NI [NUM_CFGS] = '{64, 17, 65, 13, 19, 34, 36, 40, 26, 27, 129, 200,          1,  4,  2,  1,  4,         5, 8, 3, 1};
    localparam int CFG_NO [NUM_CFGS] = '{ 1,  1,  1,  1,  1,  4,  4,  4,  2,  2,   1,   1,         17, 34, 27, 64, 32,         2, 8, 5, 1};
    localparam int CFG_MF [NUM_CFGS] = '{ 8,  8,  8,  8,  8,  4,  4,  4,  4,  4,   8,   4,          8,  4,  4,  8,  8,         8, 8, 8, 8};

    wire [NUM_CFGS-1:0]       idle_w;
    wire [NUM_CFGS-1:0][31:0] delivered_w;
    wire [NUM_CFGS-1:0][31:0] errors_w;

    for (genvar c = 0; c < NUM_CFGS; ++c) begin : g_cfgs
        VX_stream_arb_tb #(
            .NUM_INPUTS  (CFG_NI[c]),
            .NUM_OUTPUTS (CFG_NO[c]),
            .MAX_FANOUT  (CFG_MF[c]),
            .SEED        (c + 1)
        ) tb (
            .clk       (clk),
            .reset     (reset),
            .drain     (drain),
            .idle      (idle_w[c]),
            .delivered (delivered_w[c]),
            .errors    (errors_w[c])
        );
    end

    logic [31:0] delivered_s, errors_s;

    always_comb begin
        delivered_s = '0;
        errors_s    = '0;
        for (integer c = 0; c < NUM_CFGS; ++c) begin
            delivered_s = delivered_s + delivered_w[c];
            errors_s    = errors_s + errors_w[c];
        end
    end

    assign idle      = (& idle_w);
    assign busy      = ~idle_w;
    assign delivered = delivered_s;
    assign errors    = errors_s;

endmodule
