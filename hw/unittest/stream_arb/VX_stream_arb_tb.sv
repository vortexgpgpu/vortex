// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_platform.vh"

// Self-checking harness around one VX_stream_arb: every input sends numbered
// packets under random valid and output backpressure, and every packet must
// leave once, on an output its input maps to, with the select of its row.
module VX_stream_arb_tb #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter MAX_FANOUT  = 8,
    parameter SEED        = 1
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        drain,
    output wire        idle,
    output wire [31:0] delivered,
    output wire [31:0] errors
);
    localparam SRC_W     = 8;
    localparam SEQ_W     = 16;
    localparam DATAW     = SRC_W + SEQ_W;
    localparam NUM_REQS  = (NUM_INPUTS > NUM_OUTPUTS) ? `CDIV(NUM_INPUTS, NUM_OUTPUTS) : `CDIV(NUM_OUTPUTS, NUM_INPUTS);
    localparam SEL_COUNT = `MIN(NUM_INPUTS, NUM_OUTPUTS);
    localparam SEL_W     = `UP(`CLOG2(NUM_REQS));

    `STATIC_ASSERT (NUM_INPUTS < (1 << SRC_W), ("invalid parameter: NUM_INPUTS=%0d", NUM_INPUTS))

    function automatic logic [31:0] hash32(input logic [31:0] x);
        logic [31:0] h;
        h = x ^ (x >> 16);
        h = h * 32'h7feb352d;
        h = h ^ (h >> 15);
        h = h * 32'h846ca68b;
        return h ^ (h >> 16);
    endfunction

    reg  [31:0] cycle;

    reg  [NUM_INPUTS-1:0]             valid_in;
    reg  [NUM_INPUTS-1:0][SEQ_W-1:0]  seq_in;
    wire [NUM_INPUTS-1:0][DATAW-1:0]  data_in;
    wire [NUM_INPUTS-1:0]             ready_in;

    wire [NUM_OUTPUTS-1:0]            valid_out;
    wire [NUM_OUTPUTS-1:0][DATAW-1:0] data_out;
    wire [NUM_OUTPUTS-1:0]            ready_out;
    wire [SEL_COUNT-1:0][SEL_W-1:0]   sel_out;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_data_in
        assign data_in[i] = {SRC_W'(i), seq_in[i]};
    end

    // outputs stall at random, with a longer stall now and then so the
    // arbiter buffers fill up
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_ready_out
        wire [1:0] rnd = 2'(hash32(cycle * 32'd977 + 32'd7919 * 32'(o + 1024) + 32'(SEED)));
        assign ready_out[o] = (rnd != 0) && (cycle[9:6] != 4'(SEED));
    end

    VX_stream_arb #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .DATAW       (DATAW),
        .ARBITER     ("R"),
        .MAX_FANOUT  (MAX_FANOUT),
        .OUT_BUF     (3)
    ) dut (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_in),
        .data_in   (data_in),
        .ready_in  (ready_in),
        .valid_out (valid_out),
        .data_out  (data_out),
        .ready_out (ready_out),
        .sel_out   (sel_out)
    );

    wire [NUM_OUTPUTS-1:0][SRC_W-1:0] src_out;
    wire [NUM_OUTPUTS-1:0][SEQ_W-1:0] seq_out;

    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_data_out
        assign {src_out[o], seq_out[o]} = data_out[o];
    end

    reg  [NUM_INPUTS-1:0][31:0] sent_cnt, recv_cnt;
    reg  [NUM_INPUTS-1:0][31:0] sent_sum, recv_sum;
    reg  [31:0] errors_r;

    logic [NUM_INPUTS-1:0][31:0] recv_inc, recv_sum_inc;
    logic [31:0] errors_inc;

    always_comb begin
        recv_inc     = '0;
        recv_sum_inc = '0;
        errors_inc   = '0;
        for (integer o = 0; o < NUM_OUTPUTS; ++o) begin
            if (valid_out[o] && ready_out[o]) begin
                if (32'(src_out[o]) >= NUM_INPUTS) begin
                    errors_inc = errors_inc + 1;
                end else begin
                    recv_inc[src_out[o]]     = recv_inc[src_out[o]] + 1;
                    recv_sum_inc[src_out[o]] = recv_sum_inc[src_out[o]] + 32'(seq_out[o]);
                    if (NUM_INPUTS >= NUM_OUTPUTS) begin
                        // an input reaches one output, in order, with its row as select
                        if ((32'(src_out[o]) % NUM_OUTPUTS) != o
                         || seq_out[o] != recv_cnt[src_out[o]][SEQ_W-1:0]
                         || 32'(sel_out[o % SEL_COUNT]) != (32'(src_out[o]) / NUM_OUTPUTS)) begin
                            errors_inc = errors_inc + 1;
                        end
                    end else begin
                        // an input reaches any output of its column
                        if ((o % NUM_INPUTS) != 32'(src_out[o])) begin
                            errors_inc = errors_inc + 1;
                        end
                    end
                end
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            cycle    <= '0;
            valid_in <= '0;
            seq_in   <= '0;
            sent_cnt <= '0;
            sent_sum <= '0;
            recv_cnt <= '0;
            recv_sum <= '0;
            errors_r <= '0;
        end else begin
            cycle <= cycle + 1;
            for (integer i = 0; i < NUM_INPUTS; ++i) begin
                logic [2:0] rnd;
                rnd = 3'(hash32(cycle * 32'd977 + 32'd7919 * 32'(i) + 32'(SEED)));
                if (valid_in[i] && ready_in[i]) begin
                    sent_cnt[i] <= sent_cnt[i] + 1;
                    sent_sum[i] <= sent_sum[i] + 32'(seq_in[i]);
                    seq_in[i]   <= seq_in[i] + 1;
                    valid_in[i] <= ~drain && (rnd != 0);
                end else if (~valid_in[i]) begin
                    valid_in[i] <= ~drain && (rnd != 0);
                end
            end
            for (integer i = 0; i < NUM_INPUTS; ++i) begin
                recv_cnt[i] <= recv_cnt[i] + recv_inc[i];
                recv_sum[i] <= recv_sum[i] + recv_sum_inc[i];
            end
            errors_r <= errors_r + errors_inc;
            if (errors_inc != 0) begin
                $display("cycle %0d: stream_arb %0dx%0d (fanout %0d): misrouted packet", cycle, NUM_INPUTS, NUM_OUTPUTS, MAX_FANOUT);
            end
        end
    end

    logic [31:0] delivered_w;
    logic idle_w;

    always_comb begin
        delivered_w = '0;
        idle_w = (valid_in == '0);
        for (integer i = 0; i < NUM_INPUTS; ++i) begin
            delivered_w = delivered_w + recv_cnt[i];
            // every input must have been served, and nothing lost
            idle_w = idle_w && (recv_cnt[i] != 0)
                            && (recv_cnt[i] == sent_cnt[i])
                            && (recv_sum[i] == sent_sum[i]);
        end
    end

    assign idle      = idle_w;
    assign delivered = delivered_w;
    assign errors    = errors_r;

endmodule
