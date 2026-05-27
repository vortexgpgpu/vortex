// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`TRACING_OFF
module VX_avs_adapter #(
    parameter DATA_WIDTH    = 1,
    parameter ADDR_WIDTH_IN = 1,
    parameter ADDR_WIDTH_OUT= 32,
    parameter BURST_WIDTH   = 1,
    parameter NUM_PORTS_IN  = 1,
    parameter NUM_BANKS_OUT = 1,
    parameter TAG_WIDTH     = 1,
    parameter RD_QUEUE_SIZE = 1,
    parameter INTERLEAVE    = 0,
    parameter ARBITER       = "R",
    parameter REQ_OUT_BUF   = 0,
    parameter RSP_OUT_BUF   = 0
) (
    input  wire                     clk,
    input  wire                     reset,

    // Memory request
    input  wire                     mem_req_valid [NUM_PORTS_IN],
    input  wire                     mem_req_rw [NUM_PORTS_IN],
    input  wire [DATA_WIDTH/8-1:0]  mem_req_byteen [NUM_PORTS_IN],
    input  wire [ADDR_WIDTH_IN-1:0] mem_req_addr [NUM_PORTS_IN],
    input  wire [DATA_WIDTH-1:0]    mem_req_data [NUM_PORTS_IN],
    input  wire [TAG_WIDTH-1:0]     mem_req_tag [NUM_PORTS_IN],
    output wire                     mem_req_ready [NUM_PORTS_IN],

    // Memory response
    output wire                     mem_rsp_valid [NUM_PORTS_IN],
    output wire [DATA_WIDTH-1:0]    mem_rsp_data [NUM_PORTS_IN],
    output wire [TAG_WIDTH-1:0]     mem_rsp_tag [NUM_PORTS_IN],
    input  wire                     mem_rsp_ready [NUM_PORTS_IN],

    // AVS bus
    output wire [DATA_WIDTH-1:0]    avs_writedata [NUM_BANKS_OUT],
    input  wire [DATA_WIDTH-1:0]    avs_readdata [NUM_BANKS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] avs_address [NUM_BANKS_OUT],
    input  wire                     avs_waitrequest [NUM_BANKS_OUT],
    output wire                     avs_write [NUM_BANKS_OUT],
    output wire                     avs_read [NUM_BANKS_OUT],
    output wire [DATA_WIDTH/8-1:0]  avs_byteenable [NUM_BANKS_OUT],
    output wire [BURST_WIDTH-1:0]   avs_burstcount [NUM_BANKS_OUT],
    input  wire                     avs_readdatavalid [NUM_BANKS_OUT]
);
    localparam DATA_SIZE      = DATA_WIDTH/8;
    localparam BANK_SEL_BITS  = `CLOG2(NUM_BANKS_OUT);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam DST_ADDR_WDITH = ADDR_WIDTH_OUT + BANK_SEL_BITS; // convert output addresss to input space
    localparam BANK_ADDR_WIDTH = DST_ADDR_WDITH - BANK_SEL_BITS;
    localparam NUM_PORTS_IN_BITS = `CLOG2(NUM_PORTS_IN);
    localparam NUM_PORTS_IN_WIDTH = `UP(NUM_PORTS_IN_BITS);
    localparam REQ_QUEUE_DATAW = TAG_WIDTH + NUM_PORTS_IN_BITS;
    localparam REQ_XBAR_DATAW = 1 + BANK_ADDR_WIDTH + DATA_WIDTH + DATA_SIZE + TAG_WIDTH;
    localparam RSP_XBAR_DATAW = DATA_WIDTH + TAG_WIDTH;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))

    // Bank selection

    wire [NUM_PORTS_IN-1:0][BANK_SEL_WIDTH-1:0] req_bank_sel;
    wire [NUM_PORTS_IN-1:0][BANK_ADDR_WIDTH-1:0] req_bank_addr;

    if (NUM_BANKS_OUT > 1) begin : g_bank_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            wire [DST_ADDR_WDITH-1:0] mem_req_addr_dst = DST_ADDR_WDITH'(mem_req_addr[i]);
            if (INTERLEAVE) begin : g_interleave
                assign req_bank_sel[i]  = mem_req_addr_dst[BANK_SEL_BITS-1:0];
                assign req_bank_addr[i] = mem_req_addr_dst[BANK_SEL_BITS +: BANK_ADDR_WIDTH];
            end else begin : g_no_interleave
                assign req_bank_sel[i]  = mem_req_addr_dst[BANK_ADDR_WIDTH +: BANK_SEL_BITS];
                assign req_bank_addr[i] = mem_req_addr_dst[BANK_ADDR_WIDTH-1:0];
            end
        end
    end else begin : g_no_bank_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            assign req_bank_sel[i]  = '0;
            assign req_bank_addr[i] = DST_ADDR_WDITH'(mem_req_addr[i]);
        end
    end

    // Requests handling

    wire [NUM_PORTS_IN-1:0] req_xbar_valid_in;
    wire [NUM_PORTS_IN-1:0][REQ_XBAR_DATAW-1:0] req_xbar_data_in;
    wire [NUM_PORTS_IN-1:0] req_xbar_ready_in;

    wire [NUM_BANKS_OUT-1:0] req_xbar_valid_out;
    wire [NUM_BANKS_OUT-1:0][REQ_XBAR_DATAW-1:0] req_xbar_data_out;
    wire [NUM_BANKS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] req_xbar_sel_out;
    wire [NUM_BANKS_OUT-1:0] req_xbar_ready_out;

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_req_xbar_data_in
        assign req_xbar_valid_in[i] = mem_req_valid[i];
        assign req_xbar_data_in[i]  = {mem_req_rw[i], req_bank_addr[i], mem_req_byteen[i], mem_req_data[i], mem_req_tag[i]};
        assign mem_req_ready[i] = req_xbar_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS (NUM_PORTS_IN),
        .NUM_OUTPUTS(NUM_BANKS_OUT),
        .DATAW      (REQ_XBAR_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (REQ_OUT_BUF)
    ) req_xbar (
        .clk       (clk),
        .reset     (reset),
        .sel_in    (req_bank_sel),
        .valid_in  (req_xbar_valid_in),
        .data_in   (req_xbar_data_in),
        .ready_in  (req_xbar_ready_in),
        .valid_out (req_xbar_valid_out),
        .data_out  (req_xbar_data_out),
        .ready_out (req_xbar_ready_out),
        .sel_out   (req_xbar_sel_out),
        `UNUSED_PIN (collisions)
    );

    wire [NUM_BANKS_OUT-1:0][REQ_QUEUE_DATAW-1:0] rd_req_queue_data_out;
    wire [NUM_BANKS_OUT-1:0] rd_req_queue_pop;

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_req_xbar_data_out

        wire ready_out;
        wire rw_out;
        wire [BANK_ADDR_WIDTH-1:0] addr_out;
        wire [TAG_WIDTH-1:0] tag_out;
        wire [DATA_WIDTH-1:0] data_out;
        wire [DATA_SIZE-1:0] byteen_out;
        wire valid_out;

        assign {rw_out, addr_out, byteen_out, data_out, tag_out} = req_xbar_data_out[i];

        wire rd_req_queue_going_full;
        wire rd_req_queue_push;

        // stall pipeline if the request queue is needed and going full
        wire rd_req_queue_ready = rw_out || ~rd_req_queue_going_full;
        assign valid_out = req_xbar_valid_out[i] && rd_req_queue_ready;
        assign ready_out = ~avs_waitrequest[i] && rd_req_queue_ready;
        assign rd_req_queue_push = valid_out && ready_out && ~rw_out;

        VX_pending_size #(
            .SIZE (RD_QUEUE_SIZE)
        ) pending_size (
            .clk   (clk),
            .reset (reset),
            .incr  (rd_req_queue_push),
            .decr  (rd_req_queue_pop[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full  (rd_req_queue_going_full),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        wire [REQ_QUEUE_DATAW-1:0] rd_req_queue_data_in;
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rd_req_queue_data_in = {tag_out, req_xbar_sel_out[i]};
        end else begin : g_no_input_sel
            `UNUSED_VAR (req_xbar_sel_out[i])
            assign rd_req_queue_data_in = tag_out;
        end

        VX_fifo_queue #(
            .DATAW (REQ_QUEUE_DATAW),
            .DEPTH (RD_QUEUE_SIZE)
        ) rd_req_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (rd_req_queue_push),
            .pop      (rd_req_queue_pop[i]),
            .data_in  (rd_req_queue_data_in),
            .data_out (rd_req_queue_data_out[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign avs_read[i]       = valid_out && ~rw_out;
        assign avs_write[i]      = valid_out && rw_out;
        assign avs_address[i]    = ADDR_WIDTH_OUT'(addr_out);
        assign avs_byteenable[i] = byteen_out;
        assign avs_writedata[i]  = data_out;
        assign avs_burstcount[i] = BURST_WIDTH'(1);
        assign req_xbar_ready_out[i] = ready_out;
    end

    // Responses handling

    wire [NUM_BANKS_OUT-1:0] rsp_xbar_valid_in;
    wire [NUM_BANKS_OUT-1:0][RSP_XBAR_DATAW-1:0] rsp_xbar_data_in;
    wire [NUM_BANKS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] rsp_xbar_sel_in;
    wire [NUM_BANKS_OUT-1:0] rsp_xbar_ready_in;

    wire [NUM_PORTS_IN-1:0] rsp_xbar_valid_out;
    wire [NUM_PORTS_IN-1:0][RSP_XBAR_DATAW-1:0] rsp_xbar_data_out;
    wire [NUM_PORTS_IN-1:0] rsp_xbar_ready_out;

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_rsp_xbar_data_in

        wire [DATA_WIDTH-1:0] rsp_queue_data_out;
        wire rsp_queue_empty;

        VX_fifo_queue #(
            .DATAW (DATA_WIDTH),
            .DEPTH (RD_QUEUE_SIZE)
        ) rsp_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (avs_readdatavalid[i]),
            .pop      (rd_req_queue_pop[i]),
            .data_in  (avs_readdata[i]),
            .data_out (rsp_queue_data_out),
            .empty    (rsp_queue_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign rsp_xbar_valid_in[i] = ~rsp_queue_empty;
        assign rsp_xbar_data_in[i] = {rsp_queue_data_out, rd_req_queue_data_out[i][NUM_PORTS_IN_BITS +: TAG_WIDTH]};
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rsp_xbar_sel_in[i] = rd_req_queue_data_out[i][0 +: NUM_PORTS_IN_BITS];
        end else begin : g_no_input_sel
            assign rsp_xbar_sel_in[i] = 0;
        end
        assign rd_req_queue_pop[i] = rsp_xbar_valid_in[i] && rsp_xbar_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS (NUM_BANKS_OUT),
        .NUM_OUTPUTS(NUM_PORTS_IN),
        .DATAW      (RSP_XBAR_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_xbar_valid_in),
        .data_in   (rsp_xbar_data_in),
        .ready_in  (rsp_xbar_ready_in),
        .sel_in    (rsp_xbar_sel_in),
        .data_out  (rsp_xbar_data_out),
        .valid_out (rsp_xbar_valid_out),
        .ready_out (rsp_xbar_ready_out),
        `UNUSED_PIN (collisions),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_rsp_xbar_data_out
        assign mem_rsp_valid[i] = rsp_xbar_valid_out[i];
        assign {mem_rsp_data[i], mem_rsp_tag[i]} = rsp_xbar_data_out[i];
        assign rsp_xbar_ready_out[i] = mem_rsp_ready[i];
    end

endmodule
`TRACING_ON
