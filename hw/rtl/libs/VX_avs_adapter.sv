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
    parameter NUM_PORTS_OUT = 1,
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
    output wire [DATA_WIDTH-1:0]    avs_writedata [NUM_PORTS_OUT],
    input  wire [DATA_WIDTH-1:0]    avs_readdata [NUM_PORTS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] avs_address [NUM_PORTS_OUT],
    input  wire                     avs_waitrequest [NUM_PORTS_OUT],
    output wire                     avs_write [NUM_PORTS_OUT],
    output wire                     avs_read [NUM_PORTS_OUT],
    output wire [DATA_WIDTH/8-1:0]  avs_byteenable [NUM_PORTS_OUT],
    output wire [BURST_WIDTH-1:0]   avs_burstcount [NUM_PORTS_OUT],
    input  wire                     avs_readdatavalid [NUM_PORTS_OUT]
);
    localparam DATA_SIZE      = DATA_WIDTH/8;
    localparam PORT_SEL_BITS  = `CLOG2(NUM_PORTS_OUT);
    localparam PORT_SEL_WIDTH = `UP(PORT_SEL_BITS);
    localparam DST_ADDR_WDITH = ADDR_WIDTH_OUT + PORT_SEL_BITS; // to input space
    localparam PORT_OFFSETW   = DST_ADDR_WDITH - PORT_SEL_BITS;
    localparam NUM_PORTS_IN_BITS = `CLOG2(NUM_PORTS_IN);
    localparam NUM_PORTS_IN_WIDTH = `UP(NUM_PORTS_IN_BITS);
    localparam REQ_QUEUE_DATAW = TAG_WIDTH + NUM_PORTS_IN_BITS;
    localparam ARB_DATAW = 1 + PORT_OFFSETW + DATA_WIDTH + DATA_SIZE + TAG_WIDTH;
    localparam RSP_DATAW = DATA_WIDTH + TAG_WIDTH;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))

    // Ports selection

    wire [NUM_PORTS_IN-1:0][PORT_SEL_WIDTH-1:0] req_port_out_sel;
    wire [NUM_PORTS_IN-1:0][PORT_OFFSETW-1:0] req_port_out_off;

    if (NUM_PORTS_OUT > 1) begin : g_port_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            wire [DST_ADDR_WDITH-1:0] mem_req_addr_out = DST_ADDR_WDITH'(mem_req_addr[i]);
            if (INTERLEAVE) begin : g_interleave
                assign req_port_out_sel[i] = mem_req_addr_out[PORT_SEL_BITS-1:0];
                assign req_port_out_off[i] = mem_req_addr_out[PORT_SEL_BITS +: PORT_OFFSETW];
            end else begin : g_no_interleave
                assign req_port_out_sel[i] = mem_req_addr_out[PORT_OFFSETW +: PORT_SEL_BITS];
                assign req_port_out_off[i] = mem_req_addr_out[PORT_OFFSETW-1:0];
            end
        end
    end else begin : g_no_port_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            assign req_port_out_sel[i] = '0;
            assign req_port_out_off[i] = DST_ADDR_WDITH'(mem_req_addr[i]);
        end
    end

    // Request ack

    wire [NUM_PORTS_OUT-1:0][NUM_PORTS_IN-1:0] arb_ready_in;
    wire [NUM_PORTS_IN-1:0][NUM_PORTS_OUT-1:0] arb_ready_in_w;

    VX_transpose #(
        .N (NUM_PORTS_OUT),
        .M (NUM_PORTS_IN)
    ) rdy_in_transpose (
        .data_in  (arb_ready_in),
        .data_out (arb_ready_in_w)
    );

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_ready_in
        assign mem_req_ready[i] = | arb_ready_in_w[i];
    end

    // Request handling ///////////////////////////////////////////////////////

    wire [NUM_PORTS_OUT-1:0][REQ_QUEUE_DATAW-1:0] rd_req_queue_data_out;
    wire [NUM_PORTS_OUT-1:0] rd_req_queue_pop;

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_requests

        wire [PORT_OFFSETW-1:0] arb_addr_out;
        wire [TAG_WIDTH-1:0] arb_tag_out;
        wire [NUM_PORTS_IN_WIDTH-1:0] arb_sel_out;
        wire [DATA_WIDTH-1:0] arb_data_out;
        wire [DATA_SIZE-1:0] arb_byteen_out;
        wire arb_valid_out, arb_ready_out;
        wire arb_rw_out;

        wire [NUM_PORTS_IN-1:0][ARB_DATAW-1:0] arb_data_in;
        wire [NUM_PORTS_IN-1:0] arb_valid_in;

        for (genvar j = 0; j < NUM_PORTS_IN; ++j) begin : g_valid_in
            assign arb_valid_in[j] = mem_req_valid[j] && (req_port_out_sel[j] == i);
        end

        for (genvar j = 0; j < NUM_PORTS_IN; ++j) begin : g_data_in
            assign arb_data_in[j] = {mem_req_rw[j], req_port_out_off[j], mem_req_byteen[j], mem_req_data[j], mem_req_tag[j]};
        end

        VX_stream_arb #(
            .NUM_INPUTS (NUM_PORTS_IN),
            .NUM_OUTPUTS(1),
            .DATAW      (ARB_DATAW),
            .ARBITER    (ARBITER)
        ) req_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_in),
            .ready_in  (arb_ready_in[i]),
            .data_in   (arb_data_in),
            .data_out  ({arb_rw_out, arb_addr_out, arb_byteen_out, arb_data_out, arb_tag_out}),
            .valid_out (arb_valid_out),
            .ready_out (arb_ready_out),
            .sel_out   (arb_sel_out)
        );

        wire rd_req_queue_going_full;
        wire rd_req_queue_push;

        assign rd_req_queue_push = arb_valid_out && arb_ready_out && ~arb_rw_out;

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
            assign rd_req_queue_data_in = {arb_tag_out, arb_sel_out};
        end else begin : g_no_input_sel
            `UNUSED_VAR (arb_sel_out)
            assign rd_req_queue_data_in = arb_tag_out;
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

        wire                  buf_valid_out;
        wire                  buf_rw_out;
        wire [DATA_SIZE-1:0]  buf_byteen_out;
        wire [PORT_OFFSETW-1:0] buf_addr_out;
        wire [DATA_WIDTH-1:0] buf_data_out;
        wire                  buf_ready_out;

        // stall pipeline if the request queue is needed and going full
        wire arb_valid_out_w, arb_ready_out_w;
        wire rd_req_queue_ready = arb_rw_out || ~rd_req_queue_going_full;
        assign arb_valid_out_w = arb_valid_out && rd_req_queue_ready;
        assign arb_ready_out = arb_ready_out_w && rd_req_queue_ready;

        VX_elastic_buffer #(
            .DATAW    (1 + DATA_SIZE + PORT_OFFSETW + DATA_WIDTH),
            .SIZE     (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
            .OUT_REG  (`TO_OUT_BUF_REG(REQ_OUT_BUF))
        ) req_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_out_w),
            .ready_in  (arb_ready_out_w),
            .data_in   ({arb_rw_out, arb_byteen_out, arb_addr_out, arb_data_out}),
            .data_out  ({buf_rw_out, buf_byteen_out, buf_addr_out, buf_data_out}),
            .valid_out (buf_valid_out),
            .ready_out (buf_ready_out)
        );

        assign avs_read[i]       = buf_valid_out && ~buf_rw_out;
        assign avs_write[i]      = buf_valid_out && buf_rw_out;
        assign avs_address[i]    = ADDR_WIDTH_OUT'(buf_addr_out);
        assign avs_byteenable[i] = buf_byteen_out;
        assign avs_writedata[i]  = buf_data_out;
        assign avs_burstcount[i] = BURST_WIDTH'(1);
        assign buf_ready_out     = ~avs_waitrequest[i];
    end

    // Responses handling /////////////////////////////////////////////////////

    wire [NUM_PORTS_OUT-1:0] rd_rsp_valid_in;
    wire [NUM_PORTS_OUT-1:0][RSP_DATAW-1:0] rd_rsp_data_in;
    wire [NUM_PORTS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] rd_rsp_sel_in;
    wire [NUM_PORTS_OUT-1:0] rd_rsp_ready_in;

    wire [NUM_PORTS_IN-1:0] rd_rsp_valid_out;
    wire [NUM_PORTS_IN-1:0][RSP_DATAW-1:0] rd_rsp_data_out;
    wire [NUM_PORTS_IN-1:0] rd_rsp_ready_out;

    for (genvar i = 0; i < NUM_PORTS_OUT; ++i) begin : g_rd_rsp_queues

        wire [DATA_WIDTH-1:0] rd_rsp_queue_data_out;
        wire rd_rsp_queue_empty;

        VX_fifo_queue #(
            .DATAW (DATA_WIDTH),
            .DEPTH (RD_QUEUE_SIZE)
        ) rd_rsp_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (avs_readdatavalid[i]),
            .pop      (rd_req_queue_pop[i]),
            .data_in  (avs_readdata[i]),
            .data_out (rd_rsp_queue_data_out),
            .empty    (rd_rsp_queue_empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );

        assign rd_rsp_valid_in[i] = ~rd_rsp_queue_empty;
        assign rd_rsp_data_in[i] = {rd_rsp_queue_data_out, rd_req_queue_data_out[i][NUM_PORTS_IN_BITS +: TAG_WIDTH]};
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rd_rsp_sel_in[i] = rd_req_queue_data_out[i][0 +: NUM_PORTS_IN_BITS];
        end else begin : g_no_input_sel
            assign rd_rsp_sel_in[i] = 0;
        end
        assign rd_req_queue_pop[i] = rd_rsp_valid_in[i] && rd_rsp_ready_in[i];
    end

    VX_stream_xbar #(
        .NUM_INPUTS (NUM_PORTS_OUT),
        .NUM_OUTPUTS(NUM_PORTS_IN),
        .DATAW      (RSP_DATAW),
        .ARBITER    (ARBITER),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rd_rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rd_rsp_valid_in),
        .data_in   (rd_rsp_data_in),
        .ready_in  (rd_rsp_ready_in),
        .sel_in    (rd_rsp_sel_in),
        .data_out  (rd_rsp_data_out),
        .valid_out (rd_rsp_valid_out),
        .ready_out (rd_rsp_ready_out),
        `UNUSED_PIN (collisions),
        `UNUSED_PIN (sel_out)
    );

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_rd_rsp_data_out
        assign mem_rsp_valid[i] = rd_rsp_valid_out[i];
        assign {mem_rsp_data[i], mem_rsp_tag[i]} = rd_rsp_data_out[i];
        assign rd_rsp_ready_out[i] = mem_rsp_ready[i];
    end

endmodule
`TRACING_ON
