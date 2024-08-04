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
    parameter ADDR_WIDTH    = 1,
    parameter BURST_WIDTH   = 1,
    parameter NUM_BANKS     = 1,
    parameter TAG_WIDTH     = 1,
    parameter RD_QUEUE_SIZE = 1,
    parameter REQ_OUT_BUF   = 0,
    parameter RSP_OUT_BUF   = 0
) (
    input  wire                     clk,
    input  wire                     reset,

    // Memory request
    input  wire                     mem_req_valid,
    input  wire                     mem_req_rw,
    input  wire [DATA_WIDTH/8-1:0]  mem_req_byteen,
    input  wire [ADDR_WIDTH-1:0]    mem_req_addr,
    input  wire [DATA_WIDTH-1:0]    mem_req_data,
    input  wire [TAG_WIDTH-1:0]     mem_req_tag,
    output wire                     mem_req_ready,

    // Memory response
    output wire                     mem_rsp_valid,
    output wire [DATA_WIDTH-1:0]    mem_rsp_data,
    output wire [TAG_WIDTH-1:0]     mem_rsp_tag,
    input  wire                     mem_rsp_ready,

    // AVS bus
    output wire [DATA_WIDTH-1:0]    avs_writedata [NUM_BANKS],
    input  wire [DATA_WIDTH-1:0]    avs_readdata [NUM_BANKS],
    output wire [ADDR_WIDTH-1:0]    avs_address [NUM_BANKS],
    input  wire                     avs_waitrequest [NUM_BANKS],
    output wire                     avs_write [NUM_BANKS],
    output wire                     avs_read [NUM_BANKS],
    output wire [DATA_WIDTH/8-1:0]  avs_byteenable [NUM_BANKS],
    output wire [BURST_WIDTH-1:0]   avs_burstcount [NUM_BANKS],
    input  wire                     avs_readdatavalid [NUM_BANKS]
);
    localparam DATA_SIZE = DATA_WIDTH/8;
    localparam RD_QUEUE_ADDR_WIDTH = `CLOG2(RD_QUEUE_SIZE+1);
    localparam BANK_ADDRW = `LOG2UP(NUM_BANKS);
    localparam LOG2_NUM_BANKS = `CLOG2(NUM_BANKS);
    localparam BANK_OFFSETW = ADDR_WIDTH - LOG2_NUM_BANKS;

    // Requests handling //////////////////////////////////////////////////////

    wire [NUM_BANKS-1:0] req_queue_push, req_queue_pop;
    wire [NUM_BANKS-1:0][TAG_WIDTH-1:0] req_queue_tag_out;
    wire [NUM_BANKS-1:0] req_queue_going_full;
    wire [NUM_BANKS-1:0][RD_QUEUE_ADDR_WIDTH-1:0] req_queue_size;
    wire [BANK_ADDRW-1:0] req_bank_sel;
    wire [BANK_OFFSETW-1:0] req_bank_off;
    wire [NUM_BANKS-1:0] bank_req_ready;

    if (NUM_BANKS > 1) begin
        assign req_bank_sel = mem_req_addr[BANK_ADDRW-1:0];
    end else begin
        assign req_bank_sel = '0;
    end

    assign req_bank_off = mem_req_addr[ADDR_WIDTH-1:LOG2_NUM_BANKS];

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign req_queue_push[i] = mem_req_valid && ~mem_req_rw && bank_req_ready[i] && (req_bank_sel == i);
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin

        `RESET_RELAY (rd_req_reset, reset);

        VX_pending_size #(
            .SIZE (RD_QUEUE_SIZE)
        ) pending_size (
            .clk   (clk),
            .reset (rd_req_reset),
            .incr  (req_queue_push[i]),
            .decr  (req_queue_pop[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (alm_empty),
            .full  (req_queue_going_full[i]),
            `UNUSED_PIN (alm_full),
            .size  (req_queue_size[i])
        );
        `UNUSED_VAR (req_queue_size)

        VX_fifo_queue #(
            .DATAW (TAG_WIDTH),
            .DEPTH (RD_QUEUE_SIZE)
        ) rd_req_queue (
            .clk      (clk),
            .reset    (rd_req_reset),
            .push     (req_queue_push[i]),
            .pop      (req_queue_pop[i]),
            .data_in  (mem_req_tag),
            .data_out (req_queue_tag_out[i]),
            `UNUSED_PIN (empty),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        wire                  valid_out;
        wire                  rw_out;
        wire [DATA_SIZE-1:0]  byteen_out;
        wire [BANK_OFFSETW-1:0] addr_out;
        wire [DATA_WIDTH-1:0] data_out;
        wire                  ready_out;

        wire valid_out_w = mem_req_valid && ~req_queue_going_full[i] && (req_bank_sel == i);
        wire ready_out_w;

        `RESET_RELAY (req_out_reset, reset);

        VX_elastic_buffer #(
            .DATAW    (1 + DATA_SIZE + BANK_OFFSETW + DATA_WIDTH),
            .SIZE     (`TO_OUT_BUF_SIZE(REQ_OUT_BUF)),
            .OUT_REG  (`TO_OUT_BUF_REG(REQ_OUT_BUF))
        ) req_out_buf (
            .clk       (clk),
            .reset     (req_out_reset),
            .valid_in  (valid_out_w),
            .ready_in  (ready_out_w),
            .data_in   ({mem_req_rw, mem_req_byteen, req_bank_off, mem_req_data}),
            .data_out  ({rw_out,     byteen_out,     addr_out,     data_out}),
            .valid_out (valid_out),
            .ready_out (ready_out)
        );

        assign avs_read[i]       = valid_out && ~rw_out;
        assign avs_write[i]      = valid_out && rw_out;
        assign avs_address[i]    = ADDR_WIDTH'(addr_out);
        assign avs_byteenable[i] = byteen_out;
        assign avs_writedata[i]  = data_out;
        assign avs_burstcount[i] = BURST_WIDTH'(1);
        assign ready_out         = ~avs_waitrequest[i];

        assign bank_req_ready[i] = ready_out_w && ~req_queue_going_full[i];
    end

    if (NUM_BANKS > 1) begin
        assign mem_req_ready = bank_req_ready[req_bank_sel];
    end else begin
        assign mem_req_ready = bank_req_ready;
    end

    // Responses handling /////////////////////////////////////////////////////

    wire [NUM_BANKS-1:0] rsp_arb_valid_in;
    wire [NUM_BANKS-1:0][DATA_WIDTH+TAG_WIDTH-1:0] rsp_arb_data_in;
    wire [NUM_BANKS-1:0] rsp_arb_ready_in;

    wire [NUM_BANKS-1:0][DATA_WIDTH-1:0] rsp_queue_data_out;
    wire [NUM_BANKS-1:0] rsp_queue_empty;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin

        `RESET_RELAY (rd_rsp_reset, reset);

        VX_fifo_queue #(
            .DATAW (DATA_WIDTH),
            .DEPTH (RD_QUEUE_SIZE)
        ) rd_rsp_queue (
            .clk      (clk),
            .reset    (rd_rsp_reset),
            .push     (avs_readdatavalid[i]),
            .pop      (req_queue_pop[i]),
            .data_in  (avs_readdata[i]),
            .data_out (rsp_queue_data_out[i]),
            .empty    (rsp_queue_empty[i]),
            `UNUSED_PIN (full),
            `UNUSED_PIN (alm_empty),
            `UNUSED_PIN (alm_full),
            `UNUSED_PIN (size)
        );
    end

    for (genvar i = 0; i < NUM_BANKS; ++i) begin
        assign rsp_arb_valid_in[i] = !rsp_queue_empty[i];
        assign rsp_arb_data_in[i]  = {rsp_queue_data_out[i], req_queue_tag_out[i]};
        assign req_queue_pop[i]    = rsp_arb_valid_in[i] && rsp_arb_ready_in[i];
    end

    `RESET_RELAY (rsp_arb_reset, reset);

    VX_stream_arb #(
        .NUM_INPUTS (NUM_BANKS),
        .DATAW      (DATA_WIDTH + TAG_WIDTH),
        .ARBITER    ("F"),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (rsp_arb_reset),
        .valid_in  (rsp_arb_valid_in),
        .data_in   (rsp_arb_data_in),
        .ready_in  (rsp_arb_ready_in),
        .data_out  ({mem_rsp_data, mem_rsp_tag}),
        .valid_out (mem_rsp_valid),
        .ready_out (mem_rsp_ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
`TRACING_ON
