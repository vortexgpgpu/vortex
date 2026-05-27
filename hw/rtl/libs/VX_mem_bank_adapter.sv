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

`include "VX_platform.vh"

`TRACING_OFF
module VX_mem_bank_adapter #(
    parameter DATA_WIDTH     = 512,
    parameter ADDR_WIDTH_IN  = 26, // word-addressable
    parameter ADDR_WIDTH_OUT = 32, // byte-addressable
    parameter TAG_WIDTH_IN   = 8,
    parameter TAG_WIDTH_OUT  = 8,
    parameter NUM_PORTS_IN   = 1,
    parameter NUM_BANKS_OUT  = 1,
    parameter INTERLEAVE     = 0,
    parameter TAG_BUFFER_SIZE= 32,
    parameter ARBITER        = "R",
    parameter REQ_OUT_BUF    = 1,
    parameter RSP_OUT_BUF    = 1,
    parameter DATA_SIZE      = DATA_WIDTH/8
 ) (
    input  wire                     clk,
    input  wire                     reset,

    // Input request
    input wire                      mem_req_valid_in [NUM_PORTS_IN],
    input wire                      mem_req_rw_in [NUM_PORTS_IN],
    input wire [DATA_SIZE-1:0]      mem_req_byteen_in [NUM_PORTS_IN],
    input wire [ADDR_WIDTH_IN-1:0]  mem_req_addr_in [NUM_PORTS_IN],
    input wire [DATA_WIDTH-1:0]     mem_req_data_in [NUM_PORTS_IN],
    input wire [TAG_WIDTH_IN-1:0]   mem_req_tag_in [NUM_PORTS_IN],
    output wire                     mem_req_ready_in [NUM_PORTS_IN],

    // Input response
    output wire                     mem_rsp_valid_in [NUM_PORTS_IN],
    output wire [DATA_WIDTH-1:0]    mem_rsp_data_in [NUM_PORTS_IN],
    output wire [TAG_WIDTH_IN-1:0]  mem_rsp_tag_in [NUM_PORTS_IN],
    input wire                      mem_rsp_ready_in [NUM_PORTS_IN],

    // Output request
    output wire                     mem_req_valid_out [NUM_BANKS_OUT],
    output wire                     mem_req_rw_out [NUM_BANKS_OUT],
    output wire [DATA_SIZE-1:0]     mem_req_byteen_out [NUM_BANKS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] mem_req_addr_out [NUM_BANKS_OUT],
    output wire [DATA_WIDTH-1:0]    mem_req_data_out [NUM_BANKS_OUT],
    output wire [TAG_WIDTH_OUT-1:0] mem_req_tag_out [NUM_BANKS_OUT],
    input wire                      mem_req_ready_out [NUM_BANKS_OUT],

    // Output response
    input wire                      mem_rsp_valid_out [NUM_BANKS_OUT],
    input wire [DATA_WIDTH-1:0]     mem_rsp_data_out [NUM_BANKS_OUT],
    input wire [TAG_WIDTH_OUT-1:0]  mem_rsp_tag_out [NUM_BANKS_OUT],
    output wire                     mem_rsp_ready_out [NUM_BANKS_OUT]
);
    localparam BANK_SEL_BITS  = `CLOG2(NUM_BANKS_OUT);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam DST_ADDR_WDITH = ADDR_WIDTH_OUT + BANK_SEL_BITS; // convert output addresss to input space
    localparam BANK_ADDR_WIDTH = DST_ADDR_WDITH - BANK_SEL_BITS;
    localparam NUM_PORTS_IN_BITS = `CLOG2(NUM_PORTS_IN);
    localparam NUM_PORTS_IN_WIDTH = `UP(NUM_PORTS_IN_BITS);
    localparam TAG_BUFFER_ADDRW = `CLOG2(TAG_BUFFER_SIZE);
    localparam NEEDED_TAG_WIDTH = TAG_WIDTH_IN + NUM_PORTS_IN_BITS;
    localparam READ_TAG_WIDTH = (NEEDED_TAG_WIDTH > TAG_WIDTH_OUT) ? TAG_BUFFER_ADDRW : TAG_WIDTH_IN;
    localparam WRITE_TAG_WIDTH = TAG_WIDTH_IN;
    localparam XBAR_TAG_WIDTH = `MAX(READ_TAG_WIDTH, WRITE_TAG_WIDTH);
    localparam DST_TAG_WIDTH  = XBAR_TAG_WIDTH + NUM_PORTS_IN_BITS;
    localparam REQ_XBAR_DATAW = 1 + BANK_ADDR_WIDTH + DATA_SIZE + DATA_WIDTH + XBAR_TAG_WIDTH;
    localparam RSP_XBAR_DATAW = DATA_WIDTH + READ_TAG_WIDTH;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))
    `STATIC_ASSERT ((TAG_WIDTH_OUT >= DST_TAG_WIDTH), ("invalid output tag width: current=%0d, expected=%0d", TAG_WIDTH_OUT, DST_TAG_WIDTH))

    // Bank selection

    wire [NUM_PORTS_IN-1:0][BANK_SEL_WIDTH-1:0] req_bank_sel;
    wire [NUM_PORTS_IN-1:0][BANK_ADDR_WIDTH-1:0] req_bank_addr;

    if (NUM_BANKS_OUT > 1) begin : g_bank_sel
        for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_i
            wire [DST_ADDR_WDITH-1:0] mem_req_addr_dst = DST_ADDR_WDITH'(mem_req_addr_in[i]);
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
            assign req_bank_addr[i] = DST_ADDR_WDITH'(mem_req_addr_in[i]);
        end
    end

    // Tag handling logic

    wire [NUM_PORTS_IN-1:0] mem_rd_req_tag_in_ready;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_req_tag_in;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_rsp_tag_in;

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_tag_buf
        if (NEEDED_TAG_WIDTH > TAG_WIDTH_OUT) begin : g_enabled
            wire [TAG_BUFFER_ADDRW-1:0] tbuf_waddr, tbuf_raddr;
            wire tbuf_full;
            VX_index_buffer #(
                .DATAW (TAG_WIDTH_IN),
                .SIZE  (TAG_BUFFER_SIZE)
            ) tag_buf (
                .clk        (clk),
                .reset      (reset),
                .acquire_en (mem_req_valid_in[i] && ~mem_req_rw_in[i] && mem_req_ready_in[i]),
                .write_addr (tbuf_waddr),
                .write_data (mem_req_tag_in[i]),
                .read_data  (mem_rsp_tag_in[i]),
                .read_addr  (tbuf_raddr),
                .release_en (mem_rsp_valid_in[i] && mem_rsp_ready_in[i]),
                .full       (tbuf_full),
                `UNUSED_PIN (empty)
            );
            assign mem_rd_req_tag_in_ready[i] = ~tbuf_full;
            assign mem_rd_req_tag_in[i] = tbuf_waddr;
            assign tbuf_raddr = mem_rd_rsp_tag_in[i];
        end else begin : g_none
            assign mem_rd_req_tag_in_ready[i] = 1;
            assign mem_rd_req_tag_in[i] = mem_req_tag_in[i];
            assign mem_rsp_tag_in[i] = mem_rd_rsp_tag_in[i];
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
        wire tag_ready = mem_req_rw_in[i] || mem_rd_req_tag_in_ready[i];
        wire [XBAR_TAG_WIDTH-1:0] tag_value = mem_req_rw_in[i] ? XBAR_TAG_WIDTH'(mem_req_tag_in[i]) : XBAR_TAG_WIDTH'(mem_rd_req_tag_in[i]);
        assign req_xbar_valid_in[i] = mem_req_valid_in[i] && tag_ready;
        assign req_xbar_data_in[i]  = {mem_req_rw_in[i], req_bank_addr[i], mem_req_byteen_in[i], mem_req_data_in[i], tag_value};
        assign mem_req_ready_in[i] = req_xbar_ready_in[i] && tag_ready;
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

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_req_xbar_data_out

        wire xbar_rw_out;
        wire [BANK_ADDR_WIDTH-1:0] xbar_addr_out;
        wire [XBAR_TAG_WIDTH-1:0] xbar_tag_out;
        wire [DATA_WIDTH-1:0] xbar_data_out;
        wire [DATA_SIZE-1:0] xbar_byteen_out;

        assign {
            xbar_rw_out,
            xbar_addr_out,
            xbar_byteen_out,
            xbar_data_out,
            xbar_tag_out
        } = req_xbar_data_out[i];

        assign mem_req_valid_out[i]  = req_xbar_valid_out[i];
        assign mem_req_rw_out[i]     = xbar_rw_out;
        assign mem_req_addr_out[i]   = ADDR_WIDTH_OUT'(xbar_addr_out);
        assign mem_req_byteen_out[i] = xbar_byteen_out;
        assign mem_req_data_out[i]   = xbar_data_out;

        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign mem_req_tag_out[i] = TAG_WIDTH_OUT'({xbar_tag_out, req_xbar_sel_out[i]});
        end else begin : g_no_input_sel
            `UNUSED_VAR (req_xbar_sel_out[i])
            assign mem_req_tag_out[i] = TAG_WIDTH_OUT'(xbar_tag_out);
        end

        assign req_xbar_ready_out[i] = mem_req_ready_out[i];
    end

    // Responses handling

    wire [NUM_BANKS_OUT-1:0] rsp_xbar_valid_in;
    wire [NUM_BANKS_OUT-1:0][RSP_XBAR_DATAW-1:0] rsp_xbar_data_in;
    wire [NUM_BANKS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] rsp_xbar_sel_in;
    wire [NUM_BANKS_OUT-1:0] rsp_xbar_ready_in;

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_rsp_xbar_data_in
        assign rsp_xbar_valid_in[i] = mem_rsp_valid_out[i];
        assign rsp_xbar_data_in[i] = {mem_rsp_data_out[i], mem_rsp_tag_out[i][NUM_PORTS_IN_BITS +: READ_TAG_WIDTH]};
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rsp_xbar_sel_in[i] = mem_rsp_tag_out[i][0 +: NUM_PORTS_IN_BITS];
        end else begin : g_no_input_sel
            assign rsp_xbar_sel_in[i] = 0;
        end
        assign mem_rsp_ready_out[i] = rsp_xbar_ready_in[i];
    end

    wire [NUM_PORTS_IN-1:0] rsp_xbar_valid_out;
    wire [NUM_PORTS_IN-1:0][DATA_WIDTH+READ_TAG_WIDTH-1:0] rsp_xbar_data_out;
    wire [NUM_PORTS_IN-1:0] rsp_xbar_ready_out;

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
        assign mem_rsp_valid_in[i] = rsp_xbar_valid_out[i];
        assign {mem_rsp_data_in[i], mem_rd_rsp_tag_in[i]} = rsp_xbar_data_out[i];
        assign rsp_xbar_ready_out[i] = mem_rsp_ready_in[i];
    end

endmodule
`TRACING_ON
