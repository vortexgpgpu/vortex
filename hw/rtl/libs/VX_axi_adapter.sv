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
module VX_axi_adapter #(
    parameter DATA_WIDTH     = 512,
    parameter ADDR_WIDTH_IN  = 26, // word-addressable
    parameter ADDR_WIDTH_OUT = 32, // byte-addressable
    parameter TAG_WIDTH_IN   = 8,
    parameter TAG_WIDTH_OUT  = 8,
    parameter NUM_PORTS_IN   = 1,
    parameter NUM_BANKS_OUT  = 1,
    parameter INTERLEAVE     = 0,
    parameter TAG_BUFFER_SIZE= 16,
    parameter ARBITER        = "R",
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter DATA_SIZE      = DATA_WIDTH/8
 ) (
    input  wire                     clk,
    input  wire                     reset,

    // Vortex request
    input wire                      mem_req_valid [NUM_PORTS_IN],
    input wire                      mem_req_rw [NUM_PORTS_IN],
    input wire [DATA_SIZE-1:0]      mem_req_byteen [NUM_PORTS_IN],
    input wire [ADDR_WIDTH_IN-1:0]  mem_req_addr [NUM_PORTS_IN],
    input wire [DATA_WIDTH-1:0]     mem_req_data [NUM_PORTS_IN],
    input wire [TAG_WIDTH_IN-1:0]   mem_req_tag [NUM_PORTS_IN],
    output wire                     mem_req_ready [NUM_PORTS_IN],

    // Vortex response
    output wire                     mem_rsp_valid [NUM_PORTS_IN],
    output wire [DATA_WIDTH-1:0]    mem_rsp_data [NUM_PORTS_IN],
    output wire [TAG_WIDTH_IN-1:0]  mem_rsp_tag [NUM_PORTS_IN],
    input wire                      mem_rsp_ready [NUM_PORTS_IN],

    // AXI write request address channel
    output wire                     m_axi_awvalid [NUM_BANKS_OUT],
    input wire                      m_axi_awready [NUM_BANKS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_awaddr [NUM_BANKS_OUT],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_awid [NUM_BANKS_OUT],
    output wire [7:0]               m_axi_awlen [NUM_BANKS_OUT],
    output wire [2:0]               m_axi_awsize [NUM_BANKS_OUT],
    output wire [1:0]               m_axi_awburst [NUM_BANKS_OUT],
    output wire [1:0]               m_axi_awlock [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_awcache [NUM_BANKS_OUT],
    output wire [2:0]               m_axi_awprot [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_awqos [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_awregion [NUM_BANKS_OUT],

    // AXI write request data channel
    output wire                     m_axi_wvalid [NUM_BANKS_OUT],
    input wire                      m_axi_wready [NUM_BANKS_OUT],
    output wire [DATA_WIDTH-1:0]    m_axi_wdata [NUM_BANKS_OUT],
    output wire [DATA_SIZE-1:0]     m_axi_wstrb [NUM_BANKS_OUT],
    output wire                     m_axi_wlast [NUM_BANKS_OUT],

    // AXI write response channel
    input wire                      m_axi_bvalid [NUM_BANKS_OUT],
    output wire                     m_axi_bready [NUM_BANKS_OUT],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_bid [NUM_BANKS_OUT],
    input wire [1:0]                m_axi_bresp [NUM_BANKS_OUT],

    // AXI read address channel
    output wire                     m_axi_arvalid [NUM_BANKS_OUT],
    input wire                      m_axi_arready [NUM_BANKS_OUT],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_araddr [NUM_BANKS_OUT],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_arid [NUM_BANKS_OUT],
    output wire [7:0]               m_axi_arlen [NUM_BANKS_OUT],
    output wire [2:0]               m_axi_arsize [NUM_BANKS_OUT],
    output wire [1:0]               m_axi_arburst [NUM_BANKS_OUT],
    output wire [1:0]               m_axi_arlock [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_arcache [NUM_BANKS_OUT],
    output wire [2:0]               m_axi_arprot [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_arqos [NUM_BANKS_OUT],
    output wire [3:0]               m_axi_arregion [NUM_BANKS_OUT],

    // AXI read response channel
    input wire                      m_axi_rvalid [NUM_BANKS_OUT],
    output wire                     m_axi_rready [NUM_BANKS_OUT],
    input wire [DATA_WIDTH-1:0]     m_axi_rdata [NUM_BANKS_OUT],
    input wire                      m_axi_rlast [NUM_BANKS_OUT],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_rid [NUM_BANKS_OUT],
    input wire [1:0]                m_axi_rresp [NUM_BANKS_OUT]
);
    localparam LOG2_DATA_SIZE = `CLOG2(DATA_SIZE);
    localparam BANK_SEL_BITS  = `CLOG2(NUM_BANKS_OUT);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam DST_ADDR_WDITH = (ADDR_WIDTH_OUT - LOG2_DATA_SIZE) + BANK_SEL_BITS; // convert byte-addressable output addresss to block-addressable input space
    localparam BANK_ADDR_WIDTH = DST_ADDR_WDITH - BANK_SEL_BITS;
    localparam NUM_PORTS_IN_BITS = `CLOG2(NUM_PORTS_IN);
    localparam NUM_PORTS_IN_WIDTH = `UP(NUM_PORTS_IN_BITS);
    localparam TAG_BUFFER_ADDRW = `CLOG2(TAG_BUFFER_SIZE);
    localparam NEEDED_TAG_WIDTH = TAG_WIDTH_IN + NUM_PORTS_IN_BITS;
    localparam READ_TAG_WIDTH = (NEEDED_TAG_WIDTH > TAG_WIDTH_OUT) ? TAG_BUFFER_ADDRW : TAG_WIDTH_IN;
    localparam READ_FULL_TAG_WIDTH = READ_TAG_WIDTH + NUM_PORTS_IN_BITS;
    localparam WRITE_TAG_WIDTH = `MIN(TAG_WIDTH_IN, TAG_WIDTH_OUT);
    localparam DST_TAG_WIDTH  = `MAX(READ_FULL_TAG_WIDTH, WRITE_TAG_WIDTH);
    localparam XBAR_TAG_WIDTH = `MAX(READ_TAG_WIDTH, WRITE_TAG_WIDTH);
    localparam REQ_XBAR_DATAW = 1 + BANK_ADDR_WIDTH + DATA_SIZE + DATA_WIDTH + XBAR_TAG_WIDTH;
    localparam RSP_XBAR_DATAW = DATA_WIDTH + READ_TAG_WIDTH;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))
    `STATIC_ASSERT ((TAG_WIDTH_OUT >= DST_TAG_WIDTH), ("invalid output tag width: current=%0d, expected=%0d", TAG_WIDTH_OUT, DST_TAG_WIDTH))

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

    // Tag handling logic

    wire [NUM_PORTS_IN-1:0] mem_rd_req_tag_ready;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_req_tag;
    wire [NUM_PORTS_IN-1:0][READ_TAG_WIDTH-1:0] mem_rd_rsp_tag;

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
                .acquire_en (mem_req_valid[i] && ~mem_req_rw[i] && mem_req_ready[i]),
                .write_addr (tbuf_waddr),
                .write_data (mem_req_tag[i]),
                .read_data  (mem_rsp_tag[i]),
                .read_addr  (tbuf_raddr),
                .release_en (mem_rsp_valid[i] && mem_rsp_ready[i]),
                .full       (tbuf_full),
                `UNUSED_PIN (empty)
            );
            assign mem_rd_req_tag_ready[i] = ~tbuf_full;
            assign mem_rd_req_tag[i] = tbuf_waddr;
            assign tbuf_raddr = mem_rd_rsp_tag[i];
        end else begin : g_none
            assign mem_rd_req_tag_ready[i] = 1;
            assign mem_rd_req_tag[i] = mem_req_tag[i];
            assign mem_rsp_tag[i] = mem_rd_rsp_tag[i];
        end
    end

    // AXI request handling

    wire [NUM_PORTS_IN-1:0] req_xbar_valid_in;
    wire [NUM_PORTS_IN-1:0][REQ_XBAR_DATAW-1:0] req_xbar_data_in;
    wire [NUM_PORTS_IN-1:0] req_xbar_ready_in;

    wire [NUM_BANKS_OUT-1:0] req_xbar_valid_out;
    wire [NUM_BANKS_OUT-1:0][REQ_XBAR_DATAW-1:0] req_xbar_data_out;
    wire [NUM_BANKS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] req_xbar_sel_out;
    wire [NUM_BANKS_OUT-1:0] req_xbar_ready_out;

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_req_xbar_data_in
        wire tag_ready = mem_req_rw[i] || mem_rd_req_tag_ready[i];
        wire [XBAR_TAG_WIDTH-1:0] tag_value = mem_req_rw[i] ? XBAR_TAG_WIDTH'(mem_req_tag[i]) : XBAR_TAG_WIDTH'(mem_rd_req_tag[i]);
        assign req_xbar_valid_in[i] = mem_req_valid[i] && tag_ready;
        assign req_xbar_data_in[i]  = {mem_req_rw[i], req_bank_addr[i], mem_req_byteen[i], mem_req_data[i], tag_value};
        assign mem_req_ready[i]  = req_xbar_ready_in[i] && tag_ready;
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

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_axi_reqs

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

        // AXi request handshake

        wire m_axi_aw_ack, m_axi_w_ack, axi_write_ready;

        VX_axi_write_ack axi_write_ack (
            .clk    (clk),
            .reset  (reset),
            .awvalid(m_axi_awvalid[i]),
            .awready(m_axi_awready[i]),
            .wvalid (m_axi_wvalid[i]),
            .wready (m_axi_wready[i]),
            .aw_ack (m_axi_aw_ack),
            .w_ack  (m_axi_w_ack),
            .tx_rdy (axi_write_ready),
            `UNUSED_PIN (tx_ack)
        );

        assign req_xbar_ready_out[i] = xbar_rw_out ? axi_write_ready : m_axi_arready[i];

        // AXI write address channel

        assign m_axi_awvalid[i] = req_xbar_valid_out[i] && xbar_rw_out && ~m_axi_aw_ack;

    if (INTERLEAVE) begin : g_m_axi_awaddr_i
        assign m_axi_awaddr[i]  = (ADDR_WIDTH_OUT'(xbar_addr_out) << (BANK_SEL_BITS + LOG2_DATA_SIZE)) | (ADDR_WIDTH_OUT'(i) << LOG2_DATA_SIZE);
    end else begin : g_m_axi_awaddr_ni
        assign m_axi_awaddr[i]  = (ADDR_WIDTH_OUT'(xbar_addr_out) << LOG2_DATA_SIZE) | (ADDR_WIDTH_OUT'(i) << (BANK_ADDR_WIDTH + LOG2_DATA_SIZE));
    end

        assign m_axi_awid[i]    = TAG_WIDTH_OUT'(xbar_tag_out);
        assign m_axi_awlen[i]   = 8'b00000000;
        assign m_axi_awsize[i]  = 3'(LOG2_DATA_SIZE);
        assign m_axi_awburst[i] = 2'b00;
        assign m_axi_awlock[i]  = 2'b00;
        assign m_axi_awcache[i] = 4'b0000;
        assign m_axi_awprot[i]  = 3'b000;
        assign m_axi_awqos[i]   = 4'b0000;
        assign m_axi_awregion[i]= 4'b0000;

        // AXI write data channel

        assign m_axi_wvalid[i]  = req_xbar_valid_out[i] && xbar_rw_out && ~m_axi_w_ack;
        assign m_axi_wstrb[i]   = xbar_byteen_out;
        assign m_axi_wdata[i]   = xbar_data_out;
        assign m_axi_wlast[i]   = 1'b1;

        // AXI read address channel

        wire [READ_FULL_TAG_WIDTH-1:0] xbar_tag_r_out;
        if (NUM_PORTS_IN > 1) begin : g_xbar_tag_r_out
            assign xbar_tag_r_out = READ_FULL_TAG_WIDTH'({xbar_tag_out, req_xbar_sel_out[i]});
        end else begin : g_no_input_sel
            `UNUSED_VAR (req_xbar_sel_out)
            assign xbar_tag_r_out = READ_TAG_WIDTH'(xbar_tag_out);
        end

        assign m_axi_arvalid[i] = req_xbar_valid_out[i] && ~xbar_rw_out;

        // convert address to byte-addressable space
    if (INTERLEAVE) begin : g_m_axi_araddr_i
        assign m_axi_araddr[i]  = (ADDR_WIDTH_OUT'(xbar_addr_out) << (BANK_SEL_BITS + LOG2_DATA_SIZE)) | (ADDR_WIDTH_OUT'(i) << LOG2_DATA_SIZE);
    end else begin : g_m_axi_araddr_ni
        assign m_axi_araddr[i]  = (ADDR_WIDTH_OUT'(xbar_addr_out) << LOG2_DATA_SIZE) | (ADDR_WIDTH_OUT'(i) << (BANK_ADDR_WIDTH + LOG2_DATA_SIZE));
    end
        assign m_axi_arid[i]    = TAG_WIDTH_OUT'(xbar_tag_r_out);
        assign m_axi_arlen[i]   = 8'b00000000;
        assign m_axi_arsize[i]  = 3'(LOG2_DATA_SIZE);
        assign m_axi_arburst[i] = 2'b00;
        assign m_axi_arlock[i]  = 2'b00;
        assign m_axi_arcache[i] = 4'b0000;
        assign m_axi_arprot[i]  = 3'b000;
        assign m_axi_arqos[i]   = 4'b0000;
        assign m_axi_arregion[i]= 4'b0000;
    end

    // AXI write response channel (ignore)

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_axi_write_rsp
        `UNUSED_VAR (m_axi_bvalid[i])
        `UNUSED_VAR (m_axi_bid[i])
        `UNUSED_VAR (m_axi_bresp[i])
        assign m_axi_bready[i] = 1'b1;
        `RUNTIME_ASSERT(~m_axi_bvalid[i] || m_axi_bresp[i] == 0, ("%t: *** AXI response error", $time))
    end

    // AXI read response channel

    wire [NUM_BANKS_OUT-1:0] rsp_xbar_valid_in;
    wire [NUM_BANKS_OUT-1:0][RSP_XBAR_DATAW-1:0] rsp_xbar_data_in;
    wire [NUM_BANKS_OUT-1:0][NUM_PORTS_IN_WIDTH-1:0] rsp_xbar_sel_in;
    wire [NUM_BANKS_OUT-1:0] rsp_xbar_ready_in;

    for (genvar i = 0; i < NUM_BANKS_OUT; ++i) begin : g_rsp_xbar_data_in
        assign rsp_xbar_valid_in[i] = m_axi_rvalid[i];
        assign rsp_xbar_data_in[i] = {m_axi_rdata[i], m_axi_rid[i][NUM_PORTS_IN_BITS +: READ_TAG_WIDTH]};
        if (NUM_PORTS_IN > 1) begin : g_input_sel
            assign rsp_xbar_sel_in[i] = m_axi_rid[i][0 +: NUM_PORTS_IN_BITS];
        end else begin : g_no_input_sel
            assign rsp_xbar_sel_in[i] = 0;
        end
        assign m_axi_rready[i] = rsp_xbar_ready_in[i];
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rlast[i] == 0), ("%t: *** AXI response error", $time))
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rresp[i] != 0), ("%t: *** AXI response error", $time))
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
        assign mem_rsp_valid[i] = rsp_xbar_valid_out[i];
        assign {mem_rsp_data[i], mem_rd_rsp_tag[i]} = rsp_xbar_data_out[i];
        assign rsp_xbar_ready_out[i] = mem_rsp_ready[i];
    end

endmodule
`TRACING_ON
