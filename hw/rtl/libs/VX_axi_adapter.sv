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
    parameter ADDR_WIDTH_IN  = 1,
    parameter ADDR_WIDTH_OUT = 32,
    parameter TAG_WIDTH_IN   = 8,
    parameter TAG_WIDTH_OUT  = 8,
    parameter NUM_BANKS      = 1,
    parameter BANK_INTERLEAVE= 0,
    parameter TAG_BUFFER_SIZE= 32,
    parameter RSP_OUT_BUF    = 0
) (
    input  wire                     clk,
    input  wire                     reset,

    // Vortex request
    input wire                      mem_req_valid,
    input wire                      mem_req_rw,
    input wire [DATA_WIDTH/8-1:0]   mem_req_byteen,
    input wire [ADDR_WIDTH_IN-1:0]  mem_req_addr,
    input wire [DATA_WIDTH-1:0]     mem_req_data,
    input wire [TAG_WIDTH_IN-1:0]   mem_req_tag,
    output wire                     mem_req_ready,

    // Vortex response
    output wire                     mem_rsp_valid,
    output wire [DATA_WIDTH-1:0]    mem_rsp_data,
    output wire [TAG_WIDTH_IN-1:0]  mem_rsp_tag,
    input wire                      mem_rsp_ready,

    // AXI write request address channel
    output wire                     m_axi_awvalid [NUM_BANKS],
    input wire                      m_axi_awready [NUM_BANKS],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_awaddr [NUM_BANKS],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_awid [NUM_BANKS],
    output wire [7:0]               m_axi_awlen [NUM_BANKS],
    output wire [2:0]               m_axi_awsize [NUM_BANKS],
    output wire [1:0]               m_axi_awburst [NUM_BANKS],
    output wire [1:0]               m_axi_awlock [NUM_BANKS],
    output wire [3:0]               m_axi_awcache [NUM_BANKS],
    output wire [2:0]               m_axi_awprot [NUM_BANKS],
    output wire [3:0]               m_axi_awqos [NUM_BANKS],
    output wire [3:0]               m_axi_awregion [NUM_BANKS],

    // AXI write request data channel
    output wire                     m_axi_wvalid [NUM_BANKS],
    input wire                      m_axi_wready [NUM_BANKS],
    output wire [DATA_WIDTH-1:0]    m_axi_wdata [NUM_BANKS],
    output wire [DATA_WIDTH/8-1:0]  m_axi_wstrb [NUM_BANKS],
    output wire                     m_axi_wlast [NUM_BANKS],

    // AXI write response channel
    input wire                      m_axi_bvalid [NUM_BANKS],
    output wire                     m_axi_bready [NUM_BANKS],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_bid [NUM_BANKS],
    input wire [1:0]                m_axi_bresp [NUM_BANKS],

    // AXI read address channel
    output wire                     m_axi_arvalid [NUM_BANKS],
    input wire                      m_axi_arready [NUM_BANKS],
    output wire [ADDR_WIDTH_OUT-1:0] m_axi_araddr [NUM_BANKS],
    output wire [TAG_WIDTH_OUT-1:0] m_axi_arid [NUM_BANKS],
    output wire [7:0]               m_axi_arlen [NUM_BANKS],
    output wire [2:0]               m_axi_arsize [NUM_BANKS],
    output wire [1:0]               m_axi_arburst [NUM_BANKS],
    output wire [1:0]               m_axi_arlock [NUM_BANKS],
    output wire [3:0]               m_axi_arcache [NUM_BANKS],
    output wire [2:0]               m_axi_arprot [NUM_BANKS],
    output wire [3:0]               m_axi_arqos [NUM_BANKS],
    output wire [3:0]               m_axi_arregion [NUM_BANKS],

    // AXI read response channel
    input wire                      m_axi_rvalid [NUM_BANKS],
    output wire                     m_axi_rready [NUM_BANKS],
    input wire [DATA_WIDTH-1:0]     m_axi_rdata [NUM_BANKS],
    input wire                      m_axi_rlast [NUM_BANKS],
    input wire [TAG_WIDTH_OUT-1:0]  m_axi_rid [NUM_BANKS],
    input wire [1:0]                m_axi_rresp [NUM_BANKS]
);
    localparam DATA_SIZE      = `CLOG2(DATA_WIDTH/8);
    localparam BANK_SEL_BITS  = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH = `UP(BANK_SEL_BITS);
    localparam DST_ADDR_WDITH = ADDR_WIDTH_OUT + BANK_SEL_BITS - `CLOG2(DATA_WIDTH/8); // to input space
    localparam BANK_OFFSETW   = DST_ADDR_WDITH - BANK_SEL_BITS;

    `STATIC_ASSERT ((DST_ADDR_WDITH >= ADDR_WIDTH_IN), ("invalid address width: current=%0d, expected=%0d", DST_ADDR_WDITH, ADDR_WIDTH_IN))

    wire [BANK_OFFSETW-1:0]   req_bank_off;
    wire [BANK_SEL_WIDTH-1:0] req_bank_sel;

    wire [DST_ADDR_WDITH-1:0] mem_req_addr_out = DST_ADDR_WDITH'(mem_req_addr);

    if (NUM_BANKS > 1) begin : g_bank_sel
        if (BANK_INTERLEAVE) begin : g_interleave
            assign req_bank_sel = mem_req_addr_out[BANK_SEL_BITS-1:0];
            assign req_bank_off = mem_req_addr_out[BANK_SEL_BITS +: BANK_OFFSETW];
        end else begin : g_no_interleave
            assign req_bank_sel = mem_req_addr_out[BANK_OFFSETW +: BANK_SEL_BITS];
            assign req_bank_off = mem_req_addr_out[BANK_OFFSETW-1:0];
        end
    end else begin : g_no_bank_sel
        assign req_bank_sel = '0;
        assign req_bank_off = mem_req_addr_out;
    end

    // AXi write request synchronization
    reg [NUM_BANKS-1:0] m_axi_aw_ack, m_axi_w_ack, axi_write_ready;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_write_ready
        VX_axi_write_ack axi_write_ack (
            .clk    (clk),
            .reset  (reset),
            .awvalid(m_axi_awvalid[i]),
            .awready(m_axi_awready[i]),
            .wvalid (m_axi_wvalid[i]),
            .wready (m_axi_wready[i]),
            .aw_ack (m_axi_aw_ack[i]),
            .w_ack  (m_axi_w_ack[i]),
            .tx_rdy (axi_write_ready[i]),
            `UNUSED_PIN (tx_ack)
        );
    end

    wire mem_req_tag_ready;
    wire [TAG_WIDTH_OUT-1:0] mem_req_tag_out;
    wire [TAG_WIDTH_OUT-1:0] mem_rsp_tag_out;

    // handle tag width mismatch
    if (TAG_WIDTH_IN > TAG_WIDTH_OUT) begin : g_tag_buf
        localparam TBUF_ADDRW = `CLOG2(TAG_BUFFER_SIZE);
        wire [TBUF_ADDRW-1:0] tbuf_waddr, tbuf_raddr;
        wire tbuf_full;
        VX_index_buffer #(
            .DATAW (TAG_WIDTH_IN),
            .SIZE  (TAG_BUFFER_SIZE)
        ) tag_buf (
            .clk        (clk),
            .reset      (reset),
            .acquire_en (mem_req_valid && ~mem_req_rw && mem_req_ready),
            .write_addr (tbuf_waddr),
            .write_data (mem_req_tag),
            .read_data  (mem_rsp_tag),
            .read_addr  (tbuf_raddr),
            .release_en (mem_rsp_valid && mem_rsp_ready),
            .full       (tbuf_full),
            `UNUSED_PIN (empty)
        );
        assign mem_req_tag_ready = mem_req_rw || ~tbuf_full;
        assign mem_req_tag_out = TAG_WIDTH_OUT'(tbuf_waddr);
        assign tbuf_raddr = mem_rsp_tag_out[TBUF_ADDRW-1:0];
        `UNUSED_VAR (mem_rsp_tag_out)
    end else begin : g_no_tag_buf
        assign mem_req_tag_ready = 1;
        assign mem_req_tag_out = TAG_WIDTH_OUT'(mem_req_tag);
        assign mem_rsp_tag = mem_rsp_tag_out[TAG_WIDTH_IN-1:0];
        `UNUSED_VAR (mem_rsp_tag_out)
    end

    // request ack
    assign mem_req_ready = mem_req_rw ? axi_write_ready[req_bank_sel] :
                                        (m_axi_arready[req_bank_sel] && mem_req_tag_ready);

    // AXI write request address channel
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_write_addr
        assign m_axi_awvalid[i] = mem_req_valid && mem_req_rw && (req_bank_sel == i) && ~m_axi_aw_ack[i];
        assign m_axi_awaddr[i]  = ADDR_WIDTH_OUT'(req_bank_off) << `CLOG2(DATA_WIDTH/8);
        assign m_axi_awid[i]    = mem_req_tag_out;
        assign m_axi_awlen[i]   = 8'b00000000;
        assign m_axi_awsize[i]  = 3'(DATA_SIZE);
        assign m_axi_awburst[i] = 2'b00;
        assign m_axi_awlock[i]  = 2'b00;
        assign m_axi_awcache[i] = 4'b0000;
        assign m_axi_awprot[i]  = 3'b000;
        assign m_axi_awqos[i]   = 4'b0000;
        assign m_axi_awregion[i]= 4'b0000;
    end

    // AXI write request data channel
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_write_data
        assign m_axi_wvalid[i] = mem_req_valid && mem_req_rw && (req_bank_sel == i) && ~m_axi_w_ack[i];
        assign m_axi_wdata[i]  = mem_req_data;
        assign m_axi_wstrb[i]  = mem_req_byteen;
        assign m_axi_wlast[i]  = 1'b1;
    end

    // AXI write response channel (ignore)
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_write_rsp
        `UNUSED_VAR (m_axi_bvalid[i])
        `UNUSED_VAR (m_axi_bid[i])
        `UNUSED_VAR (m_axi_bresp[i])
        assign m_axi_bready[i] = 1'b1;
        `RUNTIME_ASSERT(~m_axi_bvalid[i] || m_axi_bresp[i] == 0, ("%t: *** AXI response error", $time))
    end

    // AXI read request channel
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_read_req
        assign m_axi_arvalid[i] = mem_req_valid && ~mem_req_rw && (req_bank_sel == i) && mem_req_tag_ready;
        assign m_axi_araddr[i]  = ADDR_WIDTH_OUT'(req_bank_off) << `CLOG2(DATA_WIDTH/8);
        assign m_axi_arid[i]    = mem_req_tag_out;
        assign m_axi_arlen[i]   = 8'b00000000;
        assign m_axi_arsize[i]  = 3'(DATA_SIZE);
        assign m_axi_arburst[i] = 2'b00;
        assign m_axi_arlock[i]  = 2'b00;
        assign m_axi_arcache[i] = 4'b0000;
        assign m_axi_arprot[i]  = 3'b000;
        assign m_axi_arqos[i]   = 4'b0000;
        assign m_axi_arregion[i]= 4'b0000;
    end

    // AXI read response channel

    wire [NUM_BANKS-1:0] rsp_arb_valid_in;
    wire [NUM_BANKS-1:0][DATA_WIDTH+TAG_WIDTH_OUT-1:0] rsp_arb_data_in;
    wire [NUM_BANKS-1:0] rsp_arb_ready_in;

    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_axi_read_rsp
        assign rsp_arb_valid_in[i] = m_axi_rvalid[i];
        assign rsp_arb_data_in[i] = {m_axi_rdata[i], m_axi_rid[i]};
        assign m_axi_rready[i] = rsp_arb_ready_in[i];
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rlast[i] == 0), ("%t: *** AXI response error", $time))
        `RUNTIME_ASSERT(~(m_axi_rvalid[i] && m_axi_rresp[i] != 0), ("%t: *** AXI response error", $time))
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_BANKS),
        .DATAW      (DATA_WIDTH + TAG_WIDTH_OUT),
        .ARBITER    ("R"),
        .OUT_BUF    (RSP_OUT_BUF)
    ) rsp_arb (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (rsp_arb_valid_in),
        .data_in   (rsp_arb_data_in),
        .ready_in  (rsp_arb_ready_in),
        .data_out  ({mem_rsp_data, mem_rsp_tag_out}),
        .valid_out (mem_rsp_valid),
        .ready_out (mem_rsp_ready),
        `UNUSED_PIN (sel_out)
    );

endmodule
`TRACING_ON
