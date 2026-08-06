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

// ============================================================================
// VX_mem_axi_xbar — N-master to M-slave AXI4 crossbar on VX_mem_axi_if bundles.
//
// Interface convenience wrapper over the flat-port libs core VX_mm_axi_xbar.
// For NUM_OUTPUTS > 1 each master is routed to a slave by the high address bits
// (equal-size regions). For NUM_OUTPUTS == 1 it degenerates to a fan-in and
// MULTI_OUT selects single- vs multi-outstanding (ID-routed) arbitration — the
// path the Command Processor uses to merge its sources onto one memory master.
//
// Reduced core: size/burst sunk on inputs, driven full-width INCR on outputs.
// ============================================================================

module VX_mem_axi_xbar #(
    parameter NUM_INPUTS  = 2,
    parameter NUM_OUTPUTS = 1,
    parameter ADDR_WIDTH  = 64,
    parameter DATA_WIDTH  = 512,
    parameter ID_WIDTH    = 32,
    parameter `STRING ARBITER = "R",
    parameter STICKY      = 0,
    parameter MULTI_OUT   = 0,
    parameter OUT_REG     = 0,
    parameter STRB_WIDTH  = DATA_WIDTH/8,
    parameter SEL_WIDTH   = `LOG2UP(NUM_OUTPUTS)
) (
    input  wire          clk,
    input  wire          reset,
    VX_mem_axi_if.slave  s [NUM_INPUTS],
    VX_mem_axi_if.master m [NUM_OUTPUTS]
);
    localparam AXSIZE = `CLOG2(STRB_WIDTH);

    wire [NUM_INPUTS-1:0]                  s_awvalid, s_awready;
    wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_awaddr;
    wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_awid;
    wire [NUM_INPUTS-1:0][7:0]             s_awlen;
    wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]   s_awsel;
    wire [NUM_INPUTS-1:0]                  s_wvalid, s_wready, s_wlast;
    wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_wdata;
    wire [NUM_INPUTS-1:0][STRB_WIDTH-1:0]  s_wstrb;
    wire [NUM_INPUTS-1:0]                  s_bvalid, s_bready;
    wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_bid;
    wire [NUM_INPUTS-1:0][1:0]             s_bresp;
    wire [NUM_INPUTS-1:0]                  s_arvalid, s_arready;
    wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_araddr;
    wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_arid;
    wire [NUM_INPUTS-1:0][7:0]             s_arlen;
    wire [NUM_INPUTS-1:0][SEL_WIDTH-1:0]   s_arsel;
    wire [NUM_INPUTS-1:0]                  s_rvalid, s_rready, s_rlast;
    wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_rdata;
    wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_rid;
    wire [NUM_INPUTS-1:0][1:0]             s_rresp;

    wire [NUM_OUTPUTS-1:0]                 m_awvalid, m_awready;
    wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_awaddr;
    wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_awid;
    wire [NUM_OUTPUTS-1:0][7:0]            m_awlen;
    wire [NUM_OUTPUTS-1:0]                 m_wvalid, m_wready, m_wlast;
    wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_wdata;
    wire [NUM_OUTPUTS-1:0][STRB_WIDTH-1:0] m_wstrb;
    wire [NUM_OUTPUTS-1:0]                 m_bvalid, m_bready;
    wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_bid;
    wire [NUM_OUTPUTS-1:0][1:0]            m_bresp;
    wire [NUM_OUTPUTS-1:0]                 m_arvalid, m_arready;
    wire [NUM_OUTPUTS-1:0][ADDR_WIDTH-1:0] m_araddr;
    wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_arid;
    wire [NUM_OUTPUTS-1:0][7:0]            m_arlen;
    wire [NUM_OUTPUTS-1:0]                 m_rvalid, m_rready, m_rlast;
    wire [NUM_OUTPUTS-1:0][DATA_WIDTH-1:0] m_rdata;
    wire [NUM_OUTPUTS-1:0][ID_WIDTH-1:0]   m_rid;
    wire [NUM_OUTPUTS-1:0][1:0]            m_rresp;

    // ---- Unpack upstream masters; decode target slave from high address bits ----
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_in
        assign s_awvalid[i]  = s[i].awvalid;
        assign s[i].awready  = s_awready[i];
        assign s_awaddr[i]   = s[i].awaddr;
        assign s_awid[i]     = s[i].awid;
        assign s_awlen[i]    = s[i].awlen;
        assign s_wvalid[i]   = s[i].wvalid;
        assign s[i].wready   = s_wready[i];
        assign s_wdata[i]    = s[i].wdata;
        assign s_wstrb[i]    = s[i].wstrb;
        assign s_wlast[i]    = s[i].wlast;
        assign s[i].bvalid   = s_bvalid[i];
        assign s_bready[i]   = s[i].bready;
        assign s[i].bid      = s_bid[i];
        assign s[i].bresp    = s_bresp[i];
        assign s_arvalid[i]  = s[i].arvalid;
        assign s[i].arready  = s_arready[i];
        assign s_araddr[i]   = s[i].araddr;
        assign s_arid[i]     = s[i].arid;
        assign s_arlen[i]    = s[i].arlen;
        assign s[i].rvalid   = s_rvalid[i];
        assign s_rready[i]   = s[i].rready;
        assign s[i].rdata    = s_rdata[i];
        assign s[i].rlast    = s_rlast[i];
        assign s[i].rid      = s_rid[i];
        assign s[i].rresp    = s_rresp[i];
        if (NUM_OUTPUTS > 1) begin : g_sel
            assign s_awsel[i] = s[i].awaddr[ADDR_WIDTH-1 -: SEL_WIDTH];
            assign s_arsel[i] = s[i].araddr[ADDR_WIDTH-1 -: SEL_WIDTH];
        end else begin : g_sel0
            assign s_awsel[i] = '0;
            assign s_arsel[i] = '0;
        end
        `UNUSED_VAR (s[i].awsize)
        `UNUSED_VAR (s[i].awburst)
        `UNUSED_VAR (s[i].arsize)
        `UNUSED_VAR (s[i].arburst)
    end

    // ---- Pack downstream slaves ----
    for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin : g_out
        assign m[j].awvalid  = m_awvalid[j];
        assign m_awready[j]  = m[j].awready;
        assign m[j].awaddr   = m_awaddr[j];
        assign m[j].awid     = m_awid[j];
        assign m[j].awlen    = m_awlen[j];
        assign m[j].awsize   = 3'(AXSIZE);
        assign m[j].awburst  = 2'b01;
        assign m[j].wvalid   = m_wvalid[j];
        assign m_wready[j]   = m[j].wready;
        assign m[j].wdata    = m_wdata[j];
        assign m[j].wstrb    = m_wstrb[j];
        assign m[j].wlast    = m_wlast[j];
        assign m_bvalid[j]   = m[j].bvalid;
        assign m[j].bready   = m_bready[j];
        assign m_bid[j]      = m[j].bid;
        assign m_bresp[j]    = m[j].bresp;
        assign m[j].arvalid  = m_arvalid[j];
        assign m_arready[j]  = m[j].arready;
        assign m[j].araddr   = m_araddr[j];
        assign m[j].arid     = m_arid[j];
        assign m[j].arlen    = m_arlen[j];
        assign m[j].arsize   = 3'(AXSIZE);
        assign m[j].arburst  = 2'b01;
        assign m_rvalid[j]   = m[j].rvalid;
        assign m[j].rready   = m_rready[j];
        assign m_rdata[j]    = m[j].rdata;
        assign m_rlast[j]    = m[j].rlast;
        assign m_rid[j]      = m[j].rid;
        assign m_rresp[j]    = m[j].rresp;
    end

    VX_mm_axi_xbar #(
        .NUM_INPUTS  (NUM_INPUTS),
        .NUM_OUTPUTS (NUM_OUTPUTS),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .ID_WIDTH    (ID_WIDTH),
        .ARBITER     (ARBITER),
        .STICKY      (STICKY),
        .MULTI_OUT   (MULTI_OUT),
        .OUT_REG     (OUT_REG),
        .STRB_WIDTH  (STRB_WIDTH)
    ) impl (
        .clk (clk), .reset (reset),
        .s_awvalid (s_awvalid), .s_awready (s_awready),
        .s_awaddr  (s_awaddr),  .s_awid (s_awid), .s_awlen (s_awlen), .s_awsel (s_awsel),
        .s_wvalid  (s_wvalid),  .s_wready (s_wready),
        .s_wdata   (s_wdata),   .s_wstrb (s_wstrb), .s_wlast (s_wlast),
        .s_bvalid  (s_bvalid),  .s_bready (s_bready),
        .s_bid     (s_bid),     .s_bresp (s_bresp),
        .s_arvalid (s_arvalid), .s_arready (s_arready),
        .s_araddr  (s_araddr),  .s_arid (s_arid), .s_arlen (s_arlen), .s_arsel (s_arsel),
        .s_rvalid  (s_rvalid),  .s_rready (s_rready),
        .s_rdata   (s_rdata),   .s_rlast (s_rlast),
        .s_rid     (s_rid),     .s_rresp (s_rresp),
        .m_awvalid (m_awvalid), .m_awready (m_awready),
        .m_awaddr  (m_awaddr),  .m_awid (m_awid), .m_awlen (m_awlen),
        .m_wvalid  (m_wvalid),  .m_wready (m_wready),
        .m_wdata   (m_wdata),   .m_wstrb (m_wstrb), .m_wlast (m_wlast),
        .m_bvalid  (m_bvalid),  .m_bready (m_bready),
        .m_bid     (m_bid),     .m_bresp (m_bresp),
        .m_arvalid (m_arvalid), .m_arready (m_arready),
        .m_araddr  (m_araddr),  .m_arid (m_arid), .m_arlen (m_arlen),
        .m_rvalid  (m_rvalid),  .m_rready (m_rready),
        .m_rdata   (m_rdata),   .m_rlast (m_rlast),
        .m_rid     (m_rid),     .m_rresp (m_rresp),
        `UNUSED_PIN (collisions)
    );

endmodule
