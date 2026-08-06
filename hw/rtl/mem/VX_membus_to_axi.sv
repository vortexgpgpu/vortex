// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_membus_to_axi — VX_mem_bus_if / VX_mem_axi_if interface wrapper over the
// flat libs core VX_mem_to_axi: adapts N Vortex request/response memory ports
// to M AXI4 masters. The inverse of VX_membus_from_axi. The reduced VX_mem_axi_if
// carries every meaningful AXI signal; the flat core's constant sideband
// (lock/cache/prot/qos/region) is dropped here and re-added at the pin boundary.
// ============================================================================

module VX_membus_to_axi
  import VX_gpu_pkg::*;
#(
    parameter DATA_WIDTH     = 512,
    parameter ADDR_WIDTH_IN  = 26,
    parameter ADDR_WIDTH_OUT = 32,
    parameter TAG_WIDTH_IN   = 8,
    parameter TAG_WIDTH_OUT  = 8,
    parameter NUM_PORTS_IN   = 1,
    parameter NUM_BANKS_OUT  = 1,
    parameter INTERLEAVE     = 0,
    parameter TAG_BUFFER_SIZE = 16,
    parameter ARBITER        = "R",
    parameter REQ_OUT_BUF    = 0,
    parameter RSP_OUT_BUF    = 0,
    parameter DATA_SIZE      = DATA_WIDTH/8
) (
    input  wire          clk,
    input  wire          reset,
    VX_mem_bus_if.slave  bus_in_if [NUM_PORTS_IN],
    VX_mem_axi_if.master m_axi     [NUM_BANKS_OUT]
);
    // ---- Vortex membus (flat) ----
    wire                     mem_req_valid  [NUM_PORTS_IN];
    wire                     mem_req_rw     [NUM_PORTS_IN];
    wire [DATA_SIZE-1:0]     mem_req_byteen [NUM_PORTS_IN];
    wire [ADDR_WIDTH_IN-1:0] mem_req_addr   [NUM_PORTS_IN];
    wire [DATA_WIDTH-1:0]    mem_req_data   [NUM_PORTS_IN];
    wire [TAG_WIDTH_IN-1:0]  mem_req_tag    [NUM_PORTS_IN];
    wire                     mem_req_ready  [NUM_PORTS_IN];
    wire                     mem_rsp_valid  [NUM_PORTS_IN];
    wire [DATA_WIDTH-1:0]    mem_rsp_data   [NUM_PORTS_IN];
    wire [TAG_WIDTH_IN-1:0]  mem_rsp_tag    [NUM_PORTS_IN];
    wire                     mem_rsp_ready  [NUM_PORTS_IN];

    for (genvar i = 0; i < NUM_PORTS_IN; ++i) begin : g_in
        assign mem_req_valid[i]      = bus_in_if[i].req_valid;
        assign mem_req_rw[i]         = bus_in_if[i].req_data.rw;
        assign mem_req_byteen[i]     = bus_in_if[i].req_data.byteen;
        assign mem_req_addr[i]       = bus_in_if[i].req_data.addr;
        assign mem_req_data[i]       = bus_in_if[i].req_data.data;
        assign mem_req_tag[i]        = bus_in_if[i].req_data.tag;
        assign bus_in_if[i].req_ready = mem_req_ready[i];
        assign bus_in_if[i].rsp_valid = mem_rsp_valid[i];
        assign bus_in_if[i].rsp_data.data = mem_rsp_data[i];
        assign bus_in_if[i].rsp_data.tag  = mem_rsp_tag[i];
        assign mem_rsp_ready[i]      = bus_in_if[i].rsp_ready;
    end

    // ---- AXI master (flat, from the core) ----
    wire                      m_awvalid [NUM_BANKS_OUT], m_awready [NUM_BANKS_OUT];
    wire [ADDR_WIDTH_OUT-1:0] m_awaddr  [NUM_BANKS_OUT];
    wire [TAG_WIDTH_OUT-1:0]  m_awid    [NUM_BANKS_OUT];
    wire [7:0]                m_awlen   [NUM_BANKS_OUT];
    wire [2:0]                m_awsize  [NUM_BANKS_OUT];
    wire [1:0]                m_awburst [NUM_BANKS_OUT];
    wire                      m_wvalid  [NUM_BANKS_OUT], m_wready [NUM_BANKS_OUT];
    wire [DATA_WIDTH-1:0]     m_wdata   [NUM_BANKS_OUT];
    wire [DATA_SIZE-1:0]      m_wstrb   [NUM_BANKS_OUT];
    wire                      m_wlast   [NUM_BANKS_OUT];
    wire                      m_bvalid  [NUM_BANKS_OUT], m_bready [NUM_BANKS_OUT];
    wire [TAG_WIDTH_OUT-1:0]  m_bid     [NUM_BANKS_OUT];
    wire [1:0]                m_bresp   [NUM_BANKS_OUT];
    wire                      m_arvalid [NUM_BANKS_OUT], m_arready [NUM_BANKS_OUT];
    wire [ADDR_WIDTH_OUT-1:0] m_araddr  [NUM_BANKS_OUT];
    wire [TAG_WIDTH_OUT-1:0]  m_arid    [NUM_BANKS_OUT];
    wire [7:0]                m_arlen   [NUM_BANKS_OUT];
    wire [2:0]                m_arsize  [NUM_BANKS_OUT];
    wire [1:0]                m_arburst [NUM_BANKS_OUT];
    wire                      m_rvalid  [NUM_BANKS_OUT], m_rready [NUM_BANKS_OUT];
    wire [DATA_WIDTH-1:0]     m_rdata   [NUM_BANKS_OUT];
    wire                      m_rlast   [NUM_BANKS_OUT];
    wire [TAG_WIDTH_OUT-1:0]  m_rid     [NUM_BANKS_OUT];
    wire [1:0]                m_rresp   [NUM_BANKS_OUT];

    for (genvar j = 0; j < NUM_BANKS_OUT; ++j) begin : g_out
        assign m_axi[j].awvalid = m_awvalid[j]; assign m_awready[j] = m_axi[j].awready;
        assign m_axi[j].awaddr  = m_awaddr[j];  assign m_axi[j].awid = m_awid[j]; assign m_axi[j].awlen = m_awlen[j];
        assign m_axi[j].awsize  = m_awsize[j];  assign m_axi[j].awburst = m_awburst[j];
        assign m_axi[j].wvalid  = m_wvalid[j];  assign m_wready[j] = m_axi[j].wready;
        assign m_axi[j].wdata   = m_wdata[j];   assign m_axi[j].wstrb = m_wstrb[j]; assign m_axi[j].wlast = m_wlast[j];
        assign m_bvalid[j]      = m_axi[j].bvalid; assign m_axi[j].bready = m_bready[j];
        assign m_bid[j]         = m_axi[j].bid; assign m_bresp[j] = m_axi[j].bresp;
        assign m_axi[j].arvalid = m_arvalid[j]; assign m_arready[j] = m_axi[j].arready;
        assign m_axi[j].araddr  = m_araddr[j];  assign m_axi[j].arid = m_arid[j]; assign m_axi[j].arlen = m_arlen[j];
        assign m_axi[j].arsize  = m_arsize[j];  assign m_axi[j].arburst = m_arburst[j];
        assign m_rvalid[j]      = m_axi[j].rvalid; assign m_axi[j].rready = m_rready[j];
        assign m_rdata[j]       = m_axi[j].rdata; assign m_rlast[j] = m_axi[j].rlast;
        assign m_rid[j]         = m_axi[j].rid; assign m_rresp[j] = m_axi[j].rresp;
    end

    // Constant AXI sideband from the core is not carried by VX_mem_axi_if.
    wire [1:0] u_awlock [NUM_BANKS_OUT]; wire [3:0] u_awcache [NUM_BANKS_OUT];
    wire [2:0] u_awprot [NUM_BANKS_OUT]; wire [3:0] u_awqos [NUM_BANKS_OUT];
    wire [3:0] u_awregion [NUM_BANKS_OUT];
    wire [1:0] u_arlock [NUM_BANKS_OUT]; wire [3:0] u_arcache [NUM_BANKS_OUT];
    wire [2:0] u_arprot [NUM_BANKS_OUT]; wire [3:0] u_arqos [NUM_BANKS_OUT];
    wire [3:0] u_arregion [NUM_BANKS_OUT];
    for (genvar j = 0; j < NUM_BANKS_OUT; ++j) begin : g_side
        `UNUSED_VAR (u_awlock[j])   `UNUSED_VAR (u_awcache[j])  `UNUSED_VAR (u_awprot[j])
        `UNUSED_VAR (u_awqos[j])    `UNUSED_VAR (u_awregion[j])
        `UNUSED_VAR (u_arlock[j])   `UNUSED_VAR (u_arcache[j])  `UNUSED_VAR (u_arprot[j])
        `UNUSED_VAR (u_arqos[j])    `UNUSED_VAR (u_arregion[j])
    end

    VX_mem_to_axi #(
        .DATA_WIDTH      (DATA_WIDTH),
        .ADDR_WIDTH_IN   (ADDR_WIDTH_IN),
        .ADDR_WIDTH_OUT  (ADDR_WIDTH_OUT),
        .TAG_WIDTH_IN    (TAG_WIDTH_IN),
        .TAG_WIDTH_OUT   (TAG_WIDTH_OUT),
        .NUM_PORTS_IN    (NUM_PORTS_IN),
        .NUM_BANKS_OUT   (NUM_BANKS_OUT),
        .INTERLEAVE      (INTERLEAVE),
        .TAG_BUFFER_SIZE (TAG_BUFFER_SIZE),
        .ARBITER         (ARBITER),
        .REQ_OUT_BUF     (REQ_OUT_BUF),
        .RSP_OUT_BUF     (RSP_OUT_BUF)
    ) impl (
        .clk (clk), .reset (reset),
        .mem_req_valid (mem_req_valid), .mem_req_rw (mem_req_rw), .mem_req_byteen (mem_req_byteen),
        .mem_req_addr (mem_req_addr), .mem_req_data (mem_req_data), .mem_req_tag (mem_req_tag),
        .mem_req_ready (mem_req_ready),
        .mem_rsp_valid (mem_rsp_valid), .mem_rsp_data (mem_rsp_data), .mem_rsp_tag (mem_rsp_tag),
        .mem_rsp_ready (mem_rsp_ready),
        .m_axi_awvalid (m_awvalid), .m_axi_awready (m_awready), .m_axi_awaddr (m_awaddr),
        .m_axi_awid (m_awid), .m_axi_awlen (m_awlen), .m_axi_awsize (m_awsize), .m_axi_awburst (m_awburst),
        .m_axi_awlock (u_awlock), .m_axi_awcache (u_awcache), .m_axi_awprot (u_awprot),
        .m_axi_awqos (u_awqos), .m_axi_awregion (u_awregion),
        .m_axi_wvalid (m_wvalid), .m_axi_wready (m_wready), .m_axi_wdata (m_wdata),
        .m_axi_wstrb (m_wstrb), .m_axi_wlast (m_wlast),
        .m_axi_bvalid (m_bvalid), .m_axi_bready (m_bready), .m_axi_bid (m_bid), .m_axi_bresp (m_bresp),
        .m_axi_arvalid (m_arvalid), .m_axi_arready (m_arready), .m_axi_araddr (m_araddr),
        .m_axi_arid (m_arid), .m_axi_arlen (m_arlen), .m_axi_arsize (m_arsize), .m_axi_arburst (m_arburst),
        .m_axi_arlock (u_arlock), .m_axi_arcache (u_arcache), .m_axi_arprot (u_arprot),
        .m_axi_arqos (u_arqos), .m_axi_arregion (u_arregion),
        .m_axi_rvalid (m_rvalid), .m_axi_rready (m_rready), .m_axi_rdata (m_rdata),
        .m_axi_rlast (m_rlast), .m_axi_rid (m_rid), .m_axi_rresp (m_rresp)
    );

endmodule
