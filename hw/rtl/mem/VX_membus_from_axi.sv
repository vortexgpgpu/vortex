// Copyright © 2019-2023
// Licensed under the Apache License, Version 2.0.

`include "VX_define.vh"

// ============================================================================
// VX_membus_from_axi — VX_mem_axi_if interface wrapper over the flat libs core
// VX_mem_from_axi: bridges an AXI4 slave port to a Vortex request/response
// memory master. The inverse of VX_membus_to_axi.
// ============================================================================

module VX_membus_from_axi
  import VX_gpu_pkg::*;
#(
    parameter int ADDR_W   = 64,
    parameter int DATA_W   = 512,
    parameter int ID_W     = 6,
    parameter int MEM_ADDR_W = ADDR_W - $clog2(DATA_W/8)
)(
    input wire clk,
    input wire reset,

    VX_mem_axi_if.slave axi_s,

    // VX_mem_bus master-side signals (flattened — caller wires the fields).
    output wire                       mem_req_valid,
    output wire                       mem_req_rw,
    output wire [MEM_ADDR_W-1:0]      mem_req_addr,
    output wire [DATA_W-1:0]          mem_req_data,
    output wire [DATA_W/8-1:0]        mem_req_byteen,
    output wire [ID_W-1:0]            mem_req_tag,
    input  wire                       mem_req_ready,

    input  wire                       mem_rsp_valid,
    input  wire [DATA_W-1:0]          mem_rsp_data,
    input  wire [ID_W-1:0]            mem_rsp_tag,
    output wire                       mem_rsp_ready
);
    VX_mem_from_axi #(
        .ADDR_W     (ADDR_W),
        .DATA_W     (DATA_W),
        .ID_W       (ID_W),
        .MEM_ADDR_W (MEM_ADDR_W)
    ) impl (
        .clk (clk), .reset (reset),
        .s_awvalid (axi_s.awvalid), .s_awready (axi_s.awready),
        .s_awaddr  (axi_s.awaddr),  .s_awid (axi_s.awid), .s_awlen (axi_s.awlen),
        .s_wvalid  (axi_s.wvalid),  .s_wready (axi_s.wready),
        .s_wdata   (axi_s.wdata),   .s_wstrb (axi_s.wstrb), .s_wlast (axi_s.wlast),
        .s_bvalid  (axi_s.bvalid),  .s_bready (axi_s.bready),
        .s_bid     (axi_s.bid),     .s_bresp (axi_s.bresp),
        .s_arvalid (axi_s.arvalid), .s_arready (axi_s.arready),
        .s_araddr  (axi_s.araddr),  .s_arid (axi_s.arid), .s_arlen (axi_s.arlen),
        .s_rvalid  (axi_s.rvalid),  .s_rready (axi_s.rready),
        .s_rdata   (axi_s.rdata),   .s_rid (axi_s.rid), .s_rlast (axi_s.rlast), .s_rresp (axi_s.rresp),
        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),
        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready)
    );

    // Static AXI sideband is unused by the reduced core.
    `UNUSED_VAR (axi_s.awsize)
    `UNUSED_VAR (axi_s.awburst)
    `UNUSED_VAR (axi_s.arsize)
    `UNUSED_VAR (axi_s.arburst)

endmodule
