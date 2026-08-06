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
// VX_mem_axi_slice — AXI4 register slice on the VX_mem_axi_if bundle.
//
// Interface convenience wrapper over the flat-port libs core VX_mm_axi_slice:
// inserts a registered stage on every AXI channel (OUT_REG) for an SLR-safe /
// timing boundary. The reduced core carries no size/burst sideband — those are
// static (INCR / full-width) and pass through combinationally.
// ============================================================================

module VX_mem_axi_slice #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,
    parameter ID_WIDTH   = 32,
    parameter OUT_REG    = 1
) (
    input  wire          clk,
    input  wire          reset,
    VX_mem_axi_if.slave  s,   // upstream master
    VX_mem_axi_if.master m    // downstream slave
);
    VX_mm_axi_slice #(
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH),
        .ID_WIDTH   (ID_WIDTH),
        .OUT_REG    (OUT_REG)
    ) impl (
        .clk (clk), .reset (reset),
        .s_awvalid (s.awvalid), .s_awready (s.awready),
        .s_awaddr  (s.awaddr),  .s_awid (s.awid), .s_awlen (s.awlen),
        .s_wvalid  (s.wvalid),  .s_wready (s.wready),
        .s_wdata   (s.wdata),   .s_wstrb (s.wstrb), .s_wlast (s.wlast),
        .s_bvalid  (s.bvalid),  .s_bready (s.bready),
        .s_bid     (s.bid),     .s_bresp (s.bresp),
        .s_arvalid (s.arvalid), .s_arready (s.arready),
        .s_araddr  (s.araddr),  .s_arid (s.arid), .s_arlen (s.arlen),
        .s_rvalid  (s.rvalid),  .s_rready (s.rready),
        .s_rdata   (s.rdata),   .s_rlast (s.rlast),
        .s_rid     (s.rid),     .s_rresp (s.rresp),
        .m_awvalid (m.awvalid), .m_awready (m.awready),
        .m_awaddr  (m.awaddr),  .m_awid (m.awid), .m_awlen (m.awlen),
        .m_wvalid  (m.wvalid),  .m_wready (m.wready),
        .m_wdata   (m.wdata),   .m_wstrb (m.wstrb), .m_wlast (m.wlast),
        .m_bvalid  (m.bvalid),  .m_bready (m.bready),
        .m_bid     (m.bid),     .m_bresp (m.bresp),
        .m_arvalid (m.arvalid), .m_arready (m.arready),
        .m_araddr  (m.araddr),  .m_arid (m.arid), .m_arlen (m.arlen),
        .m_rvalid  (m.rvalid),  .m_rready (m.rready),
        .m_rdata   (m.rdata),   .m_rlast (m.rlast),
        .m_rid     (m.rid),     .m_rresp (m.rresp)
    );

    // Static AXI sideband passes through the reduced-view core.
    assign m.awsize  = s.awsize;
    assign m.awburst = s.awburst;
    assign m.arsize  = s.arsize;
    assign m.arburst = s.arburst;

endmodule
