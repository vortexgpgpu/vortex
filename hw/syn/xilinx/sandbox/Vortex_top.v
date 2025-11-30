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

module Vortex_top #(
	parameter C_M_AXI_GMEM_DATA_WIDTH = 512,
	parameter C_M_AXI_GMEM_ADDR_WIDTH = `XLEN,
	parameter C_M_AXI_GMEM_ID_WIDTH   = 32,
	parameter C_M_AXI_MEM_NUM_BANKS   = 1
) (
	input wire                                  clk,
	input wire                                  reset,

	// AXI4 memory interface
	output wire                                 m_axi_mem_awvalid,
	input  wire                                 m_axi_mem_awready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_awaddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_awid,
	output wire [7:0]                           m_axi_mem_awlen,
	output wire [2:0]                           m_axi_mem_awsize,
	output wire [1:0]                           m_axi_mem_awburst,
	output wire [1:0]                           m_axi_mem_awlock,
	output wire [3:0]                           m_axi_mem_awcache,
	output wire [2:0]                           m_axi_mem_awprot,
	output wire [3:0]                           m_axi_mem_awqos,
	output wire                                 m_axi_mem_wvalid,
	input  wire                                 m_axi_mem_wready,
	output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_mem_wdata,
	output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_mem_wstrb,
	output wire                                 m_axi_mem_wlast,
	output wire                                 m_axi_mem_arvalid,
	input  wire                                 m_axi_mem_arready,
	output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_araddr,
	output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_mem_arid,
	output wire [7:0]                           m_axi_mem_arlen,
	output wire [2:0]                           m_axi_mem_arsize,
	output wire [1:0]                           m_axi_mem_arburst,
	output wire [1:0]                           m_axi_mem_arlock,
	output wire [3:0]                           m_axi_mem_arcache,
	output wire [2:0]                           m_axi_mem_arprot,
	output wire [3:0]                           m_axi_mem_arqos,
	input  wire                                 m_axi_mem_rvalid,
	output wire                                 m_axi_mem_rready,
	input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_mem_rdata,
	input  wire                                 m_axi_mem_rlast,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_rid,
	input  wire [1:0]                           m_axi_mem_rresp,
	input  wire                                 m_axi_mem_bvalid,
	output wire                                 m_axi_mem_bready,
	input  wire [1:0]                           m_axi_mem_bresp,
	input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]	m_axi_mem_bid,

	input  wire                         		dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0]		dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0] 		dcr_wr_data,

	output wire                                 busy
);

	Vortex_wrap #(
		.C_M_AXI_GMEM_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH),
		.C_M_AXI_GMEM_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
		.C_M_AXI_GMEM_ID_WIDTH(C_M_AXI_GMEM_ID_WIDTH),
		.C_M_AXI_MEM_NUM_BANKS(C_M_AXI_MEM_NUM_BANKS)
	) wrapper (
		.clk(clk),
		.reset(reset),
		.m_axi_mem_awvalid(m_axi_mem_awvalid),
		.m_axi_mem_awready(m_axi_mem_awready),
		.m_axi_mem_awaddr(m_axi_mem_awaddr),
		.m_axi_mem_awid(m_axi_mem_awid),
		.m_axi_mem_awlen(m_axi_mem_awlen),
		.m_axi_mem_awsize(m_axi_mem_awsize),
		.m_axi_mem_awburst(m_axi_mem_awburst),
		.m_axi_mem_awlock(m_axi_mem_awlock),
		.m_axi_mem_awcache(m_axi_mem_awcache),
		.m_axi_mem_awprot(m_axi_mem_awprot),
		.m_axi_mem_awqos(m_axi_mem_awqos),
		.m_axi_mem_wvalid(m_axi_mem_wvalid),
		.m_axi_mem_wready(m_axi_mem_wready),
		.m_axi_mem_wdata(m_axi_mem_wdata),
		.m_axi_mem_wstrb(m_axi_mem_wstrb),
		.m_axi_mem_wlast(m_axi_mem_wlast),
		.m_axi_mem_arvalid(m_axi_mem_arvalid),
		.m_axi_mem_arready(m_axi_mem_arready),
		.m_axi_mem_araddr(m_axi_mem_araddr),
		.m_axi_mem_arid(m_axi_mem_arid),
		.m_axi_mem_arlen(m_axi_mem_arlen),
		.m_axi_mem_arsize(m_axi_mem_arsize),
		.m_axi_mem_arburst(m_axi_mem_arburst),
		.m_axi_mem_arlock(m_axi_mem_arlock),
		.m_axi_mem_arcache(m_axi_mem_arcache),
		.m_axi_mem_arprot(m_axi_mem_arprot),
		.m_axi_mem_arqos(m_axi_mem_arqos),
		.m_axi_mem_rvalid(m_axi_mem_rvalid),
		.m_axi_mem_rready(m_axi_mem_rready),
		.m_axi_mem_rdata(m_axi_mem_rdata),
		.m_axi_mem_rlast(m_axi_mem_rlast),
		.m_axi_mem_rid(m_axi_mem_rid),
		.m_axi_mem_rresp(m_axi_mem_rresp),
		.m_axi_mem_bvalid(m_axi_mem_bvalid),
		.m_axi_mem_bready(m_axi_mem_bready),
		.m_axi_mem_bresp(m_axi_mem_bresp),
		.m_axi_mem_bid(m_axi_mem_bid),
		.dcr_wr_valid(dcr_wr_valid),
		.dcr_wr_addr(dcr_wr_addr),
		.dcr_wr_data(dcr_wr_data),
		.busy(busy)
	);

endmodule
