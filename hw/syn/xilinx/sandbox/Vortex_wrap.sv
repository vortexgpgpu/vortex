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

module Vortex_wrap #(
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

	wire                                 m_axi_mem_awvalid_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_awready_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_awaddr_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_awid_a [C_M_AXI_MEM_NUM_BANKS];
	wire [7:0]                           m_axi_mem_awlen_a [C_M_AXI_MEM_NUM_BANKS];
	wire [2:0]                           m_axi_mem_awsize_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_awburst_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_awlock_a [C_M_AXI_MEM_NUM_BANKS];
	wire [3:0]                           m_axi_mem_awcache_a [C_M_AXI_MEM_NUM_BANKS];
	wire [2:0]                           m_axi_mem_awprot_a [C_M_AXI_MEM_NUM_BANKS];
	wire [3:0]                           m_axi_mem_awqos_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_wvalid_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_wready_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_mem_wdata_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_mem_wstrb_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_wlast_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_arvalid_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_arready_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_mem_araddr_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_mem_arid_a [C_M_AXI_MEM_NUM_BANKS];
	wire [7:0]                           m_axi_mem_arlen_a [C_M_AXI_MEM_NUM_BANKS];
	wire [2:0]                           m_axi_mem_arsize_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_arburst_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_arlock_a [C_M_AXI_MEM_NUM_BANKS];
	wire [3:0]                           m_axi_mem_arcache_a [C_M_AXI_MEM_NUM_BANKS];
	wire [2:0]                           m_axi_mem_arprot_a [C_M_AXI_MEM_NUM_BANKS];
	wire [3:0]                           m_axi_mem_arqos_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_rvalid_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_rready_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_mem_rdata_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_rlast_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_mem_rid_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_rresp_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_bvalid_a [C_M_AXI_MEM_NUM_BANKS];
	wire                                 m_axi_mem_bready_a [C_M_AXI_MEM_NUM_BANKS];
	wire [1:0]                           m_axi_mem_bresp_a [C_M_AXI_MEM_NUM_BANKS];
	wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]	 m_axi_mem_bid_a [C_M_AXI_MEM_NUM_BANKS];

	assign m_axi_mem_awvalid = m_axi_mem_awvalid_a[0];
	assign m_axi_mem_awready_a[0] = m_axi_mem_awready;
	assign m_axi_mem_awaddr = m_axi_mem_awaddr_a[0];
	assign m_axi_mem_awid = m_axi_mem_awid_a[0];
	assign m_axi_mem_awlen = m_axi_mem_awlen_a[0];
	assign m_axi_mem_awsize = m_axi_mem_awsize_a[0];
	assign m_axi_mem_awburst = m_axi_mem_awburst_a[0];
	assign m_axi_mem_awlock = m_axi_mem_awlock_a[0];
	assign m_axi_mem_awcache = m_axi_mem_awcache_a[0];
	assign m_axi_mem_awprot = m_axi_mem_awprot_a[0];
	assign m_axi_mem_awqos = m_axi_mem_awqos_a[0];

	assign m_axi_mem_wvalid = m_axi_mem_wvalid_a[0];
	assign m_axi_mem_wready_a[0] = m_axi_mem_wready;
	assign m_axi_mem_wdata = m_axi_mem_wdata_a[0];
	assign m_axi_mem_wstrb = m_axi_mem_wstrb_a[0];
	assign m_axi_mem_wlast = m_axi_mem_wlast_a[0];

	assign m_axi_mem_arvalid = m_axi_mem_arvalid_a[0];
	assign m_axi_mem_arready_a[0] = m_axi_mem_arready;
	assign m_axi_mem_araddr = m_axi_mem_araddr_a[0];
	assign m_axi_mem_arid = m_axi_mem_arid_a[0];
	assign m_axi_mem_arlen = m_axi_mem_arlen_a[0];
	assign m_axi_mem_arsize = m_axi_mem_arsize_a[0];
	assign m_axi_mem_arburst = m_axi_mem_arburst_a[0];
	assign m_axi_mem_arlock = m_axi_mem_arlock_a[0];
	assign m_axi_mem_arcache = m_axi_mem_arcache_a[0];
	assign m_axi_mem_arprot = m_axi_mem_arprot_a[0];
	assign m_axi_mem_arqos = m_axi_mem_arqos_a[0];

	assign m_axi_mem_rvalid_a[0] = m_axi_mem_rvalid;
	assign m_axi_mem_rready = m_axi_mem_rready_a[0];
	assign m_axi_mem_rdata_a[0] = m_axi_mem_rdata;
	assign m_axi_mem_rlast_a[0] = m_axi_mem_rlast;
	assign m_axi_mem_rid_a[0] = m_axi_mem_rid;
	assign m_axi_mem_rresp_a[0] = m_axi_mem_rresp;

	assign m_axi_mem_bvalid_a[0] = m_axi_mem_bvalid;
	assign m_axi_mem_bready = m_axi_mem_bready_a[0];
	assign m_axi_mem_bresp_a[0] = m_axi_mem_bresp;
	assign m_axi_mem_bid_a[0] = m_axi_mem_bid;

	Vortex_axi #(
		.AXI_DATA_WIDTH (C_M_AXI_GMEM_DATA_WIDTH),
		.AXI_ADDR_WIDTH (C_M_AXI_GMEM_ADDR_WIDTH),
		.AXI_TID_WIDTH  (C_M_AXI_GMEM_ID_WIDTH)
	) inst (
		.clk			(clk),
		.reset			(reset),

		.m_axi_awvalid	(m_axi_mem_awvalid_a),
		.m_axi_awready	(m_axi_mem_awready_a),
		.m_axi_awaddr	(m_axi_mem_awaddr_a),
		.m_axi_awid		(m_axi_mem_awid_a),
		.m_axi_awlen	(m_axi_mem_awlen_a),
		.m_axi_awsize	(m_axi_mem_awsize_a),
		.m_axi_awburst	(m_axi_mem_awburst_a),
		.m_axi_awlock	(m_axi_mem_awlock_a),
		.m_axi_awcache	(m_axi_mem_awcache_a),
		.m_axi_awprot	(m_axi_mem_awprot_a),
		.m_axi_awqos	(m_axi_mem_awqos_a),

		.m_axi_wvalid	(m_axi_mem_wvalid_a),
		.m_axi_wready	(m_axi_mem_wready_a),
		.m_axi_wdata	(m_axi_mem_wdata_a),
		.m_axi_wstrb	(m_axi_mem_wstrb_a),
		.m_axi_wlast	(m_axi_mem_wlast_a),

		.m_axi_bvalid	(m_axi_mem_bvalid_a),
		.m_axi_bready	(m_axi_mem_bready_a),
		.m_axi_bid		(m_axi_mem_bid_a),
		.m_axi_bresp	(m_axi_mem_bresp_a),

		.m_axi_arvalid	(m_axi_mem_arvalid_a),
		.m_axi_arready	(m_axi_mem_arready_a),
		.m_axi_araddr	(m_axi_mem_araddr_a),
		.m_axi_arid		(m_axi_mem_arid_a),
		.m_axi_arlen	(m_axi_mem_arlen_a),
		.m_axi_arsize	(m_axi_mem_arsize_a),
		.m_axi_arburst	(m_axi_mem_arburst_a),
		.m_axi_arlock	(m_axi_mem_arlock_a),
		.m_axi_arcache	(m_axi_mem_arcache_a),
		.m_axi_arprot	(m_axi_mem_arprot_a),
		.m_axi_arqos	(m_axi_mem_arqos_a),

		.m_axi_rvalid	(m_axi_mem_rvalid_a),
		.m_axi_rready	(m_axi_mem_rready_a),
		.m_axi_rdata	(m_axi_mem_rdata_a),
		.m_axi_rid    	(m_axi_mem_rid_a),
		.m_axi_rresp	(m_axi_mem_rresp_a),
		.m_axi_rlast	(m_axi_mem_rlast_a),

		.dcr_wr_valid	(dcr_wr_valid),
		.dcr_wr_addr	(dcr_wr_addr),
		.dcr_wr_data	(dcr_wr_data),

		.busy           (busy)
	);

endmodule
