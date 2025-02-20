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

`ifndef VORTEX_AFU_VH
`define VORTEX_AFU_VH

`ifndef PLATFORM_MEMORY_OFFSET
`define PLATFORM_MEMORY_OFFSET 0
`endif

`ifndef PLATFORM_MEMORY_ID_WIDTH
`define PLATFORM_MEMORY_ID_WIDTH 32
`endif

`define GEN_AXI_MEM(i) \
	output wire                                 m_axi_mem_``i``_awvalid, \
	input  wire                                 m_axi_mem_``i``_awready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0] 	m_axi_mem_``i``_awaddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH-1:0]   	m_axi_mem_``i``_awid, \
	output wire [7:0]                           m_axi_mem_``i``_awlen, \
	output wire                                 m_axi_mem_``i``_wvalid, \
	input  wire                                 m_axi_mem_``i``_wready, \
	output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]   	m_axi_mem_``i``_wdata, \
	output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0] 	m_axi_mem_``i``_wstrb, \
	output wire                                 m_axi_mem_``i``_wlast, \
	output wire                                 m_axi_mem_``i``_arvalid, \
	input  wire                                 m_axi_mem_``i``_arready, \
	output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]   	m_axi_mem_``i``_araddr, \
	output wire [C_M_AXI_MEM_ID_WIDTH-1:0]     	m_axi_mem_``i``_arid, \
	output wire [7:0]                           m_axi_mem_``i``_arlen, \
	input  wire                                 m_axi_mem_``i``_rvalid, \
	output wire                                 m_axi_mem_``i``_rready, \
	input  wire [C_M_AXI_MEM_DATA_WIDTH-1:0] 	m_axi_mem_``i``_rdata, \
	input  wire                                 m_axi_mem_``i``_rlast, \
	input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]   	m_axi_mem_``i``_rid, \
	input  wire [1:0]                           m_axi_mem_``i``_rresp, \
	input  wire                                 m_axi_mem_``i``_bvalid, \
	output wire                                 m_axi_mem_``i``_bready, \
	input  wire [1:0]                           m_axi_mem_``i``_bresp, \
	input  wire [C_M_AXI_MEM_ID_WIDTH-1:0]   	m_axi_mem_``i``_bid

`define AXI_MEM_ARGS(i) \
    .m_axi_mem_``i``_awvalid(m_axi_mem_``i``_awvalid), \
    .m_axi_mem_``i``_awready(m_axi_mem_``i``_awready), \
    .m_axi_mem_``i``_awaddr(m_axi_mem_``i``_awaddr), \
    .m_axi_mem_``i``_awid(m_axi_mem_``i``_awid), \
    .m_axi_mem_``i``_awlen(m_axi_mem_``i``_awlen), \
    .m_axi_mem_``i``_wvalid(m_axi_mem_``i``_wvalid), \
    .m_axi_mem_``i``_wready(m_axi_mem_``i``_wready), \
    .m_axi_mem_``i``_wdata(m_axi_mem_``i``_wdata), \
    .m_axi_mem_``i``_wstrb(m_axi_mem_``i``_wstrb), \
    .m_axi_mem_``i``_wlast(m_axi_mem_``i``_wlast), \
    .m_axi_mem_``i``_arvalid(m_axi_mem_``i``_arvalid), \
    .m_axi_mem_``i``_arready(m_axi_mem_``i``_arready), \
    .m_axi_mem_``i``_araddr(m_axi_mem_``i``_araddr), \
    .m_axi_mem_``i``_arid(m_axi_mem_``i``_arid), \
    .m_axi_mem_``i``_arlen(m_axi_mem_``i``_arlen), \
    .m_axi_mem_``i``_rvalid(m_axi_mem_``i``_rvalid), \
    .m_axi_mem_``i``_rready(m_axi_mem_``i``_rready), \
    .m_axi_mem_``i``_rdata(m_axi_mem_``i``_rdata), \
    .m_axi_mem_``i``_rlast(m_axi_mem_``i``_rlast), \
    .m_axi_mem_``i``_rid(m_axi_mem_``i``_rid), \
    .m_axi_mem_``i``_rresp(m_axi_mem_``i``_rresp), \
    .m_axi_mem_``i``_bvalid(m_axi_mem_``i``_bvalid), \
    .m_axi_mem_``i``_bready(m_axi_mem_``i``_bready), \
    .m_axi_mem_``i``_bresp(m_axi_mem_``i``_bresp), \
    .m_axi_mem_``i``_bid(m_axi_mem_``i``_bid)

`define AXI_MEM_TO_ARRAY(i) \
	assign m_axi_mem_``i``_awvalid = m_axi_mem_awvalid_a[i]; \
	assign m_axi_mem_awready_a[i] = m_axi_mem_``i``_awready; \
	assign m_axi_mem_``i``_awaddr = m_axi_mem_awaddr_a[i]; \
	assign m_axi_mem_``i``_awid = m_axi_mem_awid_a[i]; \
	assign m_axi_mem_``i``_awlen = m_axi_mem_awlen_a[i]; \
	assign m_axi_mem_``i``_wvalid = m_axi_mem_wvalid_a[i]; \
	assign m_axi_mem_wready_a[i] = m_axi_mem_``i``_wready; \
	assign m_axi_mem_``i``_wdata = m_axi_mem_wdata_a[i]; \
	assign m_axi_mem_``i``_wstrb = m_axi_mem_wstrb_a[i]; \
	assign m_axi_mem_``i``_wlast = m_axi_mem_wlast_a[i]; \
	assign m_axi_mem_``i``_arvalid = m_axi_mem_arvalid_a[i]; \
	assign m_axi_mem_arready_a[i] = m_axi_mem_``i``_arready; \
	assign m_axi_mem_``i``_araddr = m_axi_mem_araddr_a[i]; \
	assign m_axi_mem_``i``_arid = m_axi_mem_arid_a[i]; \
	assign m_axi_mem_``i``_arlen = m_axi_mem_arlen_a[i]; \
	assign m_axi_mem_rvalid_a[i] = m_axi_mem_``i``_rvalid; \
	assign m_axi_mem_``i``_rready = m_axi_mem_rready_a[i]; \
	assign m_axi_mem_rdata_a[i] = m_axi_mem_``i``_rdata; \
	assign m_axi_mem_rlast_a[i] = m_axi_mem_``i``_rlast; \
	assign m_axi_mem_rid_a[i] = m_axi_mem_``i``_rid; \
	assign m_axi_mem_rresp_a[i] = m_axi_mem_``i``_rresp; \
	assign m_axi_mem_bvalid_a[i] = m_axi_mem_``i``_bvalid; \
	assign m_axi_mem_``i``_bready = m_axi_mem_bready_a[i]; \
	assign m_axi_mem_bresp_a[i] = m_axi_mem_``i``_bresp; \
	assign m_axi_mem_bid_a[i] = m_axi_mem_``i``_bid

`include "VX_define.vh"

`endif // VORTEX_AFU_VH
