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

`include "vortex_afu.vh"

// Flattens Vortex_axi's port surface for instantiation as a Chisel BlackBox.
//
// Vortex_axi declares its AXI channels as SystemVerilog unpacked arrays indexed
// by bank; a BlackBox can only express packed, scalar ports, so each bank is
// broken out into discrete m_axi_mem_<i>_* signals. The port shape and the
// array conversion are the same ones the XRT AFU uses, taken from its macros so
// the two wrappers cannot drift apart.
//
// The control surface (DCR request/response, start, busy) is already scalar and
// passes straight through, so the simulator drives it directly rather than
// through an AXI4-Lite control block.
module VX_firesim_wrap import VX_gpu_pkg::*; #(
    parameter C_M_AXI_MEM_ID_WIDTH   = `PLATFORM_MEMORY_ID_WIDTH,
    parameter C_M_AXI_MEM_DATA_WIDTH = `VX_CFG_PLATFORM_MEMORY_DATA_SIZE * 8,
    parameter C_M_AXI_MEM_ADDR_WIDTH = 64,
    parameter C_M_AXI_MEM_NUM_BANKS  = 1
) (
    input  wire clk,
    input  wire reset,

    `MP_REPEAT (1, GEN_AXI_MEM, MP_COMMA),

    // DCR write request
    input  wire                          dcr_req_valid,
    input  wire                          dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0] dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0] dcr_req_data,

    // DCR read response
    output wire                          dcr_rsp_valid,
    output wire [VX_DCR_DATA_WIDTH-1:0] dcr_rsp_data,

    // ctrl/status
    input  wire                          start,
    output wire                          busy
);
    localparam VX_MEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH;

    wire                                 m_axi_mem_awvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_awready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [VX_MEM_ADDR_WIDTH-1:0]         m_axi_mem_awaddr_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_awid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                           m_axi_mem_awlen_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_wvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_wready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_wdata_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]  m_axi_mem_wstrb_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_wlast_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_bvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_bready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_bid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                           m_axi_mem_bresp_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_arvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_arready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [VX_MEM_ADDR_WIDTH-1:0]         m_axi_mem_araddr_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_arid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [7:0]                           m_axi_mem_arlen_a [C_M_AXI_MEM_NUM_BANKS];

    wire                                 m_axi_mem_rvalid_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_rready_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_rdata_a [C_M_AXI_MEM_NUM_BANKS];
    wire                                 m_axi_mem_rlast_a [C_M_AXI_MEM_NUM_BANKS];
    wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_rid_a [C_M_AXI_MEM_NUM_BANKS];
    wire [1:0]                           m_axi_mem_rresp_a [C_M_AXI_MEM_NUM_BANKS];

    `AXI_MEM_TO_ARRAY (0);

    Vortex_axi #(
        .AXI_DATA_WIDTH (C_M_AXI_MEM_DATA_WIDTH),
        .AXI_ADDR_WIDTH (VX_MEM_ADDR_WIDTH),
        .AXI_TID_WIDTH  (C_M_AXI_MEM_ID_WIDTH),
        .AXI_NUM_BANKS  (C_M_AXI_MEM_NUM_BANKS)
    ) vortex_axi (
        .clk            (clk),
        .reset          (reset),

        .m_axi_awvalid  (m_axi_mem_awvalid_a),
        .m_axi_awready  (m_axi_mem_awready_a),
        .m_axi_awaddr   (m_axi_mem_awaddr_a),
        .m_axi_awid     (m_axi_mem_awid_a),
        .m_axi_awlen    (m_axi_mem_awlen_a),
        .m_axi_awsize   (),
        .m_axi_awburst  (),
        .m_axi_awlock   (),
        .m_axi_awcache  (),
        .m_axi_awprot   (),
        .m_axi_awqos    (),
        .m_axi_awregion (),

        .m_axi_wvalid   (m_axi_mem_wvalid_a),
        .m_axi_wready   (m_axi_mem_wready_a),
        .m_axi_wdata    (m_axi_mem_wdata_a),
        .m_axi_wstrb    (m_axi_mem_wstrb_a),
        .m_axi_wlast    (m_axi_mem_wlast_a),

        .m_axi_bvalid   (m_axi_mem_bvalid_a),
        .m_axi_bready   (m_axi_mem_bready_a),
        .m_axi_bid      (m_axi_mem_bid_a),
        .m_axi_bresp    (m_axi_mem_bresp_a),

        .m_axi_arvalid  (m_axi_mem_arvalid_a),
        .m_axi_arready  (m_axi_mem_arready_a),
        .m_axi_araddr   (m_axi_mem_araddr_a),
        .m_axi_arid     (m_axi_mem_arid_a),
        .m_axi_arlen    (m_axi_mem_arlen_a),
        .m_axi_arsize   (),
        .m_axi_arburst  (),
        .m_axi_arlock   (),
        .m_axi_arcache  (),
        .m_axi_arprot   (),
        .m_axi_arqos    (),
        .m_axi_arregion (),

        .m_axi_rvalid   (m_axi_mem_rvalid_a),
        .m_axi_rready   (m_axi_mem_rready_a),
        .m_axi_rdata    (m_axi_mem_rdata_a),
        .m_axi_rlast    (m_axi_mem_rlast_a),
        .m_axi_rid      (m_axi_mem_rid_a),
        .m_axi_rresp    (m_axi_mem_rresp_a),

        .dcr_req_valid  (dcr_req_valid),
        .dcr_req_rw     (dcr_req_rw),
        .dcr_req_addr   (dcr_req_addr),
        .dcr_req_data   (dcr_req_data),

        .dcr_rsp_valid  (dcr_rsp_valid),
        .dcr_rsp_data   (dcr_rsp_data),

        .start          (start),
        .busy           (busy)
    );

endmodule
