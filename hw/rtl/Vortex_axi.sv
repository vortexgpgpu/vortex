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

module Vortex_axi import VX_gpu_pkg::*; #(
    parameter AXI_DATA_WIDTH = `VX_MEM_DATA_WIDTH,
    parameter AXI_ADDR_WIDTH = `MEM_ADDR_WIDTH,
    parameter AXI_TID_WIDTH  = `VX_MEM_TAG_WIDTH,
    parameter AXI_NUM_BANKS  = 1
)(
    `SCOPE_IO_DECL

    // Clock
    input  wire                         clk,
    input  wire                         reset,

    // AXI write request address channel
    output wire                         m_axi_awvalid [AXI_NUM_BANKS],
    input wire                          m_axi_awready [AXI_NUM_BANKS],
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr [AXI_NUM_BANKS],
    output wire [AXI_TID_WIDTH-1:0]     m_axi_awid [AXI_NUM_BANKS],
    output wire [7:0]                   m_axi_awlen [AXI_NUM_BANKS],
    output wire [2:0]                   m_axi_awsize [AXI_NUM_BANKS],
    output wire [1:0]                   m_axi_awburst [AXI_NUM_BANKS],
    output wire [1:0]                   m_axi_awlock [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_awcache [AXI_NUM_BANKS],
    output wire [2:0]                   m_axi_awprot [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_awqos [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_awregion [AXI_NUM_BANKS],

    // AXI write request data channel
    output wire                         m_axi_wvalid [AXI_NUM_BANKS],
    input wire                          m_axi_wready [AXI_NUM_BANKS],
    output wire [AXI_DATA_WIDTH-1:0]    m_axi_wdata [AXI_NUM_BANKS],
    output wire [AXI_DATA_WIDTH/8-1:0]  m_axi_wstrb [AXI_NUM_BANKS],
    output wire                         m_axi_wlast [AXI_NUM_BANKS],

    // AXI write response channel
    input wire                          m_axi_bvalid [AXI_NUM_BANKS],
    output wire                         m_axi_bready [AXI_NUM_BANKS],
    input wire [AXI_TID_WIDTH-1:0]      m_axi_bid [AXI_NUM_BANKS],
    input wire [1:0]                    m_axi_bresp [AXI_NUM_BANKS],

    // AXI read request channel
    output wire                         m_axi_arvalid [AXI_NUM_BANKS],
    input wire                          m_axi_arready [AXI_NUM_BANKS],
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_araddr [AXI_NUM_BANKS],
    output wire [AXI_TID_WIDTH-1:0]     m_axi_arid [AXI_NUM_BANKS],
    output wire [7:0]                   m_axi_arlen [AXI_NUM_BANKS],
    output wire [2:0]                   m_axi_arsize [AXI_NUM_BANKS],
    output wire [1:0]                   m_axi_arburst [AXI_NUM_BANKS],
    output wire [1:0]                   m_axi_arlock [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_arcache [AXI_NUM_BANKS],
    output wire [2:0]                   m_axi_arprot [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_arqos [AXI_NUM_BANKS],
    output wire [3:0]                   m_axi_arregion [AXI_NUM_BANKS],

    // AXI read response channel
    input wire                          m_axi_rvalid [AXI_NUM_BANKS],
    output wire                         m_axi_rready [AXI_NUM_BANKS],
    input wire [AXI_DATA_WIDTH-1:0]     m_axi_rdata [AXI_NUM_BANKS],
    input wire                          m_axi_rlast [AXI_NUM_BANKS],
    input wire [AXI_TID_WIDTH-1:0]      m_axi_rid [AXI_NUM_BANKS],
    input wire [1:0]                    m_axi_rresp [AXI_NUM_BANKS],

    // DCR write request
    input  wire                         dcr_wr_valid,
    input  wire [`VX_DCR_ADDR_WIDTH-1:0] dcr_wr_addr,
    input  wire [`VX_DCR_DATA_WIDTH-1:0] dcr_wr_data,

    // Status
    output wire                         busy
);
    localparam MIN_TAG_WIDTH = `VX_MEM_TAG_WIDTH - `UUID_WIDTH;

    `STATIC_ASSERT((AXI_DATA_WIDTH == `VX_MEM_DATA_WIDTH), ("invalid memory data size: current=%0d, expected=%0d", AXI_DATA_WIDTH, `VX_MEM_DATA_WIDTH))
    `STATIC_ASSERT((AXI_ADDR_WIDTH >= `MEM_ADDR_WIDTH), ("invalid memory address size: current=%0d, expected=%0d", AXI_ADDR_WIDTH, `VX_MEM_ADDR_WIDTH))
    `STATIC_ASSERT((AXI_TID_WIDTH >= MIN_TAG_WIDTH), ("invalid memory tag size: current=%0d, expected=%0d", AXI_TID_WIDTH, MIN_TAG_WIDTH))

    wire                            mem_req_valid;
    wire                            mem_req_rw;
    wire [`VX_MEM_BYTEEN_WIDTH-1:0] mem_req_byteen;
    wire [`VX_MEM_ADDR_WIDTH-1:0]   mem_req_addr;
    wire [`VX_MEM_DATA_WIDTH-1:0]   mem_req_data;
    wire [`VX_MEM_TAG_WIDTH-1:0]    mem_req_tag;
    wire                            mem_req_ready;

    wire                            mem_rsp_valid;
    wire [`VX_MEM_DATA_WIDTH-1:0]   mem_rsp_data;
    wire [`VX_MEM_TAG_WIDTH-1:0]    mem_rsp_tag;
    wire                            mem_rsp_ready;

    wire [`MEM_ADDR_WIDTH-1:0] m_axi_awaddr_unqual [AXI_NUM_BANKS];
    wire [`MEM_ADDR_WIDTH-1:0] m_axi_araddr_unqual [AXI_NUM_BANKS];

    wire [`VX_MEM_TAG_WIDTH-1:0] m_axi_awid_unqual [AXI_NUM_BANKS];
    wire [`VX_MEM_TAG_WIDTH-1:0] m_axi_arid_unqual [AXI_NUM_BANKS];

    wire [`VX_MEM_TAG_WIDTH-1:0] m_axi_bid_unqual [AXI_NUM_BANKS];
    wire [`VX_MEM_TAG_WIDTH-1:0] m_axi_rid_unqual [AXI_NUM_BANKS];

    for (genvar i = 0; i < AXI_NUM_BANKS; ++i) begin
        assign m_axi_awaddr[i] = `MEM_ADDR_WIDTH'(m_axi_awaddr_unqual[i]);
        assign m_axi_araddr[i] = `MEM_ADDR_WIDTH'(m_axi_araddr_unqual[i]);

        assign m_axi_awid[i] = AXI_TID_WIDTH'(m_axi_awid_unqual[i]);
        assign m_axi_arid[i] = AXI_TID_WIDTH'(m_axi_arid_unqual[i]);

        assign m_axi_rid_unqual[i] = `VX_MEM_TAG_WIDTH'(m_axi_rid[i]);
        assign m_axi_bid_unqual[i] = `VX_MEM_TAG_WIDTH'(m_axi_bid[i]);
    end

    VX_axi_adapter #(
        .DATA_WIDTH (`VX_MEM_DATA_WIDTH),
        .ADDR_WIDTH (`MEM_ADDR_WIDTH),
        .TAG_WIDTH  (`VX_MEM_TAG_WIDTH),
        .NUM_BANKS  (AXI_NUM_BANKS),
        .RSP_OUT_BUF((AXI_NUM_BANKS > 1) ? 2 : 0)
    ) axi_adapter (
        .clk            (clk),
        .reset          (reset),

        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),

        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready),

        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_awaddr   (m_axi_awaddr_unqual),
        .m_axi_awid     (m_axi_awid_unqual),
        .m_axi_awlen    (m_axi_awlen),
        .m_axi_awsize   (m_axi_awsize),
        .m_axi_awburst  (m_axi_awburst),
        .m_axi_awlock   (m_axi_awlock),
        .m_axi_awcache  (m_axi_awcache),
        .m_axi_awprot   (m_axi_awprot),
        .m_axi_awqos    (m_axi_awqos),
        .m_axi_awregion (m_axi_awregion),

        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),

        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .m_axi_bid      (m_axi_bid_unqual),
        .m_axi_bresp    (m_axi_bresp),

        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_araddr   (m_axi_araddr_unqual),
        .m_axi_arid     (m_axi_arid_unqual),
        .m_axi_arlen    (m_axi_arlen),
        .m_axi_arsize   (m_axi_arsize),
        .m_axi_arburst  (m_axi_arburst),
        .m_axi_arlock   (m_axi_arlock),
        .m_axi_arcache  (m_axi_arcache),
        .m_axi_arprot   (m_axi_arprot),
        .m_axi_arqos    (m_axi_arqos),
        .m_axi_arregion (m_axi_arregion),

        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rlast    (m_axi_rlast) ,
        .m_axi_rid      (m_axi_rid_unqual),
        .m_axi_rresp    (m_axi_rresp)
    );

    `SCOPE_IO_SWITCH (1)

    Vortex vortex (
        `SCOPE_IO_BIND  (0)

        .clk            (clk),
        .reset          (reset),

        .mem_req_valid  (mem_req_valid),
        .mem_req_rw     (mem_req_rw),
        .mem_req_byteen (mem_req_byteen),
        .mem_req_addr   (mem_req_addr),
        .mem_req_data   (mem_req_data),
        .mem_req_tag    (mem_req_tag),
        .mem_req_ready  (mem_req_ready),

        .mem_rsp_valid  (mem_rsp_valid),
        .mem_rsp_data   (mem_rsp_data),
        .mem_rsp_tag    (mem_rsp_tag),
        .mem_rsp_ready  (mem_rsp_ready),

        .dcr_wr_valid   (dcr_wr_valid),
        .dcr_wr_addr    (dcr_wr_addr),
        .dcr_wr_data    (dcr_wr_data),

        .busy           (busy)
    );

endmodule
