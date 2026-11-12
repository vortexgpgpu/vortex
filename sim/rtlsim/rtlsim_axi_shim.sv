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

`ifndef PLATFORM_MEMORY_ID_WIDTH
`define PLATFORM_MEMORY_ID_WIDTH 32
`endif

// Vortex behind its AXI4 memory adapter (Vortex_axi), as the FPGA platforms
// see it, with the AXI channels a memory model needs. The parameters are the
// ones the platform wrappers pass, so the adapter is the same logic they run.
module rtlsim_axi_shim import VX_gpu_pkg::*; #(
    parameter AXI_DATA_WIDTH = `VX_CFG_PLATFORM_MEMORY_DATA_SIZE * 8,
    parameter AXI_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH,
    parameter AXI_TID_WIDTH  = `PLATFORM_MEMORY_ID_WIDTH,
    parameter AXI_NUM_BANKS  = `VX_CFG_PLATFORM_MEMORY_NUM_BANKS
) (
    `SCOPE_IO_DECL

    input  wire                         clk,
    input  wire                         reset,

    output wire                         m_axi_awvalid [AXI_NUM_BANKS],
    input  wire                         m_axi_awready [AXI_NUM_BANKS],
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_awaddr [AXI_NUM_BANKS],
    output wire [AXI_TID_WIDTH-1:0]     m_axi_awid [AXI_NUM_BANKS],
    output wire [7:0]                   m_axi_awlen [AXI_NUM_BANKS],

    output wire                         m_axi_wvalid [AXI_NUM_BANKS],
    input  wire                         m_axi_wready [AXI_NUM_BANKS],
    output wire [AXI_DATA_WIDTH-1:0]    m_axi_wdata [AXI_NUM_BANKS],
    output wire [AXI_DATA_WIDTH/8-1:0]  m_axi_wstrb [AXI_NUM_BANKS],
    output wire                         m_axi_wlast [AXI_NUM_BANKS],

    input  wire                         m_axi_bvalid [AXI_NUM_BANKS],
    output wire                         m_axi_bready [AXI_NUM_BANKS],
    input  wire [AXI_TID_WIDTH-1:0]     m_axi_bid [AXI_NUM_BANKS],
    input  wire [1:0]                   m_axi_bresp [AXI_NUM_BANKS],

    output wire                         m_axi_arvalid [AXI_NUM_BANKS],
    input  wire                         m_axi_arready [AXI_NUM_BANKS],
    output wire [AXI_ADDR_WIDTH-1:0]    m_axi_araddr [AXI_NUM_BANKS],
    output wire [AXI_TID_WIDTH-1:0]     m_axi_arid [AXI_NUM_BANKS],
    output wire [7:0]                   m_axi_arlen [AXI_NUM_BANKS],

    input  wire                         m_axi_rvalid [AXI_NUM_BANKS],
    output wire                         m_axi_rready [AXI_NUM_BANKS],
    input  wire [AXI_DATA_WIDTH-1:0]    m_axi_rdata [AXI_NUM_BANKS],
    input  wire                         m_axi_rlast [AXI_NUM_BANKS],
    input  wire [AXI_TID_WIDTH-1:0]     m_axi_rid [AXI_NUM_BANKS],
    input  wire [1:0]                   m_axi_rresp [AXI_NUM_BANKS],

    input  wire                         dcr_req_valid,
    input  wire                         dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0] dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0] dcr_req_data,

    output wire                         dcr_rsp_valid,
    output wire [VX_DCR_DATA_WIDTH-1:0] dcr_rsp_data,

    input  wire                         start,
    output wire                         busy
);
    `SCOPE_IO_SWITCH (1);

    Vortex_axi #(
        .AXI_DATA_WIDTH (AXI_DATA_WIDTH),
        .AXI_ADDR_WIDTH (AXI_ADDR_WIDTH),
        .AXI_TID_WIDTH  (AXI_TID_WIDTH),
        .AXI_NUM_BANKS  (AXI_NUM_BANKS)
    ) vortex_axi (
        `SCOPE_IO_BIND  (0)

        .clk            (clk),
        .reset          (reset),

        .m_axi_awvalid  (m_axi_awvalid),
        .m_axi_awready  (m_axi_awready),
        .m_axi_awaddr   (m_axi_awaddr),
        .m_axi_awid     (m_axi_awid),
        .m_axi_awlen    (m_axi_awlen),
        `UNUSED_PIN     (m_axi_awsize),
        `UNUSED_PIN     (m_axi_awburst),
        `UNUSED_PIN     (m_axi_awlock),
        `UNUSED_PIN     (m_axi_awcache),
        `UNUSED_PIN     (m_axi_awprot),
        `UNUSED_PIN     (m_axi_awqos),
        `UNUSED_PIN     (m_axi_awregion),

        .m_axi_wvalid   (m_axi_wvalid),
        .m_axi_wready   (m_axi_wready),
        .m_axi_wdata    (m_axi_wdata),
        .m_axi_wstrb    (m_axi_wstrb),
        .m_axi_wlast    (m_axi_wlast),

        .m_axi_bvalid   (m_axi_bvalid),
        .m_axi_bready   (m_axi_bready),
        .m_axi_bid      (m_axi_bid),
        .m_axi_bresp    (m_axi_bresp),

        .m_axi_arvalid  (m_axi_arvalid),
        .m_axi_arready  (m_axi_arready),
        .m_axi_araddr   (m_axi_araddr),
        .m_axi_arid     (m_axi_arid),
        .m_axi_arlen    (m_axi_arlen),
        `UNUSED_PIN     (m_axi_arsize),
        `UNUSED_PIN     (m_axi_arburst),
        `UNUSED_PIN     (m_axi_arlock),
        `UNUSED_PIN     (m_axi_arcache),
        `UNUSED_PIN     (m_axi_arprot),
        `UNUSED_PIN     (m_axi_arqos),
        `UNUSED_PIN     (m_axi_arregion),

        .m_axi_rvalid   (m_axi_rvalid),
        .m_axi_rready   (m_axi_rready),
        .m_axi_rdata    (m_axi_rdata),
        .m_axi_rlast    (m_axi_rlast),
        .m_axi_rid      (m_axi_rid),
        .m_axi_rresp    (m_axi_rresp),

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
