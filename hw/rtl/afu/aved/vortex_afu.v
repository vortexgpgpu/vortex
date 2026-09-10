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

// The V80 linker infers bus interfaces from port-name prefixes when it
// packages this module as IP-XACT, and requires the AXI4-Lite slave to be
// named exactly `s_axi_control`. Everything below that boundary is shared
// with the other Xilinx AFU via hw/rtl/afu/common.
module vortex_afu #(
    parameter C_S_AXI_CONTROL_ADDR_WIDTH = 16,
    parameter C_S_AXI_CONTROL_DATA_WIDTH = 32,
    parameter C_M_AXI_MEM_ID_WIDTH       = `PLATFORM_MEMORY_ID_WIDTH,
    parameter C_M_AXI_MEM_DATA_WIDTH     = (`VX_CFG_PLATFORM_MEMORY_DATA_SIZE * 8),
    parameter C_M_AXI_MEM_ADDR_WIDTH     = 64,
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
    parameter C_M_AXI_MEM_NUM_BANKS      = 1
`else
    parameter C_M_AXI_MEM_NUM_BANKS      = `VX_CFG_PLATFORM_MEMORY_NUM_BANKS
`endif
) (
    // System signals
    input wire                                     ap_clk,
    input wire                                     ap_rst_n,

    // AXI4 master interface
`ifdef PLATFORM_MERGED_MEMORY_INTERFACE
    `MP_REPEAT (1, GEN_AXI_MEM, MP_COMMA),
`else
    `MP_REPEAT (`VX_CFG_PLATFORM_MEMORY_NUM_BANKS, GEN_AXI_MEM, MP_COMMA),
`endif

    // AXI4 host-memory master interface (CP command ring + host side of DMA)
    `GEN_AXI_HOST,

    // AXI4-Lite slave interface
    input  wire                                    s_axi_control_awvalid,
    output wire                                    s_axi_control_awready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr,

    input  wire                                    s_axi_control_wvalid,
    output wire                                    s_axi_control_wready,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata,
    input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb,

    input  wire                                    s_axi_control_arvalid,
    output wire                                    s_axi_control_arready,
    input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr,

    output wire                                    s_axi_control_rvalid,
    input  wire                                    s_axi_control_rready,
    output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata,
    output wire [1:0]                              s_axi_control_rresp,

    output wire                                    s_axi_control_bvalid,
    input  wire                                    s_axi_control_bready,
    output wire [1:0]                              s_axi_control_bresp,

    output wire                                    interrupt
);

    VX_afu_wrap #(
        .C_S_AXI_CTRL_ADDR_WIDTH (C_S_AXI_CONTROL_ADDR_WIDTH),
        .C_S_AXI_CTRL_DATA_WIDTH (C_S_AXI_CONTROL_DATA_WIDTH),
        .C_M_AXI_MEM_ID_WIDTH    (C_M_AXI_MEM_ID_WIDTH),
        .C_M_AXI_MEM_ADDR_WIDTH  (C_M_AXI_MEM_ADDR_WIDTH),
        .C_M_AXI_MEM_DATA_WIDTH  (C_M_AXI_MEM_DATA_WIDTH),
        .C_M_AXI_MEM_NUM_BANKS   (C_M_AXI_MEM_NUM_BANKS)
    ) afu_wrap (
        .clk                (ap_clk),
        .reset              (~ap_rst_n),
    `ifdef PLATFORM_MERGED_MEMORY_INTERFACE
        `MP_REPEAT (1, AXI_MEM_ARGS, MP_COMMA),
    `else
        `MP_REPEAT (`VX_CFG_PLATFORM_MEMORY_NUM_BANKS, AXI_MEM_ARGS, MP_COMMA),
    `endif
        `AXI_HOST_ARGS,
        .s_axi_ctrl_awvalid (s_axi_control_awvalid),
        .s_axi_ctrl_awready (s_axi_control_awready),
        .s_axi_ctrl_awaddr  (s_axi_control_awaddr),

        .s_axi_ctrl_wvalid  (s_axi_control_wvalid),
        .s_axi_ctrl_wready  (s_axi_control_wready),
        .s_axi_ctrl_wdata   (s_axi_control_wdata),
        .s_axi_ctrl_wstrb   (s_axi_control_wstrb),

        .s_axi_ctrl_arvalid (s_axi_control_arvalid),
        .s_axi_ctrl_arready (s_axi_control_arready),
        .s_axi_ctrl_araddr  (s_axi_control_araddr),

        .s_axi_ctrl_rvalid  (s_axi_control_rvalid),
        .s_axi_ctrl_rready  (s_axi_control_rready),
        .s_axi_ctrl_rdata   (s_axi_control_rdata),
        .s_axi_ctrl_rresp   (s_axi_control_rresp),

        .s_axi_ctrl_bvalid  (s_axi_control_bvalid),
        .s_axi_ctrl_bready  (s_axi_control_bready),
        .s_axi_ctrl_bresp   (s_axi_control_bresp),

        .interrupt          (interrupt)
    );

endmodule
