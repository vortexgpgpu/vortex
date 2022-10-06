/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

///////////////////////////////////////////////////////////////////////////////
// Description: This is a wrapper of module "krnl_vadd_rtl_int"
///////////////////////////////////////////////////////////////////////////////

// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps

`ifndef NOGLOBALS
`include "globals.vh"
`endif

module krnl_vadd_rtl #( 
  parameter integer  C_S_AXI_CTRL_DATA_WIDTH = 32,
  parameter integer  C_S_AXI_CTRL_ADDR_WIDTH = 6,
  parameter integer  C_M_AXI_MEM_ID_WIDTH    = 1,
  parameter integer  C_M_AXI_MEM_ADDR_WIDTH  = 64,
  parameter integer  C_M_AXI_MEM_DATA_WIDTH  = 32
)
(
  // System signals
  input  wire  ap_clk,
  input  wire  ap_rst_n,

  // AXI4 master interface 
  output wire                                 m_axi_mem_awvalid,
  input  wire                                 m_axi_mem_awready,
  output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_awaddr,
  output wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_awid,
  output wire [7:0]                           m_axi_mem_awlen,
  output wire [2:0]                           m_axi_mem_awsize,
  output wire [1:0]                           m_axi_mem_awburst,
  output wire [1:0]                           m_axi_mem_awlock,
  output wire [3:0]                           m_axi_mem_awcache,
  output wire [2:0]                           m_axi_mem_awprot,
  output wire [3:0]                           m_axi_mem_awqos,
  output wire [3:0]                           m_axi_mem_awregion,
  output wire                                 m_axi_mem_wvalid,
  input  wire                                 m_axi_mem_wready,
  output wire [C_M_AXI_MEM_DATA_WIDTH-1:0]    m_axi_mem_wdata,
  output wire [C_M_AXI_MEM_DATA_WIDTH/8-1:0]  m_axi_mem_wstrb,
  output wire                                 m_axi_mem_wlast,
  output wire                                 m_axi_mem_arvalid,
  input  wire                                 m_axi_mem_arready,
  output wire [C_M_AXI_MEM_ADDR_WIDTH-1:0]    m_axi_mem_araddr,
  output wire [C_M_AXI_MEM_ID_WIDTH-1:0]      m_axi_mem_arid,
  output wire [7:0]                           m_axi_mem_arlen,
  output wire [2:0]                           m_axi_mem_arsize,
  output wire [1:0]                           m_axi_mem_arburst,
  output wire [1:0]                           m_axi_mem_arlock,
  output wire [3:0]                           m_axi_mem_arcache,
  output wire [2:0]                           m_axi_mem_arprot,
  output wire [3:0]                           m_axi_mem_arqos,
  output wire [3:0]                           m_axi_mem_arregion,
  input  wire                                 m_axi_mem_rvalid,
  output wire                                 m_axi_mem_rready,
  input  wire [C_M_AXI_MEM_DATA_WIDTH - 1:0]  m_axi_mem_rdata,
  input  wire                                 m_axi_mem_rlast,
  input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_rid,
  input  wire [1:0]                           m_axi_mem_rresp,
  input  wire                                 m_axi_mem_bvalid,
  output wire                                 m_axi_mem_bready,
  input  wire [1:0]                           m_axi_mem_bresp,
  input  wire [C_M_AXI_MEM_ID_WIDTH - 1:0]    m_axi_mem_bid,

  // AXI4-Lite slave interface
  input  wire                                 s_axi_ctrl_awvalid,
  output wire                                 s_axi_ctrl_awready,
  input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_awaddr,
  input  wire                                 s_axi_ctrl_wvalid,
  output wire                                 s_axi_ctrl_wready,
  input  wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_wdata,
  input  wire [C_S_AXI_CTRL_DATA_WIDTH/8-1:0] s_axi_ctrl_wstrb,
  input  wire                                 s_axi_ctrl_arvalid,
  output wire                                 s_axi_ctrl_arready,
  input  wire [C_S_AXI_CTRL_ADDR_WIDTH-1:0]   s_axi_ctrl_araddr,
  output wire                                 s_axi_ctrl_rvalid,
  input  wire                                 s_axi_ctrl_rready,
  output wire [C_S_AXI_CTRL_DATA_WIDTH-1:0]   s_axi_ctrl_rdata,
  output wire [1:0]                           s_axi_ctrl_rresp,
  output wire                                 s_axi_ctrl_bvalid,
  input  wire                                 s_axi_ctrl_bready,
  output wire [1:0]                           s_axi_ctrl_bresp,
  output wire                                 interrupt 
);

krnl_vadd_rtl_int #(
  .C_S_AXI_CTRL_DATA_WIDTH  ( C_S_AXI_CTRL_DATA_WIDTH ),
  .C_S_AXI_CTRL_ADDR_WIDTH  ( C_S_AXI_CTRL_ADDR_WIDTH ),
  .C_M_AXI_MEM_ID_WIDTH     ( C_M_AXI_MEM_ID_WIDTH ),
  .C_M_AXI_MEM_ADDR_WIDTH   ( C_M_AXI_MEM_ADDR_WIDTH ),
  .C_M_AXI_MEM_DATA_WIDTH   ( C_M_AXI_MEM_DATA_WIDTH )
)
inst_krnl_vadd_rtl_int (
  .ap_clk                ( ap_clk ),
  .ap_rst_n              ( ap_rst_n ),
  .m_axi_mem_awvalid     ( m_axi_mem_awvalid ),
  .m_axi_mem_awready     ( m_axi_mem_awready ),
  .m_axi_mem_awaddr      ( m_axi_mem_awaddr ),
  .m_axi_mem_awid        ( m_axi_mem_awid ),
  .m_axi_mem_awlen       ( m_axi_mem_awlen ),
  .m_axi_mem_awsize      ( m_axi_mem_awsize ),
  .m_axi_mem_awburst     ( m_axi_mem_awburst ),
  .m_axi_mem_awlock      ( m_axi_mem_awlock ),
  .m_axi_mem_awcache     ( m_axi_mem_awcache ),
  .m_axi_mem_awprot      ( m_axi_mem_awprot ),
  .m_axi_mem_awqos       ( m_axi_mem_awqos ),
  .m_axi_mem_awregion    ( m_axi_mem_awregion ),
  .m_axi_mem_wvalid      ( m_axi_mem_wvalid ),
  .m_axi_mem_wready      ( m_axi_mem_wready ),
  .m_axi_mem_wdata       ( m_axi_mem_wdata ),
  .m_axi_mem_wstrb       ( m_axi_mem_wstrb ),
  .m_axi_mem_wlast       ( m_axi_mem_wlast ),
  .m_axi_mem_arvalid     ( m_axi_mem_arvalid ),
  .m_axi_mem_arready     ( m_axi_mem_arready ),
  .m_axi_mem_araddr      ( m_axi_mem_araddr ),
  .m_axi_mem_arid        ( m_axi_mem_arid ),
  .m_axi_mem_arlen       ( m_axi_mem_arlen ),
  .m_axi_mem_arsize      ( m_axi_mem_arsize ),
  .m_axi_mem_arburst     ( m_axi_mem_arburst ),
  .m_axi_mem_arlock      ( m_axi_mem_arlock ),
  .m_axi_mem_arcache     ( m_axi_mem_arcache ),
  .m_axi_mem_arprot      ( m_axi_mem_arprot ),
  .m_axi_mem_arqos       ( m_axi_mem_arqos ),
  .m_axi_mem_arregion    ( m_axi_mem_arregion ),
  .m_axi_mem_rvalid      ( m_axi_mem_rvalid ),
  .m_axi_mem_rready      ( m_axi_mem_rready ),
  .m_axi_mem_rdata       ( m_axi_mem_rdata ),
  .m_axi_mem_rlast       ( m_axi_mem_rlast ),
  .m_axi_mem_rid         ( m_axi_mem_rid ),
  .m_axi_mem_rresp       ( m_axi_mem_rresp ),
  .m_axi_mem_bvalid      ( m_axi_mem_bvalid ),
  .m_axi_mem_bready      ( m_axi_mem_bready ),
  .m_axi_mem_bresp       ( m_axi_mem_bresp ),
  .m_axi_mem_bid         ( m_axi_mem_bid ),
  .s_axi_ctrl_awvalid    ( s_axi_ctrl_awvalid ),
  .s_axi_ctrl_awready    ( s_axi_ctrl_awready ),
  .s_axi_ctrl_awaddr     ( s_axi_ctrl_awaddr ),
  .s_axi_ctrl_wvalid     ( s_axi_ctrl_wvalid ),
  .s_axi_ctrl_wready     ( s_axi_ctrl_wready ),
  .s_axi_ctrl_wdata      ( s_axi_ctrl_wdata ),
  .s_axi_ctrl_wstrb      ( s_axi_ctrl_wstrb ),
  .s_axi_ctrl_arvalid    ( s_axi_ctrl_arvalid ),
  .s_axi_ctrl_arready    ( s_axi_ctrl_arready ),
  .s_axi_ctrl_araddr     ( s_axi_ctrl_araddr ),
  .s_axi_ctrl_rvalid     ( s_axi_ctrl_rvalid ),
  .s_axi_ctrl_rready     ( s_axi_ctrl_rready ),
  .s_axi_ctrl_rdata      ( s_axi_ctrl_rdata ),
  .s_axi_ctrl_rresp      ( s_axi_ctrl_rresp ),
  .s_axi_ctrl_bvalid     ( s_axi_ctrl_bvalid ),
  .s_axi_ctrl_bready     ( s_axi_ctrl_bready ),
  .s_axi_ctrl_bresp      ( s_axi_ctrl_bresp ),
  .interrupt             ( interrupt )
);
endmodule : krnl_vadd_rtl

`default_nettype wire
