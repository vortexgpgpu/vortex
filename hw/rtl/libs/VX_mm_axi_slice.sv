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

`include "VX_platform.vh"

// ============================================================================
// VX_mm_axi_slice — full-bandwidth AXI4 register slice (reduced view), flat ports.
// Inserts a registered stage (VX_skid_buffer) on every channel between an
// upstream master `s` and a downstream slave `m`, for an SLR-safe / timing-
// friendly boundary. OUT_REG selects the skid output-register depth.
// ============================================================================

`TRACING_OFF
module VX_mm_axi_slice #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512,
    parameter ID_WIDTH   = 32,
    parameter OUT_REG    = 1,
    parameter STRB_WIDTH = DATA_WIDTH/8
) (
    input  wire clk,
    input  wire reset,

    // ---- Upstream master (slave-side of the slice) ----
    input  wire                  s_awvalid,
    output wire                  s_awready,
    input  wire [ADDR_WIDTH-1:0] s_awaddr,
    input  wire [ID_WIDTH-1:0]   s_awid,
    input  wire [7:0]            s_awlen,

    input  wire                  s_wvalid,
    output wire                  s_wready,
    input  wire [DATA_WIDTH-1:0] s_wdata,
    input  wire [STRB_WIDTH-1:0] s_wstrb,
    input  wire                  s_wlast,

    output wire                  s_bvalid,
    input  wire                  s_bready,
    output wire [ID_WIDTH-1:0]   s_bid,
    output wire [1:0]            s_bresp,

    input  wire                  s_arvalid,
    output wire                  s_arready,
    input  wire [ADDR_WIDTH-1:0] s_araddr,
    input  wire [ID_WIDTH-1:0]   s_arid,
    input  wire [7:0]            s_arlen,

    output wire                  s_rvalid,
    input  wire                  s_rready,
    output wire [DATA_WIDTH-1:0] s_rdata,
    output wire                  s_rlast,
    output wire [ID_WIDTH-1:0]   s_rid,
    output wire [1:0]            s_rresp,

    // ---- Downstream slave (master-side of the slice) ----
    output wire                  m_awvalid,
    input  wire                  m_awready,
    output wire [ADDR_WIDTH-1:0] m_awaddr,
    output wire [ID_WIDTH-1:0]   m_awid,
    output wire [7:0]            m_awlen,

    output wire                  m_wvalid,
    input  wire                  m_wready,
    output wire [DATA_WIDTH-1:0] m_wdata,
    output wire [STRB_WIDTH-1:0] m_wstrb,
    output wire                  m_wlast,

    input  wire                  m_bvalid,
    output wire                  m_bready,
    input  wire [ID_WIDTH-1:0]   m_bid,
    input  wire [1:0]            m_bresp,

    output wire                  m_arvalid,
    input  wire                  m_arready,
    output wire [ADDR_WIDTH-1:0] m_araddr,
    output wire [ID_WIDTH-1:0]   m_arid,
    output wire [7:0]            m_arlen,

    input  wire                  m_rvalid,
    output wire                  m_rready,
    input  wire [DATA_WIDTH-1:0] m_rdata,
    input  wire                  m_rlast,
    input  wire [ID_WIDTH-1:0]   m_rid,
    input  wire [1:0]            m_rresp
);
    localparam AW_W = ADDR_WIDTH + ID_WIDTH + 8;              // addr,id,len
    localparam W_W  = DATA_WIDTH + STRB_WIDTH + 1;            // data,strb,last
    localparam B_W  = ID_WIDTH + 2;                          // id,resp
    localparam R_W  = DATA_WIDTH + ID_WIDTH + 1 + 2;         // data,id,last,resp

    // ---- AW : s -> m ----
    VX_skid_buffer #(.DATAW (AW_W), .OUT_REG (OUT_REG)) aw_slice (
        .clk (clk), .reset (reset),
        .valid_in  (s_awvalid), .ready_in  (s_awready),
        .data_in   ({s_awaddr, s_awid, s_awlen}),
        .valid_out (m_awvalid), .ready_out (m_awready),
        .data_out  ({m_awaddr, m_awid, m_awlen})
    );

    // ---- W : s -> m ----
    VX_skid_buffer #(.DATAW (W_W), .OUT_REG (OUT_REG)) w_slice (
        .clk (clk), .reset (reset),
        .valid_in  (s_wvalid), .ready_in  (s_wready),
        .data_in   ({s_wdata, s_wstrb, s_wlast}),
        .valid_out (m_wvalid), .ready_out (m_wready),
        .data_out  ({m_wdata, m_wstrb, m_wlast})
    );

    // ---- B : m -> s ----
    VX_skid_buffer #(.DATAW (B_W), .OUT_REG (OUT_REG)) b_slice (
        .clk (clk), .reset (reset),
        .valid_in  (m_bvalid), .ready_in  (m_bready),
        .data_in   ({m_bid, m_bresp}),
        .valid_out (s_bvalid), .ready_out (s_bready),
        .data_out  ({s_bid, s_bresp})
    );

    // ---- AR : s -> m ----
    VX_skid_buffer #(.DATAW (AW_W), .OUT_REG (OUT_REG)) ar_slice (
        .clk (clk), .reset (reset),
        .valid_in  (s_arvalid), .ready_in  (s_arready),
        .data_in   ({s_araddr, s_arid, s_arlen}),
        .valid_out (m_arvalid), .ready_out (m_arready),
        .data_out  ({m_araddr, m_arid, m_arlen})
    );

    // ---- R : m -> s ----
    VX_skid_buffer #(.DATAW (R_W), .OUT_REG (OUT_REG)) r_slice (
        .clk (clk), .reset (reset),
        .valid_in  (m_rvalid), .ready_in  (m_rready),
        .data_in   ({m_rdata, m_rid, m_rlast, m_rresp}),
        .valid_out (s_rvalid), .ready_out (s_rready),
        .data_out  ({s_rdata, s_rid, s_rlast, s_rresp})
    );

endmodule
`TRACING_ON
