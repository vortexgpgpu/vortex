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

// Harness for the ID-routed, multi-outstanding 2:1 VX_mm_axi_arb; the
// testbench plays both masters and the slave.
module VX_mm_axi_arb_top #(
    parameter NUM_INPUTS = 2,
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter ID_WIDTH   = 8
) (
    input  wire clk,
    input  wire reset,

    input  wire [NUM_INPUTS-1:0]                  s_awvalid,
    output wire [NUM_INPUTS-1:0]                  s_awready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_awaddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_awid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_awlen,

    input  wire [NUM_INPUTS-1:0]                  s_wvalid,
    output wire [NUM_INPUTS-1:0]                  s_wready,
    input  wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_wdata,
    input  wire [NUM_INPUTS-1:0]                  s_wlast,

    output wire [NUM_INPUTS-1:0]                  s_bvalid,
    input  wire [NUM_INPUTS-1:0]                  s_bready,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_bid,

    input  wire [NUM_INPUTS-1:0]                  s_arvalid,
    output wire [NUM_INPUTS-1:0]                  s_arready,
    input  wire [NUM_INPUTS-1:0][ADDR_WIDTH-1:0]  s_araddr,
    input  wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_arid,
    input  wire [NUM_INPUTS-1:0][7:0]             s_arlen,

    output wire [NUM_INPUTS-1:0]                  s_rvalid,
    input  wire [NUM_INPUTS-1:0]                  s_rready,
    output wire [NUM_INPUTS-1:0][DATA_WIDTH-1:0]  s_rdata,
    output wire [NUM_INPUTS-1:0]                  s_rlast,
    output wire [NUM_INPUTS-1:0][ID_WIDTH-1:0]    s_rid,

    output wire                                   m_awvalid,
    input  wire                                   m_awready,
    output wire [ADDR_WIDTH-1:0]                  m_awaddr,
    output wire [ID_WIDTH-1:0]                    m_awid,
    output wire [7:0]                             m_awlen,

    output wire                                   m_wvalid,
    input  wire                                   m_wready,
    output wire [DATA_WIDTH-1:0]                  m_wdata,
    output wire                                   m_wlast,

    input  wire                                   m_bvalid,
    output wire                                   m_bready,
    input  wire [ID_WIDTH-1:0]                    m_bid,

    output wire                                   m_arvalid,
    input  wire                                   m_arready,
    output wire [ADDR_WIDTH-1:0]                  m_araddr,
    output wire [ID_WIDTH-1:0]                    m_arid,
    output wire [7:0]                             m_arlen,

    input  wire                                   m_rvalid,
    output wire                                   m_rready,
    input  wire [DATA_WIDTH-1:0]                  m_rdata,
    input  wire                                   m_rlast,
    input  wire [ID_WIDTH-1:0]                    m_rid
);
    localparam STRB_WIDTH = DATA_WIDTH / 8;

    wire [NUM_INPUTS-1:0][1:0] s_bresp, s_rresp;
    wire [STRB_WIDTH-1:0]      m_wstrb;
    `UNUSED_VAR ({s_bresp, s_rresp, m_wstrb})

    VX_mm_axi_arb #(
        .NUM_INPUTS (NUM_INPUTS),
        .ADDR_WIDTH (ADDR_WIDTH),
        .DATA_WIDTH (DATA_WIDTH),
        .ID_WIDTH   (ID_WIDTH),
        .ARBITER    ("R"),
        .MULTI_OUT  (1)
    ) arb (
        .clk       (clk),
        .reset     (reset),

        .s_awvalid (s_awvalid),
        .s_awready (s_awready),
        .s_awaddr  (s_awaddr),
        .s_awid    (s_awid),
        .s_awlen   (s_awlen),

        .s_wvalid  (s_wvalid),
        .s_wready  (s_wready),
        .s_wdata   (s_wdata),
        .s_wstrb   ({NUM_INPUTS{{STRB_WIDTH{1'b1}}}}),
        .s_wlast   (s_wlast),

        .s_bvalid  (s_bvalid),
        .s_bready  (s_bready),
        .s_bid     (s_bid),
        .s_bresp   (s_bresp),

        .s_arvalid (s_arvalid),
        .s_arready (s_arready),
        .s_araddr  (s_araddr),
        .s_arid    (s_arid),
        .s_arlen   (s_arlen),

        .s_rvalid  (s_rvalid),
        .s_rready  (s_rready),
        .s_rdata   (s_rdata),
        .s_rlast   (s_rlast),
        .s_rid     (s_rid),
        .s_rresp   (s_rresp),

        .m_awvalid (m_awvalid),
        .m_awready (m_awready),
        .m_awaddr  (m_awaddr),
        .m_awid    (m_awid),
        .m_awlen   (m_awlen),

        .m_wvalid  (m_wvalid),
        .m_wready  (m_wready),
        .m_wdata   (m_wdata),
        .m_wstrb   (m_wstrb),
        .m_wlast   (m_wlast),

        .m_bvalid  (m_bvalid),
        .m_bready  (m_bready),
        .m_bid     (m_bid),
        .m_bresp   (2'b00),

        .m_arvalid (m_arvalid),
        .m_arready (m_arready),
        .m_araddr  (m_araddr),
        .m_arid    (m_arid),
        .m_arlen   (m_arlen),

        .m_rvalid  (m_rvalid),
        .m_rready  (m_rready),
        .m_rdata   (m_rdata),
        .m_rlast   (m_rlast),
        .m_rid     (m_rid),
        .m_rresp   (2'b00)
    );

endmodule
