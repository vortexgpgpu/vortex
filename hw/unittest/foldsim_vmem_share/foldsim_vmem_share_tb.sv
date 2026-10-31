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

// NC foldsim_vmem controllers on one AXI4 master through foldsim_vmem_share,
// as foldsim_afu shares a master: one reset, `zero` and `move` for all, each
// controller its own base. The controllers' request and response ports are
// flattened, controller k at bits [k*W +: W].
module foldsim_vmem_share_tb #(
    parameter NC             = 3,
    parameter IW             = 8,
    parameter AW             = 10,
    parameter DW             = 512,
    parameter MW             = DW / 8,
    parameter RDW_READ_FIRST = 1,
    parameter QUEUE          = 32,
    parameter AXI_ID_W       = 1,
    parameter OUTSTANDING    = 1 << AXI_ID_W,
    parameter INSTANCES      = 1 << IW,
    parameter CACHE_ROWS     = 0,
    parameter KNOWN_ZERO     = 1,
    parameter FW             = $clog2(QUEUE + 1)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     zero,
    input  wire                     move,
    input  wire [NC*64-1:0]         base,

    input  wire [NC-1:0]            rd_valid,
    input  wire [NC*IW-1:0]         rd_inst,
    input  wire [NC*AW-1:0]         rd_row,
    input  wire [NC-1:0]            wr_valid,
    input  wire [NC*IW-1:0]         wr_inst,
    input  wire [NC*AW-1:0]         wr_row,
    input  wire [NC*DW-1:0]         wr_data,
    input  wire [NC*MW-1:0]         wr_mask,
    output wire [NC-1:0]            rsp_valid,
    output wire [NC*IW-1:0]         rsp_inst,
    output wire [NC*DW-1:0]         rsp_data,
    output wire [NC*FW-1:0]         free,
    output wire [NC-1:0]            moving,

    output wire                     m_awvalid,
    input  wire                     m_awready,
    output wire [63:0]              m_awaddr,
    output wire [AXI_ID_W-1:0]      m_awid,
    output wire [7:0]               m_awlen,
    output wire                     m_wvalid,
    input  wire                     m_wready,
    output wire [511:0]             m_wdata,
    output wire [63:0]              m_wstrb,
    output wire                     m_wlast,
    input  wire                     m_bvalid,
    output wire                     m_bready,
    input  wire [AXI_ID_W-1:0]      m_bid,
    output wire                     m_arvalid,
    input  wire                     m_arready,
    output wire [63:0]              m_araddr,
    output wire [AXI_ID_W-1:0]      m_arid,
    output wire [7:0]               m_arlen,
    input  wire                     m_rvalid,
    output wire                     m_rready,
    input  wire [511:0]             m_rdata,
    input  wire [AXI_ID_W-1:0]      m_rid,
    input  wire                     m_rlast
);
    wire                s_awvalid [NC], s_awready [NC];
    wire [63:0]         s_awaddr  [NC];
    wire [AXI_ID_W-1:0] s_awid    [NC];
    wire [7:0]          s_awlen   [NC];
    wire                s_wvalid  [NC], s_wready [NC], s_wlast [NC];
    wire [511:0]        s_wdata   [NC];
    wire [63:0]         s_wstrb   [NC];
    wire                s_bvalid  [NC], s_bready [NC];
    wire [AXI_ID_W-1:0] s_bid     [NC];
    wire                s_arvalid [NC], s_arready [NC];
    wire [63:0]         s_araddr  [NC];
    wire [AXI_ID_W-1:0] s_arid    [NC];
    wire [7:0]          s_arlen   [NC];
    wire                s_rvalid  [NC], s_rready [NC], s_rlast [NC];
    wire [511:0]        s_rdata   [NC];
    wire [AXI_ID_W-1:0] s_rid     [NC];

    for (genvar k = 0; k < NC; ++k) begin : g_ctrl
        foldsim_vmem #(
            .IW             (IW),
            .AW             (AW),
            .DW             (DW),
            .MW             (MW),
            .RDW_READ_FIRST (RDW_READ_FIRST),
            .QUEUE          (QUEUE),
            .AXI_ID_W       (AXI_ID_W),
            .OUTSTANDING    (OUTSTANDING),
            .INSTANCES      (INSTANCES),
            .CACHE_ROWS     (CACHE_ROWS),
            .KNOWN_ZERO     (KNOWN_ZERO)
        ) vmem (
            .clk       (clk),
            .reset     (reset),
            .rd_valid  (rd_valid[k]),
            .rd_inst   (rd_inst[k*IW +: IW]),
            .rd_row    (rd_row[k*AW +: AW]),
            .wr_valid  (wr_valid[k]),
            .wr_inst   (wr_inst[k*IW +: IW]),
            .wr_row    (wr_row[k*AW +: AW]),
            .wr_data   (wr_data[k*DW +: DW]),
            .wr_mask   (wr_mask[k*MW +: MW]),
            .rsp_valid (rsp_valid[k]),
            .rsp_inst  (rsp_inst[k*IW +: IW]),
            .rsp_data  (rsp_data[k*DW +: DW]),
            .free      (free[k*FW +: FW]),
            .base      (base[k*64 +: 64]),
            .zero      (zero),
            .move      (move),
            .moving    (moving[k]),
            .m_awvalid (s_awvalid[k]),
            .m_awready (s_awready[k]),
            .m_awaddr  (s_awaddr[k]),
            .m_awid    (s_awid[k]),
            .m_awlen   (s_awlen[k]),
            .m_wvalid  (s_wvalid[k]),
            .m_wready  (s_wready[k]),
            .m_wdata   (s_wdata[k]),
            .m_wstrb   (s_wstrb[k]),
            .m_wlast   (s_wlast[k]),
            .m_bvalid  (s_bvalid[k]),
            .m_bready  (s_bready[k]),
            .m_bid     (s_bid[k]),
            .m_arvalid (s_arvalid[k]),
            .m_arready (s_arready[k]),
            .m_araddr  (s_araddr[k]),
            .m_arid    (s_arid[k]),
            .m_arlen   (s_arlen[k]),
            .m_rvalid  (s_rvalid[k]),
            .m_rready  (s_rready[k]),
            .m_rdata   (s_rdata[k]),
            .m_rid     (s_rid[k]),
            .m_rlast   (s_rlast[k])
        );
    end

    foldsim_vmem_share #(
        .N      (NC),
        .ID_W   (AXI_ID_W),
        .RDEPTH (NC),
        .WDEPTH (NC * QUEUE)
    ) share (
        .clk       (clk),
        .s_awvalid (s_awvalid), .s_awready (s_awready), .s_awaddr (s_awaddr), .s_awid (s_awid),
        .s_awlen   (s_awlen),
        .s_wvalid  (s_wvalid),  .s_wready  (s_wready),  .s_wdata  (s_wdata),  .s_wstrb (s_wstrb),
        .s_wlast   (s_wlast),
        .s_bvalid  (s_bvalid),  .s_bready  (s_bready),  .s_bid    (s_bid),
        .s_arvalid (s_arvalid), .s_arready (s_arready), .s_araddr (s_araddr), .s_arid  (s_arid),
        .s_arlen   (s_arlen),
        .s_rvalid  (s_rvalid),  .s_rready  (s_rready),  .s_rdata  (s_rdata),  .s_rid   (s_rid),
        .s_rlast   (s_rlast),
        .m_awvalid (m_awvalid), .m_awready (m_awready), .m_awaddr (m_awaddr), .m_awid  (m_awid),
        .m_awlen   (m_awlen),
        .m_wvalid  (m_wvalid),  .m_wready  (m_wready),  .m_wdata  (m_wdata),  .m_wstrb (m_wstrb),
        .m_wlast   (m_wlast),
        .m_bvalid  (m_bvalid),  .m_bready  (m_bready),  .m_bid    (m_bid),
        .m_arvalid (m_arvalid), .m_arready (m_arready), .m_araddr (m_araddr), .m_arid  (m_arid),
        .m_arlen   (m_arlen),
        .m_rvalid  (m_rvalid),  .m_rready  (m_rready),  .m_rdata  (m_rdata),  .m_rid   (m_rid),
        .m_rlast   (m_rlast)
    );

endmodule
