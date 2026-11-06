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

// foldsim_vmem_group with every port registered, for synthesis and the logic
// screen on its own: each path then starts and ends at a flop. Defaults: the
// small GPU's four L2 data stores (4096 rows of 512 bits, one instance).
module foldsim_vmem_group_top #(
    parameter N              = 4,
    parameter IW             = 1,
    parameter INSTANCES      = 1,
    parameter AW             = 14,
    parameter DW             = 512,
    parameter MW             = 64,
    parameter int MAW [N]    = '{12, 12, 12, 12},
    parameter int MDW [N]    = '{512, 512, 512, 512},
    parameter int MMW [N]    = '{64, 64, 64, 64},
    parameter int MROW [N]   = '{0, 4096, 8192, 12288},
    parameter RDW_READ_FIRST = 1,
    parameter AXI_ID_W       = 4,
    parameter OUTSTANDING    = 16,
    parameter CACHE_ROWS     = 1024,
    parameter KZ_SHIFT       = 0,
    parameter ROW_BYTES      = (((DW + 7) / 8) > 64) ? (1 << $clog2((DW + 7) / 8)) : 64,
    parameter FIFO           = 8,
    parameter WRITES_OUT     = 128,
    parameter HAZARD_BITS    = $clog2(32 + WRITES_OUT) + 3,
    parameter SEQ_W          = 16,
    localparam GF            = $clog2(2 * FIFO + 1)
) (
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 rd_valid [N],
    input  wire [IW-1:0]        rd_inst  [N],
    input  wire [AW-1:0]        rd_row   [N],
    input  wire                 wr_valid [N],
    input  wire [IW-1:0]        wr_inst  [N],
    input  wire [AW-1:0]        wr_row   [N],
    input  wire [DW-1:0]        wr_data  [N],
    input  wire [MW-1:0]        wr_mask  [N],
    output reg                  rsp_valid [N],
    output reg  [IW-1:0]        rsp_inst,
    output reg  [DW-1:0]        rsp_data,
    output reg  [GF-1:0]        free [N],
    input  wire [63:0]          base,
    input  wire                 zero,
    input  wire                 move,
    output reg                  moving,
    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [63:0]          m_awaddr,
    output reg  [AXI_ID_W-1:0]  m_awid,
    output reg  [7:0]           m_awlen,
    output reg                  m_wvalid,
    input  wire                 m_wready,
    output reg  [511:0]         m_wdata,
    output reg  [63:0]          m_wstrb,
    output reg                  m_wlast,
    input  wire                 m_bvalid,
    output reg                  m_bready,
    input  wire [AXI_ID_W-1:0]  m_bid,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    output reg  [63:0]          m_araddr,
    output reg  [AXI_ID_W-1:0]  m_arid,
    output reg  [7:0]           m_arlen,
    input  wire                 m_rvalid,
    output reg                  m_rready,
    input  wire [511:0]         m_rdata,
    input  wire [AXI_ID_W-1:0]  m_rid,
    input  wire                 m_rlast
);
    reg                rd_valid_r [N], wr_valid_r [N];
    reg [IW-1:0]       rd_inst_r [N], wr_inst_r [N];
    reg [AW-1:0]       rd_row_r [N], wr_row_r [N];
    reg [DW-1:0]       wr_data_r [N];
    reg [MW-1:0]       wr_mask_r [N];
    reg                reset_r, zero_r, move_r;
    reg [63:0]         base_r;
    reg                awready_r, wready_r, bvalid_r, arready_r, rvalid_r, rlast_r;
    reg [AXI_ID_W-1:0] bid_r, rid_r;
    reg [511:0]        rdata_r;
    always @(posedge clk) begin
        rd_valid_r <= rd_valid;
        rd_inst_r  <= rd_inst;
        rd_row_r   <= rd_row;
        wr_valid_r <= wr_valid;
        wr_inst_r  <= wr_inst;
        wr_row_r   <= wr_row;
        wr_data_r  <= wr_data;
        wr_mask_r  <= wr_mask;
        reset_r    <= reset;
        zero_r     <= zero;
        move_r     <= move;
        base_r     <= base;
        awready_r  <= m_awready;
        wready_r   <= m_wready;
        bvalid_r   <= m_bvalid;
        bid_r      <= m_bid;
        arready_r  <= m_arready;
        rvalid_r   <= m_rvalid;
        rdata_r    <= m_rdata;
        rid_r      <= m_rid;
        rlast_r    <= m_rlast;
    end

    wire                rsp_valid_w [N];
    wire [IW-1:0]       rsp_inst_w;
    wire [DW-1:0]       rsp_data_w;
    wire [GF-1:0]       free_w [N];
    wire                moving_w;
    wire                awvalid_w, wvalid_w, wlast_w, bready_w, arvalid_w, rready_w;
    wire [63:0]         awaddr_w, araddr_w, wstrb_w;
    wire [AXI_ID_W-1:0] awid_w, arid_w;
    wire [7:0]          awlen_w, arlen_w;
    wire [511:0]        wdata_w;

    foldsim_vmem_group #(
        .N              (N),
        .IW             (IW),
        .INSTANCES      (INSTANCES),
        .AW             (AW),
        .DW             (DW),
        .MW             (MW),
        .MAW            (MAW),
        .MDW            (MDW),
        .MMW            (MMW),
        .MROW           (MROW),
        .RDW_READ_FIRST (RDW_READ_FIRST),
        .AXI_ID_W       (AXI_ID_W),
        .OUTSTANDING    (OUTSTANDING),
        .KZ_SHIFT       (KZ_SHIFT),
        .ROW_BYTES      (ROW_BYTES),
        .CACHE_ROWS     (CACHE_ROWS),
        .FIFO           (FIFO),
        .WRITES_OUT     (WRITES_OUT),
        .HAZARD_BITS    (HAZARD_BITS),
        .SEQ_W          (SEQ_W)
    ) group (
        .clk       (clk),
        .reset     (reset_r),
        .rd_valid  (rd_valid_r),
        .rd_inst   (rd_inst_r),
        .rd_row    (rd_row_r),
        .wr_valid  (wr_valid_r),
        .wr_inst   (wr_inst_r),
        .wr_row    (wr_row_r),
        .wr_data   (wr_data_r),
        .wr_mask   (wr_mask_r),
        .rsp_valid (rsp_valid_w),
        .rsp_inst  (rsp_inst_w),
        .rsp_data  (rsp_data_w),
        .free      (free_w),
        .base      (base_r),
        .zero      (zero_r),
        .move      (move_r),
        .moving    (moving_w),
        .m_awvalid (awvalid_w),
        .m_awready (awready_r),
        .m_awaddr  (awaddr_w),
        .m_awid    (awid_w),
        .m_awlen   (awlen_w),
        .m_wvalid  (wvalid_w),
        .m_wready  (wready_r),
        .m_wdata   (wdata_w),
        .m_wstrb   (wstrb_w),
        .m_wlast   (wlast_w),
        .m_bvalid  (bvalid_r),
        .m_bready  (bready_w),
        .m_bid     (bid_r),
        .m_arvalid (arvalid_w),
        .m_arready (arready_r),
        .m_araddr  (araddr_w),
        .m_arid    (arid_w),
        .m_arlen   (arlen_w),
        .m_rvalid  (rvalid_r),
        .m_rready  (rready_w),
        .m_rdata   (rdata_r),
        .m_rid     (rid_r),
        .m_rlast   (rlast_r)
    );

    always @(posedge clk) begin
        rsp_valid <= rsp_valid_w;
        rsp_inst  <= rsp_inst_w;
        rsp_data  <= rsp_data_w;
        free      <= free_w;
        moving    <= moving_w;
        m_awvalid <= awvalid_w;
        m_awaddr  <= awaddr_w;
        m_awid    <= awid_w;
        m_awlen   <= awlen_w;
        m_wvalid  <= wvalid_w;
        m_wdata   <= wdata_w;
        m_wstrb   <= wstrb_w;
        m_wlast   <= wlast_w;
        m_bready  <= bready_w;
        m_arvalid <= arvalid_w;
        m_araddr  <= araddr_w;
        m_arid    <= arid_w;
        m_arlen   <= arlen_w;
        m_rready  <= rready_w;
    end

endmodule
