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

// foldsim_vmem_group under one of the setups of main.cpp (VG_CFG_<n>, the
// same table there): its members' widths, rows and places in the image.
`ifdef VG_CFG_1
`define VG_P N = 3, IW = 3, INSTANCES = 8, AW = 7, DW = 512, MW = 64, RDWF = 0, CACHE = 64
`define VG_M MAW [N] = '{6, 5, 4}, MDW [N] = '{512, 64, 26}, MMW [N] = '{64, 8, 1}, MROW [N] = '{0, 64, 96}
`elsif VG_CFG_2
`define VG_P N = 2, IW = 2, INSTANCES = 3, AW = 6, DW = 128, MW = 16, RDWF = 1, CACHE = 32
`define VG_M MAW [N] = '{5, 4}, MDW [N] = '{128, 40}, MMW [N] = '{16, 5}, MROW [N] = '{0, 32}
`elsif VG_CFG_3
`define VG_P N = 4, IW = 4, INSTANCES = 16, AW = 6, DW = 64, MW = 1, RDWF = 0, CACHE = 16
`define VG_M MAW [N] = '{4, 4, 3, 3}, MDW [N] = '{64, 64, 16, 8}, MMW [N] = '{1, 1, 1, 1}, MROW [N] = '{0, 16, 32, 40}
`elsif VG_CFG_4
`define VG_P N = 4, IW = 1, INSTANCES = 1, AW = 8, DW = 512, MW = 64, RDWF = 1, CACHE = 64
`define VG_M MAW [N] = '{6, 6, 6, 6}, MDW [N] = '{512, 512, 512, 512}, MMW [N] = '{64, 64, 64, 64}, MROW [N] = '{0, 64, 128, 192}
`elsif VG_CFG_5
`define VG_P N = 3, IW = 4, INSTANCES = 16, AW = 7, DW = 128, MW = 16, RDWF = 0, CACHE = 64
`define VG_M MAW [N] = '{6, 5, 5}, MDW [N] = '{128, 96, 128}, MMW [N] = '{16, 12, 16}, MROW [N] = '{0, 64, 96}
`define VG_X AXI_ID_W = 1, OUTSTANDING = 32, ROW_BYTES = 16, KZ_SHIFT = 2
`elsif VG_CFG_6
`define VG_P N = 3, IW = 3, INSTANCES = 8, AW = 8, DW = 64, MW = 8, RDWF = 1, CACHE = 0
`define VG_M MAW [N] = '{7, 6, 6}, MDW [N] = '{64, 64, 64}, MMW [N] = '{8, 8, 8}, MROW [N] = '{0, 128, 192}
`define VG_X AXI_ID_W = 1, OUTSTANDING = 64, ROW_BYTES = 8, KZ_SHIFT = 4
`else
`define VG_P N = 3, IW = 3, INSTANCES = 8, AW = 7, DW = 512, MW = 64, RDWF = 1, CACHE = 0
`define VG_M MAW [N] = '{6, 5, 4}, MDW [N] = '{512, 64, 26}, MMW [N] = '{64, 8, 1}, MROW [N] = '{0, 64, 96}
`endif
`ifndef VG_X
`define VG_X AXI_ID_W = 4, OUTSTANDING = 16, ROW_BYTES = 64, KZ_SHIFT = 0
`endif
// VG_SMALL_W: two writes on AXI at once, a two-entry hazard table and 7-bit
// write numbers (every read meets another row's write; numbers wrap).
`ifdef VG_SMALL_W
`define VG_W WRITES_OUT = 2, HAZARD_BITS = 1, SEQ_W = 7
`else
`define VG_W WRITES_OUT = 128, HAZARD_BITS = 11, SEQ_W = 16
`endif

module foldsim_vmem_group_tb #(
    parameter `VG_P,
    parameter int `VG_M,
    parameter `VG_X,
    parameter `VG_W,
    parameter FIFO     = 8,
    localparam GF      = $clog2(2 * FIFO + 1)
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
    output wire                 rsp_valid [N],
    output wire [IW-1:0]        rsp_inst,
    output wire [DW-1:0]        rsp_data,
    output wire [GF-1:0]        free [N],
    input  wire [63:0]          base,
    input  wire                 zero,
    input  wire                 move,
    output wire                 moving,
    output wire                 m_awvalid,
    input  wire                 m_awready,
    output wire [63:0]          m_awaddr,
    output wire [AXI_ID_W-1:0]  m_awid,
    output wire [7:0]           m_awlen,
    output wire                 m_wvalid,
    input  wire                 m_wready,
    output wire [511:0]         m_wdata,
    output wire [63:0]          m_wstrb,
    output wire                 m_wlast,
    input  wire                 m_bvalid,
    output wire                 m_bready,
    input  wire [AXI_ID_W-1:0]  m_bid,
    output wire                 m_arvalid,
    input  wire                 m_arready,
    output wire [63:0]          m_araddr,
    output wire [AXI_ID_W-1:0]  m_arid,
    output wire [7:0]           m_arlen,
    input  wire                 m_rvalid,
    output wire                 m_rready,
    input  wire [511:0]         m_rdata,
    input  wire [AXI_ID_W-1:0]  m_rid,
    input  wire                 m_rlast
);
    // The controller's telemetry (tested by foldsim_vmem's test).
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7*48-1:0] tel;
    /* verilator lint_on UNUSEDSIGNAL */
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
        .RDW_READ_FIRST (RDWF),
        .AXI_ID_W       (AXI_ID_W),
        .OUTSTANDING    (OUTSTANDING),
        .ROW_BYTES      (ROW_BYTES),
        .KZ_SHIFT       (KZ_SHIFT),
        .CACHE_ROWS     (CACHE),
        .WRITES_OUT     (WRITES_OUT),
        .HAZARD_BITS    (HAZARD_BITS),
        .SEQ_W          (SEQ_W),
        .FIFO           (FIFO)
    ) group (.*);
endmodule
