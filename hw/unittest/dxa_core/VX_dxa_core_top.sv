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

// Flat-port testbench wrapper around VX_dxa_core.
// Exposes all interface signals as individual logic ports for Verilator.

module VX_dxa_core_top import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",
    parameter GMEM_OUT_PORTS  = `VX_CFG_DXA_MEM_PORTS,
    parameter ENABLE          = 1,
    // gmem bus geometry (matches VX_mem_bus_if defaults for L1_LINE_SIZE)
    parameter GMEM_LINE_SIZE  = `VX_CFG_L1_LINE_SIZE,
    parameter GMEM_TAG_WIDTH  = L1_MEM_ARB_TAG_WIDTH,
    parameter GMEM_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE)
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_transfers,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_reads,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_dedup,
    output wire [PERF_CTR_BITS-1:0] perf_lmem_writes,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_latency,
`endif

    output wire busy,

    // -----------------------------------------------------------------------
    // DCR configuration bus (slave)
    // -----------------------------------------------------------------------
    input  wire                          dcr_req_valid,
    input  wire                          dcr_req_rw,
    input  wire [VX_DCR_ADDR_WIDTH-1:0]  dcr_req_addr,
    input  wire [VX_DCR_DATA_WIDTH-1:0]  dcr_req_data,
    output wire                          dcr_rsp_valid,
    output wire [VX_DCR_DATA_WIDTH-1:0]  dcr_rsp_data,

    // -----------------------------------------------------------------------
    // DXA request bus (slave, 1 socket)
    // -----------------------------------------------------------------------
    input  wire                          dxa_req_valid,
    input  wire [NC_WIDTH-1:0]           dxa_req_core_id,
    input  wire [UUID_WIDTH-1:0]         dxa_req_uuid,
    input  wire [NW_WIDTH-1:0]           dxa_req_wid,
    input  wire [DXA_SMEM_ADDR_W-1:0]    dxa_req_smem_addr,
    input  wire [31:0]                   dxa_req_meta,
    input  wire [31:0]                   dxa_req_coord0,
    input  wire [31:0]                   dxa_req_coord1,
    input  wire [31:0]                   dxa_req_coord2,
    input  wire [31:0]                   dxa_req_coord3,
    input  wire [31:0]                   dxa_req_coord4,
    input  wire [`VX_CFG_NUM_WARPS-1:0] dxa_req_cta_mask,
    output wire                          dxa_req_ready,

    // -----------------------------------------------------------------------
    // LMEM bank-write bus (one engine output, switched by VX_socket in-system)
    // -----------------------------------------------------------------------
    output wire                          lmem_req_valid,
    output wire                          lmem_req_rw,
    output wire [DXA_LMEM_ADDR_W-1:0]    lmem_req_addr,
    output wire [31:0]                   lmem_req_data_lo,
    output wire [15:0]                   lmem_req_byteen_count,
    output wire                          lmem_req_last,
    input  wire                          lmem_req_ready,

    // -----------------------------------------------------------------------
    // Global memory bus (master, GMEM_OUT_PORTS ports)
    // Individual fields; widths from GMEM_LINE_SIZE / GMEM_TAG_WIDTH params.
    // -----------------------------------------------------------------------
    output wire [GMEM_OUT_PORTS-1:0]                            gmem_req_valid,
    output wire [GMEM_OUT_PORTS-1:0]                            gmem_req_rw,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_LINE_SIZE-1:0]        gmem_req_byteen,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_ADDR_WIDTH-1:0]       gmem_req_addr,
    output wire [GMEM_OUT_PORTS-1:0][MEM_ATTR_WIDTH-1:0]        gmem_req_attr,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_LINE_SIZE*8-1:0]      gmem_req_data,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_TAG_WIDTH-1:0]        gmem_req_tag,
    input  wire [GMEM_OUT_PORTS-1:0]                            gmem_req_ready,
    input  wire [GMEM_OUT_PORTS-1:0]                            gmem_rsp_valid,
    input  wire [GMEM_OUT_PORTS-1:0][GMEM_LINE_SIZE*8-1:0]      gmem_rsp_data,
    input  wire [GMEM_OUT_PORTS-1:0][GMEM_TAG_WIDTH-1:0]        gmem_rsp_tag,
    output wire [GMEM_OUT_PORTS-1:0]                            gmem_rsp_ready
);

    // -----------------------------------------------------------------------
    // Interface instances
    // -----------------------------------------------------------------------

    VX_dcr_bus_if dcr_bus_if();

    assign dcr_bus_if.req_valid      = dcr_req_valid;
    assign dcr_bus_if.req_data.rw    = dcr_req_rw;
    assign dcr_bus_if.req_data.addr  = dcr_req_addr;
    assign dcr_bus_if.req_data.data  = dcr_req_data;
    assign dcr_rsp_valid             = dcr_bus_if.rsp_valid;
    assign dcr_rsp_data              = dcr_bus_if.rsp_data.data;

    VX_dxa_req_bus_if req_bus_if[`VX_CFG_SOCKET_SIZE]();

    dxa_req_data_t dxa_req_data;
    always_comb begin
        dxa_req_data = '0;
        dxa_req_data.core_id   = dxa_req_core_id;
        dxa_req_data.uuid      = dxa_req_uuid;
        dxa_req_data.wid       = dxa_req_wid;
        dxa_req_data.smem_addr = dxa_req_smem_addr;
        dxa_req_data.meta      = dxa_req_meta;
        dxa_req_data.coords[0] = dxa_req_coord0;
        dxa_req_data.coords[1] = dxa_req_coord1;
        dxa_req_data.coords[2] = dxa_req_coord2;
        dxa_req_data.coords[3] = dxa_req_coord3;
        dxa_req_data.coords[4] = dxa_req_coord4;
        dxa_req_data.cta_mask  = dxa_req_cta_mask;
    end

    assign req_bus_if[0].req_valid = dxa_req_valid;
    assign req_bus_if[0].req_data  = dxa_req_data;
    assign dxa_req_ready           = req_bus_if[0].req_ready;
    for (genvar i = 1; i < `VX_CFG_SOCKET_SIZE; ++i) begin : g_unused_req
        assign req_bus_if[i].req_valid = 1'b0;
        assign req_bus_if[i].req_data  = '0;
    end

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_LMEM_WORD_SIZE),
        .TAG_WIDTH   (DXA_LMEM_OUT_TAG_W),
        .ATTR_WIDTH (DXA_LMEM_ATTR_W),
        .ADDR_WIDTH  (DXA_LMEM_ADDR_W)
    ) lmem_bus_if[1]();

    assign lmem_req_valid              = lmem_bus_if[0].req_valid;
    assign lmem_req_rw                 = lmem_bus_if[0].req_data.rw;
    assign lmem_req_addr               = lmem_bus_if[0].req_data.addr;
    assign lmem_req_data_lo            = lmem_bus_if[0].req_data.data[31:0];
    assign lmem_req_byteen_count       = 16'($countones(lmem_bus_if[0].req_data.byteen));
    assign lmem_req_last               = lmem_bus_if[0].req_data.attr[DXA_LMEM_ATTR_W-1];
    assign lmem_bus_if[0].req_ready    = lmem_req_ready;
    assign lmem_bus_if[0].rsp_valid    = 1'b0;
    assign lmem_bus_if[0].rsp_data     = '0;

    VX_mem_bus_if #(
        .DATA_SIZE (GMEM_LINE_SIZE),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) gmem_bus_if[GMEM_OUT_PORTS]();

    for (genvar i = 0; i < GMEM_OUT_PORTS; i++) begin
        assign gmem_req_valid[i]           = gmem_bus_if[i].req_valid;
        assign gmem_req_rw[i]              = gmem_bus_if[i].req_data.rw;
        assign gmem_req_byteen[i]          = gmem_bus_if[i].req_data.byteen;
        assign gmem_req_addr[i]            = gmem_bus_if[i].req_data.addr;
        assign gmem_req_attr[i]           = gmem_bus_if[i].req_data.attr;
        assign gmem_req_data[i]            = gmem_bus_if[i].req_data.data;
        assign gmem_req_tag[i]             = gmem_bus_if[i].req_data.tag;
        assign gmem_bus_if[i].req_ready    = gmem_req_ready[i];
        assign gmem_bus_if[i].rsp_valid    = gmem_rsp_valid[i];
        assign gmem_bus_if[i].rsp_data.data= gmem_rsp_data[i];
        assign gmem_bus_if[i].rsp_data.tag = gmem_rsp_tag[i];
        assign gmem_rsp_ready[i]           = gmem_bus_if[i].rsp_ready;
    end

    // -----------------------------------------------------------------------
    // DUT
    // -----------------------------------------------------------------------

`ifdef PERF_ENABLE
    dxa_perf_t dxa_perf;
    assign perf_transfers   = dxa_perf.transfers;
    assign perf_gmem_reads  = dxa_perf.gmem_reads;
    assign perf_gmem_dedup  = dxa_perf.gmem_dedup;
    assign perf_lmem_writes = dxa_perf.lmem_writes;
    assign perf_gmem_latency= dxa_perf.gmem_latency;
`endif

    `UNUSED_PARAM (ENABLE)

    VX_dxa_core #(
        .INSTANCE_ID     (INSTANCE_ID),
        .NUM_REQS        (`VX_CFG_SOCKET_SIZE),
        .GMEM_OUT_PORTS  (GMEM_OUT_PORTS)
    ) dxa_core (
        .clk               (clk),
        .reset             (reset),
    `ifdef PERF_ENABLE
        .dxa_perf          (dxa_perf),
    `endif
        .dcr_bus_if        (dcr_bus_if),
        .req_bus_if        (req_bus_if),
        .smem_bus_if       (lmem_bus_if),
        .gmem_bus_if       (gmem_bus_if),
        .busy              (busy)
    );

endmodule
