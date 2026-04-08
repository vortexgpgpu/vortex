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
// Exposes all interface signals as individual logic ports for Verilator
// and FPGA/ASIC synthesis.  Configured for a single-socket, single-unit
// cluster with one global-memory output port.

module VX_dxa_core_top import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID    = "",
    parameter         NUM_DXA_UNITS  = 1,
    parameter         GMEM_OUT_PORTS = 1,
    parameter         CORE_LOCAL_BITS = 0,
    parameter         ENABLE          = 1,
    // gmem bus geometry (matches VX_mem_bus_if defaults for L1_LINE_SIZE)
    parameter         GMEM_LINE_SIZE  = `L1_LINE_SIZE,
    parameter         GMEM_TAG_WIDTH  = L1_MEM_ARB_TAG_WIDTH,
    parameter         GMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - `CLOG2(GMEM_LINE_SIZE)
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] perf_transfers,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_reads,
    output wire [PERF_CTR_BITS-1:0] perf_gmem_dedup,
    output wire [PERF_CTR_BITS-1:0] perf_smem_writes,
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
    input  wire [DXA_REQ_DATAW-1:0]      dxa_req_data,
    output wire                          dxa_req_ready,

    // -----------------------------------------------------------------------
    // SMEM bank-write bus (master, SOCKET_SIZE ports)
    // -----------------------------------------------------------------------
    output wire [`SOCKET_SIZE-1:0]                                              smem_req_valid,
    output wire [`SOCKET_SIZE-1:0]                                              smem_req_rw,
    output wire [`SOCKET_SIZE-1:0][DXA_SMEM_BANK_ADDR_WIDTH-1:0]              smem_req_addr,
    output wire [`SOCKET_SIZE-1:0][DXA_SMEM_WORD_SIZE*8-1:0]                  smem_req_data,
    output wire [`SOCKET_SIZE-1:0][DXA_SMEM_WORD_SIZE-1:0]                    smem_req_byteen,
    output wire [`SOCKET_SIZE-1:0][DXA_SMEM_FLAGS_WIDTH-1:0]                  smem_req_flags,
    input  wire [`SOCKET_SIZE-1:0]                                              smem_req_ready,

    // -----------------------------------------------------------------------
    // Global memory bus (master, GMEM_OUT_PORTS ports)
    // Individual fields; widths from GMEM_LINE_SIZE / GMEM_TAG_WIDTH params.
    // -----------------------------------------------------------------------
    output wire [GMEM_OUT_PORTS-1:0]                            gmem_req_valid,
    output wire [GMEM_OUT_PORTS-1:0]                            gmem_req_rw,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_LINE_SIZE-1:0]        gmem_req_byteen,
    output wire [GMEM_OUT_PORTS-1:0][GMEM_ADDR_WIDTH-1:0]       gmem_req_addr,
    output wire [GMEM_OUT_PORTS-1:0][MEM_FLAGS_WIDTH-1:0]       gmem_req_flags,
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

    VX_dxa_req_bus_if req_bus_if[1]();

    assign req_bus_if[0].req_valid = dxa_req_valid;
    assign req_bus_if[0].req_data  = dxa_req_data;
    assign dxa_req_ready           = req_bus_if[0].req_ready;

    VX_mem_bus_if #(
        .DATA_SIZE   (DXA_SMEM_WORD_SIZE),
        .TAG_WIDTH   (SMEM_DMA_TAG_W),
        .FLAGS_WIDTH (DXA_SMEM_FLAGS_WIDTH),
        .ADDR_WIDTH  (DXA_SMEM_BANK_ADDR_WIDTH)
    ) smem_bus_if[`SOCKET_SIZE]();

    for (genvar i = 0; i < `SOCKET_SIZE; i++) begin
        assign smem_req_valid[i]         = smem_bus_if[i].req_valid;
        assign smem_req_rw[i]            = smem_bus_if[i].req_data.rw;
        assign smem_req_addr[i]          = smem_bus_if[i].req_data.addr;
        assign smem_req_data[i]          = smem_bus_if[i].req_data.data;
        assign smem_req_byteen[i]        = smem_bus_if[i].req_data.byteen;
        assign smem_req_flags[i]         = smem_bus_if[i].req_data.flags;
        assign smem_bus_if[i].req_ready  = smem_req_ready[i];
        assign smem_bus_if[i].rsp_valid  = 1'b0;
        assign smem_bus_if[i].rsp_data   = '0;
    end

    VX_mem_bus_if #(
        .DATA_SIZE (GMEM_LINE_SIZE),
        .TAG_WIDTH (GMEM_TAG_WIDTH)
    ) gmem_bus_if[GMEM_OUT_PORTS]();

    for (genvar i = 0; i < GMEM_OUT_PORTS; i++) begin
        assign gmem_req_valid[i]           = gmem_bus_if[i].req_valid;
        assign gmem_req_rw[i]              = gmem_bus_if[i].req_data.rw;
        assign gmem_req_byteen[i]          = gmem_bus_if[i].req_data.byteen;
        assign gmem_req_addr[i]            = gmem_bus_if[i].req_data.addr;
        assign gmem_req_flags[i]           = gmem_bus_if[i].req_data.flags;
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
    assign perf_smem_writes = dxa_perf.smem_writes;
    assign perf_gmem_latency= dxa_perf.gmem_latency;
`endif

    `UNUSED_PARAM (ENABLE)

    VX_dxa_core #(
        .INSTANCE_ID     (INSTANCE_ID),
        .DXA_NUM_SOCKETS (1),
        .NUM_DXA_UNITS   (NUM_DXA_UNITS),
        .GMEM_OUT_PORTS  (GMEM_OUT_PORTS),
        .CORE_LOCAL_BITS (CORE_LOCAL_BITS)
    ) dxa_core (
        .clk               (clk),
        .reset             (reset),
    `ifdef PERF_ENABLE
        .dxa_perf          (dxa_perf),
    `endif
        .dcr_bus_if        (dcr_bus_if),
        .req_bus_if        (req_bus_if),
        .smem_bus_if       (smem_bus_if),
        .gmem_bus_if       (gmem_bus_if),
        .busy              (busy)
    );

endmodule
