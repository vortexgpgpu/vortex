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

// DXA Descriptor Table — BRAM-based, single read port.
// DCR writes update individual 32-bit words via byte-enable.

`include "VX_define.vh"

module VX_dxa_desc_table import VX_gpu_pkg::*, VX_dxa_pkg::*; (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    input wire [DXA_DESC_SLOT_W-1:0] read_addr,
    output dxa_desc_t read_desc
);
    localparam NUM_SLOTS = `VX_DCR_DXA_DESC_COUNT;
    localparam STRIDE    = `VX_DCR_DXA_DESC_STRIDE;
    localparam ENTRY_W   = STRIDE * 32;

    // ---- DCR write logic ----
    wire dcr_write = dcr_bus_if.req_valid
                  && dcr_bus_if.req_data.rw
                  && (dcr_bus_if.req_data.addr >= `VX_DCR_DXA_DESC_BASE)
                  && (dcr_bus_if.req_data.addr < (`VX_DCR_DXA_DESC_BASE + (NUM_SLOTS * STRIDE)));

    wire [31:0] dcr_off = 32'(dcr_bus_if.req_data.addr - `VX_DCR_DXA_DESC_BASE);
    wire [DXA_DESC_SLOT_W-1:0]  dcr_slot = DXA_DESC_SLOT_W'(dcr_off / STRIDE);
    wire [`CLOG2(STRIDE)-1:0]   dcr_word = `CLOG2(STRIDE)'(dcr_off % STRIDE);

    wire [STRIDE-1:0] dcr_wren;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wren
        assign dcr_wren[j] = dcr_write && (dcr_word == `CLOG2(STRIDE)'(j));
    end

    wire [ENTRY_W-1:0] dcr_wdata;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wdata
        assign dcr_wdata[j*32 +: 32] = dcr_bus_if.req_data.data;
    end

    // ---- BRAM storage ----
    wire [ENTRY_W-1:0] entry_rdata;
    `UNUSED_VAR (entry_rdata)

    VX_dp_ram #(
        .DATAW    (ENTRY_W),
        .SIZE     (NUM_SLOTS),
        .WRENW    (STRIDE),
        .OUT_REG  (0),
        .RDW_MODE ("W")
    ) desc_store (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (dcr_write),
        .wren  (dcr_wren),
        .raddr (read_addr),
        .waddr (dcr_slot),
        .wdata (dcr_wdata),
        .rdata (entry_rdata)
    );

    // Unpack fields
    assign read_desc.base_addr = `MEM_ADDR_WIDTH'({
        entry_rdata[`VX_DCR_DXA_DESC_BASE_HI_OFF*32 +: 32],
        entry_rdata[`VX_DCR_DXA_DESC_BASE_LO_OFF*32 +: 32]
    });
    assign read_desc.meta    = entry_rdata[`VX_DCR_DXA_DESC_META_OFF*32 +: 32];
    assign read_desc.tile01  = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE01_OFF*32 +: 32];
    assign read_desc.tile23  = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE23_OFF*32 +: 32];
    assign read_desc.tile4   = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE4_OFF*32 +: 32];
    assign read_desc.cfill   = entry_rdata[`VX_DCR_DXA_DESC_CFILL_OFF*32 +: 32];
    assign read_desc.size0   = entry_rdata[`VX_DCR_DXA_DESC_SIZE0_OFF*32 +: 32];
    assign read_desc.size1   = entry_rdata[`VX_DCR_DXA_DESC_SIZE1_OFF*32 +: 32];
    assign read_desc.size2   = entry_rdata[`VX_DCR_DXA_DESC_SIZE2_OFF*32 +: 32];
    assign read_desc.size3   = entry_rdata[`VX_DCR_DXA_DESC_SIZE3_OFF*32 +: 32];
    assign read_desc.size4   = entry_rdata[`VX_DCR_DXA_DESC_SIZE4_OFF*32 +: 32];
    assign read_desc.stride0 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE0_OFF*32 +: 32];
    assign read_desc.stride1 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE1_OFF*32 +: 32];
    assign read_desc.stride2 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE2_OFF*32 +: 32];
    assign read_desc.stride3 = entry_rdata[`VX_DCR_DXA_DESC_STRIDE3_OFF*32 +: 32];
    assign read_desc.smem_stride = entry_rdata[`VX_DCR_DXA_DESC_SMEM_STRIDE_OFF*32 +: 32];

    assign dcr_bus_if.rsp_valid = '0;
    assign dcr_bus_if.rsp_data  = '0;

    `UNUSED_VAR (reset)

endmodule
