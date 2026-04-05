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

// DXA Descriptor Table — BRAM-based with registered output.
// DCR writes update individual 32-bit words via byte-enable.
// Reads have 1-cycle latency (OUT_REG=1) to break timing path.

`include "VX_define.vh"

module VX_dxa_desc_table import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter NUM_READ_PORTS = 1
) (
    input wire clk,
    input wire reset,

    VX_dcr_bus_if.slave dcr_bus_if,

    input wire [NUM_READ_PORTS-1:0][DXA_DESC_SLOT_W-1:0] read_desc_slot,

    output wire [NUM_READ_PORTS-1:0][`MEM_ADDR_WIDTH-1:0] read_base_addr,
    output wire [NUM_READ_PORTS-1:0][31:0] read_desc_meta,
    output wire [NUM_READ_PORTS-1:0][31:0] read_desc_tile01,
    output wire [NUM_READ_PORTS-1:0][31:0] read_desc_tile23,
    output wire [NUM_READ_PORTS-1:0][31:0] read_desc_tile4,
    output wire [NUM_READ_PORTS-1:0][31:0] read_desc_cfill,
    output wire [NUM_READ_PORTS-1:0][31:0] read_size0,
    output wire [NUM_READ_PORTS-1:0][31:0] read_size1,
    output wire [NUM_READ_PORTS-1:0][31:0] read_stride0
`ifdef EXT_DXA_MULTICAST_ENABLE
    ,
    output wire [NUM_READ_PORTS-1:0][31:0] read_smem_stride,
    output wire [NUM_READ_PORTS-1:0][31:0] read_bar_stride
`endif
);
    localparam DESC_SLOT_W = DXA_DESC_SLOT_W;
    localparam NUM_SLOTS   = `VX_DCR_DXA_DESC_COUNT;
    localparam STRIDE      = `VX_DCR_DXA_DESC_STRIDE;

    // Pack all fields per slot into one wide entry for BRAM storage.
    // Layout: {stride0, size1, size0, cfill, tile4, tile23, tile01, meta, base_hi, base_lo, [multicast fields]}
    // Each field is 32 bits. Total = STRIDE * 32 bits per slot.
    localparam ENTRY_W     = STRIDE * 32;

    // ---- DCR write logic ----
    wire dcr_dxa_desc_write;
    wire [VX_DCR_ADDR_WIDTH-1:0] dcr_desc_off;
    wire [31:0] dcr_desc_off_w;
    wire [DESC_SLOT_W-1:0] dcr_desc_slot;
    wire [`CLOG2(STRIDE)-1:0] dcr_desc_word;

    assign dcr_dxa_desc_write = dcr_bus_if.req_valid
                             && dcr_bus_if.req_data.rw
                             && (dcr_bus_if.req_data.addr >= `VX_DCR_DXA_DESC_BASE)
                             && (dcr_bus_if.req_data.addr < (`VX_DCR_DXA_DESC_BASE + (NUM_SLOTS * STRIDE)));

    assign dcr_desc_off = dcr_bus_if.req_data.addr - `VX_DCR_DXA_DESC_BASE;
    assign dcr_desc_off_w = 32'(dcr_desc_off);
    assign dcr_desc_slot = DESC_SLOT_W'(dcr_desc_off_w / STRIDE);
    assign dcr_desc_word = `CLOG2(STRIDE)'(dcr_desc_off_w % STRIDE);

    // Build write-enable mask: one bit per 32-bit word in the entry
    wire [STRIDE-1:0] dcr_wren;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wren
        assign dcr_wren[j] = dcr_dxa_desc_write && (dcr_desc_word == `CLOG2(STRIDE)'(j));
    end

    // Build write data: replicate the 32-bit DCR data across all word positions
    wire [ENTRY_W-1:0] dcr_wdata;
    for (genvar j = 0; j < STRIDE; ++j) begin : g_wdata
        assign dcr_wdata[j*32 +: 32] = dcr_bus_if.req_data.data;
    end

    // ---- BRAM-based storage with registered output ----
    /* verilator lint_off UNUSEDSIGNAL */
    for (genvar i = 0; i < NUM_READ_PORTS; ++i) begin : g_read
        wire [ENTRY_W-1:0] entry_rdata;

        VX_dp_ram #(
            .DATAW    (ENTRY_W),
            .SIZE     (NUM_SLOTS),
            .WRENW    (STRIDE),
            .OUT_REG  (1),
            .RDW_MODE ("W")
        ) desc_store (
            .clk   (clk),
            .reset (reset),
            .read  (1'b1),
            .write (dcr_dxa_desc_write),
            .wren  (dcr_wren),
            .raddr (read_desc_slot[i]),
            .waddr (dcr_desc_slot),
            .wdata (dcr_wdata),
            .rdata (entry_rdata)
        );

        // Unpack fields from the BRAM entry (registered output)
        assign read_base_addr[i] = `MEM_ADDR_WIDTH'({
            entry_rdata[`VX_DCR_DXA_DESC_BASE_HI_OFF*32 +: 32],
            entry_rdata[`VX_DCR_DXA_DESC_BASE_LO_OFF*32 +: 32]
        });
        assign read_desc_meta[i]  = entry_rdata[`VX_DCR_DXA_DESC_META_OFF*32 +: 32];
        assign read_desc_tile01[i]= entry_rdata[`VX_DCR_DXA_DESC_TILESIZE01_OFF*32 +: 32];
        assign read_desc_tile23[i]= entry_rdata[`VX_DCR_DXA_DESC_TILESIZE23_OFF*32 +: 32];
        assign read_desc_tile4[i] = entry_rdata[`VX_DCR_DXA_DESC_TILESIZE4_OFF*32 +: 32];
        assign read_desc_cfill[i] = entry_rdata[`VX_DCR_DXA_DESC_CFILL_OFF*32 +: 32];
        assign read_size0[i]      = entry_rdata[`VX_DCR_DXA_DESC_SIZE0_OFF*32 +: 32];
        assign read_size1[i]      = entry_rdata[`VX_DCR_DXA_DESC_SIZE1_OFF*32 +: 32];
        assign read_stride0[i]    = entry_rdata[`VX_DCR_DXA_DESC_STRIDE0_OFF*32 +: 32];
    `ifdef EXT_DXA_MULTICAST_ENABLE
        assign read_smem_stride[i]= entry_rdata[`VX_DCR_DXA_DESC_SMEM_STRIDE_OFF*32 +: 32];
        assign read_bar_stride[i] = entry_rdata[`VX_DCR_DXA_DESC_BAR_STRIDE_OFF*32 +: 32];
    `endif
    end
    /* verilator lint_on UNUSEDSIGNAL */

    assign dcr_bus_if.rsp_valid = '0;
    assign dcr_bus_if.rsp_data  = '0;

    `UNUSED_VAR (reset)

endmodule
