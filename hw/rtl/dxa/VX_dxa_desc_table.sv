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
    localparam DESC_WORD_W = DXA_DESC_WORD_W;
    localparam NUM_SLOTS   = `VX_DCR_DXA_DESC_COUNT;
    localparam NUM_WORDS   = `VX_DCR_DXA_DESC_STRIDE;

    // ── DCR write decode ──
    wire dcr_dxa_desc_write;
    wire [VX_DCR_ADDR_WIDTH-1:0] dcr_desc_off;
    wire [31:0] dcr_desc_off_w;
    wire [DESC_SLOT_W-1:0] dcr_desc_slot;
    wire [DESC_WORD_W-1:0] dcr_desc_word;

    assign dcr_dxa_desc_write = dcr_bus_if.req_valid
                             && dcr_bus_if.req_data.rw
                             && (dcr_bus_if.req_data.addr >= `VX_DCR_DXA_DESC_BASE)
                             && (dcr_bus_if.req_data.addr < (`VX_DCR_DXA_DESC_BASE + (NUM_SLOTS * NUM_WORDS)));

    assign dcr_desc_off = dcr_bus_if.req_data.addr - `VX_DCR_DXA_DESC_BASE;
    assign dcr_desc_off_w = 32'(dcr_desc_off);
    assign dcr_desc_slot = DESC_SLOT_W'(dcr_desc_off_w / NUM_WORDS);
    assign dcr_desc_word = DESC_WORD_W'(dcr_desc_off_w % NUM_WORDS);

    // ── Write-enable mask: one bit per 32-bit word in the descriptor ──
    wire [NUM_WORDS-1:0] wren;
    for (genvar w = 0; w < NUM_WORDS; ++w) begin : g_wren
        assign wren[w] = dcr_dxa_desc_write && (dcr_desc_word == DESC_WORD_W'(w));
    end

    // ── Write data: replicate the 32-bit DCR data across all word slots ──
    wire [DXA_DESC_DATAW-1:0] wdata;
    for (genvar w = 0; w < NUM_WORDS; ++w) begin : g_wdata
        assign wdata[w*32 +: 32] = dcr_bus_if.req_data.data;
    end

    // ── One BRAM per read port (dual-port: DCR writes, worker reads) ──
    for (genvar i = 0; i < NUM_READ_PORTS; ++i) begin : g_read
        wire [DXA_DESC_DATAW-1:0] rdata_raw;

        VX_dp_ram #(
            .DATAW     (DXA_DESC_DATAW),
            .SIZE      (NUM_SLOTS),
            .WRENW     (NUM_WORDS),
            .RDW_MODE  ("R"),
            .RADDR_REG (1)
        ) desc_store (
            .clk   (clk),
            .reset (reset),
            .read  (1'b1),
            .write (dcr_dxa_desc_write),
            .wren  (wren),
            .waddr (dcr_desc_slot),
            .wdata (wdata),
            .raddr (read_desc_slot[i]),
            .rdata (rdata_raw)
        );

        // ── Unpack BRAM output into descriptor struct ──
        dxa_desc_t desc;
        assign desc = rdata_raw;

        assign read_base_addr[i]   = `MEM_ADDR_WIDTH'({desc.base_hi, desc.base_lo});
        assign read_desc_meta[i]   = desc.meta;
        assign read_desc_tile01[i] = desc.tilesize01;
        assign read_desc_tile23[i] = desc.tilesize23;
        assign read_desc_tile4[i]  = desc.tilesize4;
        assign read_desc_cfill[i]  = desc.cfill;
        assign read_size0[i]       = desc.size0;
        assign read_size1[i]       = desc.size1;
        assign read_stride0[i]     = desc.stride0;
    `ifdef EXT_DXA_MULTICAST_ENABLE
        assign read_smem_stride[i] = desc.smem_stride;
        assign read_bar_stride[i]  = desc.bar_stride;
    `endif

        `UNUSED_VAR (desc._pad0)
        `UNUSED_VAR (desc.estride0)
        `UNUSED_VAR (desc.estride1)
        `UNUSED_VAR (desc.estride2)
        `UNUSED_VAR (desc.estride3)
        `UNUSED_VAR (desc.estride4)
        `UNUSED_VAR (desc.size2)
        `UNUSED_VAR (desc.size3)
        `UNUSED_VAR (desc.size4)
        `UNUSED_VAR (desc.stride1)
        `UNUSED_VAR (desc.stride2)
        `UNUSED_VAR (desc.stride3)
    end

    assign dcr_bus_if.rsp_valid = '0;
    assign dcr_bus_if.rsp_data  = '0;

    `UNUSED_VAR (reset)

endmodule
