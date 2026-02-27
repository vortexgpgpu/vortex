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
);
    localparam DESC_SLOT_W = DXA_DESC_SLOT_W;
    localparam DESC_WORD_W = DXA_DESC_WORD_W;

    wire dcr_dxa_desc_write;
    wire [VX_DCR_ADDR_WIDTH-1:0] dcr_desc_off;
    wire [31:0] dcr_desc_off_w;
    wire [DESC_SLOT_W-1:0] dcr_desc_slot;
    wire [DESC_WORD_W-1:0] dcr_desc_word;

    reg [`VX_DCR_DXA_DESC_COUNT-1:0][`VX_DCR_DXA_DESC_STRIDE-1:0][31:0] dxa_desc_r;

    assign dcr_dxa_desc_write = dcr_bus_if.write_valid
                             && (dcr_bus_if.write_addr >= `VX_DCR_DXA_DESC_BASE)
                             && (dcr_bus_if.write_addr < (`VX_DCR_DXA_DESC_BASE + (`VX_DCR_DXA_DESC_COUNT * `VX_DCR_DXA_DESC_STRIDE)));

    assign dcr_desc_off = dcr_bus_if.write_addr - `VX_DCR_DXA_DESC_BASE;
    assign dcr_desc_off_w = 32'(dcr_desc_off);
    assign dcr_desc_slot = DESC_SLOT_W'(dcr_desc_off_w / `VX_DCR_DXA_DESC_STRIDE);
    assign dcr_desc_word = DESC_WORD_W'(dcr_desc_off_w % `VX_DCR_DXA_DESC_STRIDE);

    always @(posedge clk) begin
        if (dcr_dxa_desc_write) begin
            dxa_desc_r[dcr_desc_slot][dcr_desc_word] <= dcr_bus_if.write_data;
        end
    end

    `UNUSED_VAR (reset)

    for (genvar i = 0; i < NUM_READ_PORTS; ++i) begin : g_read
        wire [DESC_SLOT_W-1:0] slot = read_desc_slot[i];
        assign read_base_addr[i] = `MEM_ADDR_WIDTH'({
            dxa_desc_r[slot][`VX_DCR_DXA_DESC_BASE_HI_OFF],
            dxa_desc_r[slot][`VX_DCR_DXA_DESC_BASE_LO_OFF]
        });
        assign read_desc_meta[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_META_OFF];
        assign read_desc_tile01[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_TILESIZE01_OFF];
        assign read_desc_tile23[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_TILESIZE23_OFF];
        assign read_desc_tile4[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_TILESIZE4_OFF];
        assign read_desc_cfill[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_CFILL_OFF];
        assign read_size0[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_SIZE0_OFF];
        assign read_size1[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_SIZE1_OFF];
        assign read_stride0[i] = dxa_desc_r[slot][`VX_DCR_DXA_DESC_STRIDE0_OFF];
    end

endmodule
