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

module VX_dxa_issue_state import VX_gpu_pkg::*; #(
    parameter DXA_CTX_COUNT = (`NUM_CORES * `NUM_WARPS),
    parameter DXA_CTX_BITS = `UP(`CLOG2(DXA_CTX_COUNT)),
    parameter DXA_DESC_SLOT_BITS = `CLOG2(`VX_DCR_DXA_DESC_COUNT),
    parameter DXA_DESC_SLOT_W = `UP(DXA_DESC_SLOT_BITS),
    parameter DXA_DESC_WORD_BITS = `CLOG2(`VX_DCR_DXA_DESC_STRIDE),
    parameter DXA_DESC_WORD_W = `UP(DXA_DESC_WORD_BITS)
) (
    input wire clk,
    input wire reset,

    input wire req_fire,
    input wire [2:0] req_op,
    input wire [DXA_CTX_BITS-1:0] req_ctx_idx,
    input wire [`XLEN-1:0] req_rs1,
    input wire [`XLEN-1:0] req_rs2,
    input wire [BAR_ADDR_W-1:0] req_bar_addr,

    VX_dcr_bus_if.slave dcr_bus_if,

    output wire [BAR_ADDR_W-1:0] issue_bar_addr,
    output wire [DXA_DESC_SLOT_W-1:0] issue_desc_slot,
    output wire [`XLEN-1:0] issue_smem_addr,
    output wire [`XLEN-1:0] issue_flags,
    output wire [4:0][`XLEN-1:0] issue_coords,

    output wire [`MEM_ADDR_WIDTH-1:0] issue_base_addr,
    output wire [31:0] issue_desc_meta,
    output wire [31:0] issue_desc_tile01,
    output wire [31:0] issue_desc_tile23,
    output wire [31:0] issue_desc_tile4,
    output wire [31:0] issue_desc_cfill,
    output wire [31:0] issue_size0,
    output wire [31:0] issue_size1,
    output wire [31:0] issue_stride0
);
    localparam DXA_OP_SETUP0  = 3'd0;
    localparam DXA_OP_SETUP1  = 3'd1;
    localparam DXA_OP_COORD01 = 3'd2;
    localparam DXA_OP_COORD23 = 3'd3;
    localparam DXA_OP_ISSUE   = 3'd4;

    wire dcr_dxa_desc_write;
    wire [VX_DCR_ADDR_WIDTH-1:0] dcr_desc_off;
    wire [31:0] dcr_desc_off_w;
    wire [DXA_DESC_SLOT_W-1:0] dcr_desc_slot;
    wire [DXA_DESC_WORD_W-1:0] dcr_desc_word;

    reg [DXA_CTX_COUNT-1:0][BAR_ADDR_W-1:0] ctx_bar_addr_r;
    reg [DXA_CTX_COUNT-1:0][DXA_DESC_SLOT_W-1:0] ctx_desc_slot_r;
    reg [DXA_CTX_COUNT-1:0][`XLEN-1:0] ctx_smem_addr_r;
    reg [DXA_CTX_COUNT-1:0][`XLEN-1:0] ctx_flags_r;
    reg [DXA_CTX_COUNT-1:0][4:0][`XLEN-1:0] ctx_coords_r;

    reg [`VX_DCR_DXA_DESC_COUNT-1:0][`VX_DCR_DXA_DESC_STRIDE-1:0][31:0] dxa_desc_r;

    assign dcr_dxa_desc_write = dcr_bus_if.write_valid
                             && (dcr_bus_if.write_addr >= `VX_DCR_DXA_DESC_BASE)
                             && (dcr_bus_if.write_addr < (`VX_DCR_DXA_DESC_BASE + (`VX_DCR_DXA_DESC_COUNT * `VX_DCR_DXA_DESC_STRIDE)));
    assign dcr_desc_off = dcr_bus_if.write_addr - `VX_DCR_DXA_DESC_BASE;
    assign dcr_desc_off_w = 32'(dcr_desc_off);
    assign dcr_desc_slot = DXA_DESC_SLOT_W'(dcr_desc_off_w / `VX_DCR_DXA_DESC_STRIDE);
    assign dcr_desc_word = DXA_DESC_WORD_W'(dcr_desc_off_w % `VX_DCR_DXA_DESC_STRIDE);

    assign issue_bar_addr = ctx_bar_addr_r[req_ctx_idx];
    assign issue_desc_slot = ctx_desc_slot_r[req_ctx_idx];
    assign issue_smem_addr = ctx_smem_addr_r[req_ctx_idx];
    assign issue_flags = ctx_flags_r[req_ctx_idx];
    assign issue_coords = ctx_coords_r[req_ctx_idx];

    assign issue_base_addr = `MEM_ADDR_WIDTH'({
        dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_BASE_HI_OFF],
        dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_BASE_LO_OFF]
    });
    assign issue_desc_meta = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_META_OFF];
    assign issue_desc_tile01 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_TILESIZE01_OFF];
    assign issue_desc_tile23 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_TILESIZE23_OFF];
    assign issue_desc_tile4 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_TILESIZE4_OFF];
    assign issue_desc_cfill = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_CFILL_OFF];
    assign issue_size0 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_SIZE0_OFF];
    assign issue_size1 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_SIZE1_OFF];
    assign issue_stride0 = dxa_desc_r[issue_desc_slot][`VX_DCR_DXA_DESC_STRIDE0_OFF];

    always @(posedge clk) begin
        if (dcr_dxa_desc_write) begin
            dxa_desc_r[dcr_desc_slot][dcr_desc_word] <= dcr_bus_if.write_data;
        end

        if (reset) begin
            for (integer i = 0; i < DXA_CTX_COUNT; ++i) begin
                ctx_bar_addr_r[i] <= '0;
                ctx_desc_slot_r[i] <= '0;
                ctx_smem_addr_r[i] <= '0;
                ctx_flags_r[i] <= '0;
                for (integer j = 0; j < 5; ++j) begin
                    ctx_coords_r[i][j] <= '0;
                end
            end
        end else if (req_fire) begin
            case (req_op)
                DXA_OP_SETUP0: begin
                    ctx_desc_slot_r[req_ctx_idx] <= DXA_DESC_SLOT_W'(req_rs1[DXA_DESC_SLOT_W-1:0]);
                    ctx_bar_addr_r[req_ctx_idx] <= req_bar_addr;
                end
                DXA_OP_SETUP1: begin
                    ctx_smem_addr_r[req_ctx_idx] <= req_rs1;
                    // Packed launch meta path marks bit31 and stores flags in [15:8].
                    // Legacy setup1 keeps raw flags in rs2.
                    ctx_flags_r[req_ctx_idx] <= req_rs2[31] ? (`XLEN'(req_rs2[15:8])) : req_rs2;
                end
                DXA_OP_COORD01: begin
                    ctx_coords_r[req_ctx_idx][0] <= req_rs1;
                    ctx_coords_r[req_ctx_idx][1] <= req_rs2;
                end
                DXA_OP_COORD23: begin
                    ctx_coords_r[req_ctx_idx][2] <= req_rs1;
                    ctx_coords_r[req_ctx_idx][3] <= req_rs2;
                end
                DXA_OP_ISSUE: begin
                    ctx_coords_r[req_ctx_idx][4] <= req_rs1;
                end
                default: begin
                end
            endcase
        end
    end

endmodule
