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
    parameter DXA_DESC_SLOT_W = `UP(DXA_DESC_SLOT_BITS)
) (
    input wire clk,
    input wire reset,

    input wire req_fire,
    input wire [2:0] req_op,
    input wire [DXA_CTX_BITS-1:0] req_ctx_idx,
    input wire [`XLEN-1:0] req_rs1,
    input wire [`XLEN-1:0] req_rs2,
    input wire [BAR_ADDR_W-1:0] req_bar_addr,

    output wire [BAR_ADDR_W-1:0] issue_bar_addr,
    output wire [DXA_DESC_SLOT_W-1:0] issue_desc_slot,
    output wire [`XLEN-1:0] issue_smem_addr,
    output wire [4:0][`XLEN-1:0] issue_coords,
    output wire issue_ctx_valid
);
    localparam DXA_OP_SETUP   = 3'd0;
    localparam DXA_OP_COORD01 = 3'd1;
    localparam DXA_OP_COORD23 = 3'd2;
    localparam DXA_OP_ISSUE   = 3'd3;

    reg [DXA_CTX_COUNT-1:0][BAR_ADDR_W-1:0] ctxbar_addr_r;
    reg [DXA_CTX_COUNT-1:0][DXA_DESC_SLOT_W-1:0] ctx_desc_slot_r;
    reg [DXA_CTX_COUNT-1:0][`XLEN-1:0] ctx_smem_addr_r;
    reg [DXA_CTX_COUNT-1:0][4:0][`XLEN-1:0] ctx_coords_r;
    reg [DXA_CTX_COUNT-1:0] ctx_valid_r;

    assign issue_bar_addr = ctxbar_addr_r[req_ctx_idx];
    assign issue_desc_slot = ctx_desc_slot_r[req_ctx_idx];
    assign issue_smem_addr = ctx_smem_addr_r[req_ctx_idx];
    assign issue_coords = ctx_coords_r[req_ctx_idx];
    assign issue_ctx_valid = ctx_valid_r[req_ctx_idx];

    always @(posedge clk) begin
        if (reset) begin
            for (integer i = 0; i < DXA_CTX_COUNT; ++i) begin
                ctxbar_addr_r[i] <= '0;
                ctx_desc_slot_r[i] <= '0;
                ctx_smem_addr_r[i] <= '0;
                ctx_valid_r[i] <= 1'b0;
                for (integer j = 0; j < 5; ++j) begin
                    ctx_coords_r[i][j] <= '0;
                end
            end
        end else if (req_fire) begin
            case (req_op)
                DXA_OP_SETUP: begin
                    // rs1 = smem_addr, rs2 = packed meta
                    ctx_smem_addr_r[req_ctx_idx] <= req_rs1;
                    ctx_desc_slot_r[req_ctx_idx] <= DXA_DESC_SLOT_W'(req_rs2[DXA_DESC_SLOT_W-1:0]);
                    ctxbar_addr_r[req_ctx_idx] <= req_bar_addr;
                    ctx_valid_r[req_ctx_idx] <= 1'b1;
                    // Clear all coords to prevent stale leakage from
                    // previous higher-dimension instructions on this context.
                    for (integer j = 0; j < 5; ++j) begin
                        ctx_coords_r[req_ctx_idx][j] <= '0;
                    end
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
                    if (ctx_valid_r[req_ctx_idx]) begin
                        ctx_coords_r[req_ctx_idx][4] <= req_rs1;
                    end
                end
                default: begin
                end
            endcase
        end
    end

endmodule
