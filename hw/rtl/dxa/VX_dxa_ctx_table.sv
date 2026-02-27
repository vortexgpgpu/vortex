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

/* verilator lint_off VARHIDDEN */
module VX_dxa_ctx_table import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
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
    // Next-gen skeleton: keep context-table as an explicit module.
    // Functional behavior is currently equivalent to the legacy issue_state.
    VX_dxa_issue_state #(
        .DXA_CTX_COUNT     (DXA_CTX_COUNT),
        .DXA_CTX_BITS      (DXA_CTX_BITS),
        .DXA_DESC_SLOT_BITS(DXA_DESC_SLOT_BITS),
        .DXA_DESC_SLOT_W   (DXA_DESC_SLOT_W)
    ) issue_state (
        .clk            (clk),
        .reset          (reset),
        .req_fire       (req_fire),
        .req_op         (req_op),
        .req_ctx_idx    (req_ctx_idx),
        .req_rs1        (req_rs1),
        .req_rs2        (req_rs2),
        .req_bar_addr   (req_bar_addr),
        .issue_bar_addr (issue_bar_addr),
        .issue_desc_slot(issue_desc_slot),
        .issue_smem_addr(issue_smem_addr),
        .issue_coords   (issue_coords),
        .issue_ctx_valid(issue_ctx_valid)
    );

endmodule
