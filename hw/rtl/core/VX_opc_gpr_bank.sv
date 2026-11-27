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

// reset all GPRs in debug mode
`ifdef SIMULATION
`ifndef NDEBUG
`ifndef GPR_RESET
`define GPR_RESET
`endif
`endif
`endif

// One GPR bank of an operand collector: a register is in bank rd % NUM_BANKS,
// at {sid, warp slot, rd / NUM_BANKS}. The bank's index is an input, so every
// bank is the same module.
module VX_opc_gpr_bank import VX_gpu_pkg::*; #(
    parameter NUM_BANKS = 4,
    parameter BANK_SIZE = 1,
    parameter SEL_W     = `UP(`CLOG2(NUM_BANKS)),
    parameter ADDRW     = `CLOG2(BANK_SIZE),
    parameter REG_W     = NUM_REGS_BITS - `CLOG2(NUM_BANKS)
) (
    input wire              clk,
    input wire              reset,

    input wire [SEL_W-1:0]  bank_id,

    // write port
    input wire              wr_valid,
    input wire [SEL_W-1:0]  wr_bank_idx,
    input wire [ADDRW-1:0]  wr_addr,
    input wire [`VX_CFG_SIMD_WIDTH*XLENB-1:0] wr_byteen,
    input wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] wr_data,

    // read port
    input wire              rd_fire,
    input wire              rd_valid,
    input wire [PER_OPC_NW_W-1:0] rd_slot,
    input wire [SIMD_IDX_W-1:0] rd_sid,
    input wire [REG_W-1:0]  rd_reg,
    output wire [`VX_CFG_SIMD_WIDTH-1:0][`VX_CFG_XLEN-1:0] rd_data
);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_DATA_WIDTH = `VX_CFG_XLEN * `VX_CFG_SIMD_WIDTH;
    localparam BANK_DATA_SIZE  = BANK_DATA_WIDTH / 8;

    wire gpr_wr_enabled;
    if (BANK_SEL_BITS != 0) begin : g_gpr_wr_enabled_bn
        assign gpr_wr_enabled = wr_valid && (wr_bank_idx == BANK_SEL_BITS'(bank_id));
    end else begin : g_gpr_wr_enabled_b1
        assign gpr_wr_enabled = wr_valid;
        `UNUSED_VAR ({wr_bank_idx, bank_id})
    end

    if (PER_OPC_NW_BITS == 0) begin : g_slot_unused
        `UNUSED_VAR (rd_slot)
    end

    wire [ADDRW-1:0] gpr_rd_addr;
    if (SIMD_COUNT != 1) begin : g_gpr_rd_addr_sid
        wire [PER_OPC_NW_BITS + REG_W-1:0] tmp;
        `CONCAT(tmp, rd_slot, rd_reg, PER_OPC_NW_BITS, REG_W)
        assign gpr_rd_addr = {rd_sid, tmp};
    end else begin : g_gpr_rd_addr
        `CONCAT(gpr_rd_addr, rd_slot, rd_reg, PER_OPC_NW_BITS, REG_W)
        `UNUSED_VAR (rd_sid)
    end

    VX_dp_ram #(
        .DATAW (BANK_DATA_WIDTH),
        .SIZE  (BANK_SIZE),
        .WRENW (BANK_DATA_SIZE),
     `ifdef GPR_RESET
        .RESET_RAM (1),
     `endif
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) gpr_ram (
        .clk   (clk),
        .reset (reset),
        .read  (rd_fire && rd_valid),
        .wren  (wr_byteen),
        .write (gpr_wr_enabled),
        .waddr (wr_addr),
        .wdata (wr_data),
        .raddr (gpr_rd_addr),
        .rdata (rd_data)
    );

endmodule
