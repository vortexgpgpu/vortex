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

//
// DXA uop expander.
//
module VX_dxa_uops import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    localparam DXA_OP_SETUP   = 3'd0;
    localparam DXA_OP_COORD01 = 3'd1;
    localparam DXA_OP_COORD23 = 3'd2;
    localparam DXA_OP_ISSUE   = 3'd3;

    // Maximum micro-ops across all dimensions (5D = 4 uops).
    localparam DXA_MAX_UOPS = 4;
    localparam CTR_W = `CLOG2(DXA_MAX_UOPS);   // 2 bits

    // Truncate the shared uop_idx to the bits DXA actually needs.
    wire [CTR_W-1:0] ctr = CTR_W'(uop_idx);
    `UNUSED_VAR (uop_idx)

    // Fixed coordinate register bank in the integer register file:
    // x28/x29/x30/x31/x5 = t3/t4/t5/t6/t0.
    localparam [4:0] COORD0_REG = 5'd28;
    localparam [4:0] COORD1_REG = 5'd29;
    localparam [4:0] COORD2_REG = 5'd30;
    localparam [4:0] COORD3_REG = 5'd31;
    localparam [4:0] COORD4_REG = 5'd5;

    function automatic [NUM_REGS_BITS-1:0] dxa_coord_reg(input [4:0] ridx);
        dxa_coord_reg = make_reg_num(REG_TYPE_I, ridx);
    endfunction

    // -----------------------------------------------------------------------
    // uop_count: combinational from ibuf_in.
    //   1D/2D (funct3 0-1) -> 3 uops:  SETUP, COORD01, ISSUE
    //   3D-5D (funct3 2-4) -> 4 uops:  SETUP, COORD01, COORD23, ISSUE
    // The sequencer reads this on the start cycle and latches last_ctr.
    // -----------------------------------------------------------------------
    assign uop_count = (ibuf_in.op_args[2:0] <= 3'd1)
        ? UOP_CTR_W'(3)
        : UOP_CTR_W'(4);

    // -----------------------------------------------------------------------
    // Latched dimensionality — stable for the whole uop burst.
    // (The combinational operand decode below uses dim_r, not ibuf_in.op_args,
    //  so it is correct even if ibuf_in changes after the start cycle.)
    // -----------------------------------------------------------------------
    reg [2:0] dim_r;

    wire has_coord23 = (dim_r >= 3'd2);

    // -----------------------------------------------------------------------
    // Combinational operand decode (uop_idx-index -> register operands).
    // Unchanged from original except `uop_idx` replaced by `ctr`.
    // -----------------------------------------------------------------------
    reg [2:0] uop_op;
    reg [NUM_SRC_OPDS-1:0] uop_used_rs;
    reg [NUM_REGS_BITS-1:0] uop_rs1;
    reg [NUM_REGS_BITS-1:0] uop_rs2;
    reg [NUM_REGS_BITS-1:0] uop_rs3;

    always @(*) begin
        uop_op      = DXA_OP_SETUP;
        uop_used_rs = '0;
        uop_rs1     = ibuf_in.rs1;
        uop_rs2     = ibuf_in.rs2;
        uop_rs3     = ibuf_in.rs3;

        case (ctr)
            2'd0: begin
                // SETUP: rs1 = smem_addr, rs2 = meta (architected operands).
                uop_op      = DXA_OP_SETUP;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                uop_rs1     = ibuf_in.rs1;
                uop_rs2     = ibuf_in.rs2;
            end
            2'd1: begin
                // COORD01: always present.
                // coord0 = t3; coord1 = t4 when dim >= 2D, else x0.
                uop_op  = DXA_OP_COORD01;
                uop_rs1 = dxa_coord_reg(COORD0_REG);
                if (dim_r >= 3'd1) begin
                    uop_used_rs = NUM_SRC_OPDS'(3'b011);
                    uop_rs2     = dxa_coord_reg(COORD1_REG);
                end else begin
                    uop_used_rs = NUM_SRC_OPDS'(3'b001);
                    uop_rs2     = '0;
                end
            end
            2'd2: begin
                if (has_coord23) begin
                    // COORD23: present for dim >= 3D.
                    // coord2 = t5; coord3 = t6 when dim >= 4D, else x0.
                    uop_op  = DXA_OP_COORD23;
                    uop_rs1 = dxa_coord_reg(COORD2_REG);
                    if (dim_r >= 3'd3) begin
                        uop_used_rs = NUM_SRC_OPDS'(3'b011);
                        uop_rs2     = dxa_coord_reg(COORD3_REG);
                    end else begin
                        uop_used_rs = NUM_SRC_OPDS'(3'b001);
                        uop_rs2     = '0;
                    end
                end else begin
                    // ISSUE for 1D/2D (no COORD23 step).
                    uop_op      = DXA_OP_ISSUE;
                    uop_used_rs = '0;
                    uop_rs1     = '0;
                    uop_rs2     = '0;
                end
            end
            default: begin
                // ISSUE for 3D/4D/5D.
                uop_op = DXA_OP_ISSUE;
                if (dim_r == 3'd4) begin
                    // 5D: coord4 = t0.
                    uop_used_rs = NUM_SRC_OPDS'(3'b001);
                    uop_rs1     = dxa_coord_reg(COORD4_REG);
                end else begin
                    uop_used_rs = '0;
                    uop_rs1     = '0;
                end
                uop_rs2 = '0;
            end
        endcase
    end

    // -----------------------------------------------------------------------
    // Output assembly (combinational; unchanged from original).
    // -----------------------------------------------------------------------
    assign ibuf_out.uuid     = get_uop_uuid(ibuf_in.uuid, uop_idx);
    assign ibuf_out.tmask    = ibuf_in.tmask;
    assign ibuf_out.PC       = ibuf_in.PC;
    assign ibuf_out.ex_type  = ibuf_in.ex_type;
    assign ibuf_out.op_type  = ibuf_in.op_type;
    assign ibuf_out.op_args  = op_args_t'({ibuf_in.op_args[INST_ARGS_BITS-1:3], uop_op});
    assign ibuf_out.wb       = 1'b0;
    assign ibuf_out.rd_xregs = ibuf_in.rd_xregs;
    assign ibuf_out.wr_xregs = '0;   // side-effect ops; no writeback state
    assign ibuf_out.used_rs  = uop_used_rs;
    assign ibuf_out.rd       = '0;
    assign ibuf_out.rs1      = uop_rs1;
    assign ibuf_out.rs2      = uop_rs2;
    assign ibuf_out.rs3      = uop_rs3;
    `UNUSED_VAR (ibuf_in)

    // -----------------------------------------------------------------------
    // Minimal sequential state: only dim_r remains.
    // Latched on start, stable for the whole uop burst.
    // (uop_idx / busy / done / last_counter all moved to VX_uop_sequencer)
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            dim_r <= '0;
        end else if (start) begin
            dim_r <= ibuf_in.op_args[2:0];   // latch funct3 as dimensionality
        end
    end

endmodule
/* verilator lint_on UNUSEDSIGNAL */
