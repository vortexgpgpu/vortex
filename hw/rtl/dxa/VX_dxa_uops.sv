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

/* verilator lint_off UNUSEDSIGNAL */
module VX_dxa_uops import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,
    input  wire      start,
    input  wire      next,
    output reg       done
);
    localparam DXA_OP_SETUP   = 3'd0;
    localparam DXA_OP_COORD01 = 3'd1;
    localparam DXA_OP_COORD23 = 3'd2;
    localparam DXA_OP_ISSUE   = 3'd3;

    // Maximum micro-ops across all dimensions (5D = 4 uops).
    localparam DXA_MAX_UOPS = 4;
    localparam CTR_W = `CLOG2(DXA_MAX_UOPS);

    // Fixed coordinate register bank in integer register file:
    // x28/x29/x30/x31/x5 = t3/t4/t5/t6/t0.
    localparam [4:0] COORD0_REG = 5'd28;
    localparam [4:0] COORD1_REG = 5'd29;
    localparam [4:0] COORD2_REG = 5'd30;
    localparam [4:0] COORD3_REG = 5'd31;
    localparam [4:0] COORD4_REG = 5'd5;

    function automatic [NUM_REGS_BITS-1:0] dxa_coord_reg(input [4:0] ridx);
    begin
        dxa_coord_reg = make_reg_num(REG_TYPE_I, ridx);
    end
    endfunction

    reg [CTR_W-1:0] counter;
    reg busy;

    // Latched dimensionality from original instruction's funct3 (0=1D..4=5D).
    reg [2:0] dim_r;

    // Total uop count depends on dimensionality:
    //   1D/2D (dim 0-1): 3 uops  (SETUP, COORD01, ISSUE)
    //   3D-5D (dim 2-4): 4 uops  (SETUP, COORD01, COORD23, ISSUE)
    wire [CTR_W-1:0] last_counter = (dim_r <= 3'd1) ? CTR_W'(2) : CTR_W'(3);

    reg [2:0] uop_op;
    reg [NUM_SRC_OPDS-1:0] uop_used_rs;
    reg [NUM_REGS_BITS-1:0] uop_rs1;
    reg [NUM_REGS_BITS-1:0] uop_rs2;
    reg [NUM_REGS_BITS-1:0] uop_rs3;

    // Determine whether COORD23 step exists for this dimension.
    wire has_coord23 = (dim_r >= 3'd2);

    always @(*) begin
        uop_op = DXA_OP_SETUP;
        uop_used_rs = '0;
        uop_rs1 = ibuf_in.rs1;
        uop_rs2 = ibuf_in.rs2;
        uop_rs3 = ibuf_in.rs3;

        case (counter)
            0: begin
                // SETUP: rs1=smem_addr (architected rs1), rs2=meta (architected rs2)
                uop_op = DXA_OP_SETUP;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                uop_rs1 = ibuf_in.rs1;
                uop_rs2 = ibuf_in.rs2;
            end
            1: begin
                // COORD01: always present. coord0 = t3.
                // coord1 = t4 if dim >= 2D, else x0.
                uop_op = DXA_OP_COORD01;
                uop_rs1 = dxa_coord_reg(COORD0_REG);
                if (dim_r >= 3'd1) begin
                    uop_used_rs = NUM_SRC_OPDS'(3'b011);
                    uop_rs2 = dxa_coord_reg(COORD1_REG);
                end else begin
                    uop_used_rs = NUM_SRC_OPDS'(3'b001);
                    uop_rs2 = '0;
                end
            end
            2: begin
                if (has_coord23) begin
                    // COORD23: present for dim >= 3D.
                    // coord2 = t5. coord3 = t6 if dim >= 4D, else x0.
                    uop_op = DXA_OP_COORD23;
                    uop_rs1 = dxa_coord_reg(COORD2_REG);
                    if (dim_r >= 3'd3) begin
                        uop_used_rs = NUM_SRC_OPDS'(3'b011);
                        uop_rs2 = dxa_coord_reg(COORD3_REG);
                    end else begin
                        uop_used_rs = NUM_SRC_OPDS'(3'b001);
                        uop_rs2 = '0;
                    end
                end else begin
                    // ISSUE for 1D/2D (no COORD23 step).
                    uop_op = DXA_OP_ISSUE;
                    uop_used_rs = '0;
                    uop_rs1 = '0;
                    uop_rs2 = '0;
                end
            end
            default: begin
                // ISSUE for 3D/4D/5D.
                uop_op = DXA_OP_ISSUE;
                if (dim_r == 3'd4) begin
                    // 5D: coord4 = t0
                    uop_used_rs = NUM_SRC_OPDS'(3'b001);
                    uop_rs1 = dxa_coord_reg(COORD4_REG);
                end else begin
                    uop_used_rs = '0;
                    uop_rs1 = '0;
                end
                uop_rs2 = '0;
            end
        endcase
    end

`ifdef UUID_ENABLE
    wire [31:0] uuid_lo = {counter, ibuf_in.uuid[0 +: (32-CTR_W)]};
    wire [UUID_WIDTH-1:0] uuid = {ibuf_in.uuid[UUID_WIDTH-1:32], uuid_lo};
`else
    wire [UUID_WIDTH-1:0] uuid = ibuf_in.uuid;
`endif

    assign ibuf_out.uuid = uuid;
    assign ibuf_out.tmask = ibuf_in.tmask;
    assign ibuf_out.PC = ibuf_in.PC;
    assign ibuf_out.ex_type = ibuf_in.ex_type;
    assign ibuf_out.op_type = ibuf_in.op_type;
    assign ibuf_out.op_args = op_args_t'({ibuf_in.op_args[INST_ARGS_BITS-1:3], uop_op});
    assign ibuf_out.wb = 1'b0;
    assign ibuf_out.rd_xregs = ibuf_in.rd_xregs;
    // DXA launch/uops are side-effect ops; they must not reserve writeback state.
    assign ibuf_out.wr_xregs = '0;
    assign ibuf_out.used_rs = uop_used_rs;
    assign ibuf_out.rd = '0;
    assign ibuf_out.rs1 = uop_rs1;
    assign ibuf_out.rs2 = uop_rs2;
    assign ibuf_out.rs3 = uop_rs3;
    `UNUSED_VAR (ibuf_in.wb)

    always_ff @(posedge clk) begin
        if (reset) begin
            counter <= '0;
            busy    <= 1'b0;
            done    <= 1'b0;
            dim_r   <= '0;
        end else begin
            if (~busy && start) begin
                counter <= '0;
                busy    <= 1'b1;
                dim_r   <= ibuf_in.op_args[2:0]; // latch funct3 as dimensionality
                done    <= 1'b0;
            end else if (busy && next) begin
                counter <= counter + CTR_W'(1);
                done <= (counter == (last_counter - CTR_W'(1)));
                busy <= ~done;
            end
        end
    end

endmodule
/* verilator lint_on UNUSEDSIGNAL */
