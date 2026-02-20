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
module VX_tma_uops import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,
    input  wire      start,
    input  wire      next,
    output reg       done
);
    localparam TMA_OP_SETUP0  = 3'd0;
    localparam TMA_OP_SETUP1  = 3'd1;
    localparam TMA_OP_COORD01 = 3'd2;
    localparam TMA_OP_COORD23 = 3'd3;
    localparam TMA_OP_ISSUE   = 3'd4;

    // One architected launch op expands to:
    // setup0(meta) -> setup1(smem,meta) -> coord01 -> coord23 -> issue(coord4)
    localparam TMA_UOPS = 5;
    localparam CTR_W = `CLOG2(TMA_UOPS);

    // Fixed coordinate register bank (fa5..fa9 / f5..f9 by index).
    localparam [4:0] COORD0_REG = 5'd5;
    localparam [4:0] COORD1_REG = 5'd6;
    localparam [4:0] COORD2_REG = 5'd7;
    localparam [4:0] COORD3_REG = 5'd8;
    localparam [4:0] COORD4_REG = 5'd9;

    function automatic [NUM_REGS_BITS-1:0] tma_coord_reg(input [4:0] ridx);
    begin
    `ifdef EXT_F_ENABLE
        tma_coord_reg = make_reg_num(REG_TYPE_F, ridx);
    `else
        tma_coord_reg = make_reg_num(REG_TYPE_I, ridx);
    `endif
    end
    endfunction

    reg [CTR_W-1:0] counter;
    reg busy;

    reg [2:0] uop_op;
    reg [NUM_SRC_OPDS-1:0] uop_used_rs;
    reg [NUM_REGS_BITS-1:0] uop_rs1;
    reg [NUM_REGS_BITS-1:0] uop_rs2;
    reg [NUM_REGS_BITS-1:0] uop_rs3;

    always @(*) begin
        uop_op = TMA_OP_SETUP0;
        uop_used_rs = '0;
        uop_rs1 = ibuf_in.rs1;
        uop_rs2 = ibuf_in.rs2;
        uop_rs3 = ibuf_in.rs3;

        case (counter)
            0: begin
                uop_op = TMA_OP_SETUP0;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                // setup0 uses rs1(desc/meta) + rs2(meta->bar addr)
                uop_rs1 = ibuf_in.rs2;
                uop_rs2 = ibuf_in.rs2;
            end
            1: begin
                uop_op = TMA_OP_SETUP1;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                // setup1 uses rs1(smem addr) + rs2(flags from meta high bits)
                uop_rs1 = ibuf_in.rs1;
                uop_rs2 = ibuf_in.rs2;
            end
            2: begin
                uop_op = TMA_OP_COORD01;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                uop_rs1 = tma_coord_reg(COORD0_REG);
                uop_rs2 = tma_coord_reg(COORD1_REG);
            end
            3: begin
                uop_op = TMA_OP_COORD23;
                uop_used_rs = NUM_SRC_OPDS'(3'b011);
                uop_rs1 = tma_coord_reg(COORD2_REG);
                uop_rs2 = tma_coord_reg(COORD3_REG);
            end
            default: begin
                uop_op = TMA_OP_ISSUE;
                uop_used_rs = NUM_SRC_OPDS'(3'b001);
                uop_rs1 = tma_coord_reg(COORD4_REG);
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
    assign ibuf_out.wr_xregs = ibuf_in.wr_xregs;
    assign ibuf_out.used_rs = uop_used_rs;
    assign ibuf_out.rd = ibuf_in.rd;
    assign ibuf_out.rs1 = uop_rs1;
    assign ibuf_out.rs2 = uop_rs2;
    assign ibuf_out.rs3 = uop_rs3;
    `UNUSED_VAR (ibuf_in.wb)

    always_ff @(posedge clk) begin
        if (reset) begin
            counter <= '0;
            busy    <= 1'b0;
            done    <= 1'b0;
        end else begin
            if (~busy && start) begin
                counter <= '0;
                busy    <= 1'b1;
                done    <= (TMA_UOPS == 1);
            end else if (busy && next) begin
                counter <= counter + CTR_W'(1);
                done <= (counter == CTR_W'(TMA_UOPS - 2));
                busy <= ~done;
            end
        end
    end

endmodule
/* verilator lint_on UNUSEDSIGNAL */
