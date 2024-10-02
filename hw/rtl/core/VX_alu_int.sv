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

module VX_alu_int #(
    parameter `STRING INSTANCE_ID = "",
    parameter BLOCK_IDX = 0,
    parameter NUM_LANES = 1
) (
    input wire              clk,
    input wire              reset,

    // Inputs
    VX_execute_if.slave     execute_if,

    // Outputs
    VX_commit_if.master     commit_if,
    VX_branch_ctl_if.master branch_ctl_if
);

    `UNUSED_SPARAM (INSTANCE_ID)
    localparam LANE_BITS      = `CLOG2(NUM_LANES);
    localparam LANE_WIDTH     = `UP(LANE_BITS);
    localparam PID_BITS       = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH      = `UP(PID_BITS);
    localparam SHIFT_IMM_BITS = `CLOG2(`XLEN);

    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [NUM_LANES-1:0][`XLEN-1:0] add_result;
    wire [NUM_LANES-1:0][`XLEN:0]   sub_result; // +1 bit for branch compare
    reg  [NUM_LANES-1:0][`XLEN-1:0] shr_zic_result;
    reg  [NUM_LANES-1:0][`XLEN-1:0] msc_result;

    wire [NUM_LANES-1:0][`XLEN-1:0] add_result_w;
    wire [NUM_LANES-1:0][`XLEN-1:0] sub_result_w;
    wire [NUM_LANES-1:0][`XLEN-1:0] shr_result_w;
    reg  [NUM_LANES-1:0][`XLEN-1:0] msc_result_w;

    reg [NUM_LANES-1:0][`XLEN-1:0] alu_result;
    wire [NUM_LANES-1:0][`XLEN-1:0] alu_result_r;

`ifdef XLEN_64
    wire is_alu_w = execute_if.data.op_args.alu.is_w;
`else
    wire is_alu_w = 0;
`endif

    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_BITS'(execute_if.data.op_type);
    wire [`INST_BR_BITS-1:0]   br_op = `INST_BR_BITS'(execute_if.data.op_type);
    wire                    is_br_op = (execute_if.data.op_args.alu.xtype == `ALU_TYPE_BRANCH);
    wire                   is_sub_op = `INST_ALU_IS_SUB(alu_op);
    wire                   is_signed = `INST_ALU_SIGNED(alu_op);
    wire [1:0]              op_class = is_br_op ? `INST_BR_CLASS(alu_op) : `INST_ALU_CLASS(alu_op);

    wire [NUM_LANES-1:0][`XLEN-1:0] alu_in1 = execute_if.data.rs1_data;
    wire [NUM_LANES-1:0][`XLEN-1:0] alu_in2 = execute_if.data.rs2_data;

    wire [NUM_LANES-1:0][`XLEN-1:0] alu_in1_PC  = execute_if.data.op_args.alu.use_PC ? {NUM_LANES{execute_if.data.PC, 1'd0}} : alu_in1;
    wire [NUM_LANES-1:0][`XLEN-1:0] alu_in2_imm = execute_if.data.op_args.alu.use_imm ? {NUM_LANES{`SEXT(`XLEN, execute_if.data.op_args.alu.imm)}} : alu_in2;
    wire [NUM_LANES-1:0][`XLEN-1:0] alu_in2_br  = (execute_if.data.op_args.alu.use_imm && ~is_br_op) ? {NUM_LANES{`SEXT(`XLEN, execute_if.data.op_args.alu.imm)}} : alu_in2;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign add_result[i] = alu_in1_PC[i] + alu_in2_imm[i];
        assign add_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] + alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [`XLEN:0] sub_in1 = {is_signed & alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN:0] sub_in2 = {is_signed & alu_in2_br[i][`XLEN-1], alu_in2_br[i]};
        assign sub_result[i] = sub_in1 - sub_in2;
        assign sub_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] - alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [`XLEN:0] shr_in1 = {is_signed && alu_in1[i][`XLEN-1], alu_in1[i]};
        always @(*) begin
            case (alu_op[1:0])
            `ifdef EXT_ZICOND_ENABLE
                2'b10, 2'b11: begin // CZERO
                    shr_zic_result[i] = alu_in1[i] & {`XLEN{alu_op[0] ^ (| alu_in2[i])}};
                end
            `endif
                default: begin // SRL, SRA, SRLI, SRAI
                    shr_zic_result[i] = `XLEN'($signed(shr_in1) >>> alu_in2_imm[i][SHIFT_IMM_BITS-1:0]);
                end
            endcase
        end
        wire [32:0] shr_in1_w = {is_signed && alu_in1[i][31], alu_in1[i][31:0]};
        wire [31:0] shr_res_w = 32'($signed(shr_in1_w) >>> alu_in2_imm[i][4:0]);
        assign shr_result_w[i] = `XLEN'($signed(shr_res_w));
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            case (alu_op[1:0])
                2'b00: msc_result[i] = alu_in1[i] & alu_in2_imm[i]; // AND
                2'b01: msc_result[i] = alu_in1[i] | alu_in2_imm[i]; // OR
                2'b10: msc_result[i] = alu_in1[i] ^ alu_in2_imm[i]; // XOR
                2'b11: msc_result[i] = alu_in1[i] << alu_in2_imm[i][SHIFT_IMM_BITS-1:0]; // SLL
            endcase
        end
        assign msc_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] << alu_in2_imm[i][4:0])); // SLLW
    end

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [`XLEN-1:0] slt_br_result = `XLEN'({is_br_op && ~(| sub_result[i][`XLEN-1:0]), sub_result[i][`XLEN]});
        wire [`XLEN-1:0] sub_slt_br_result = (is_sub_op && ~is_br_op) ? sub_result[i][`XLEN-1:0] : slt_br_result;
        always @(*) begin
            case ({is_alu_w, op_class})
                3'b000: alu_result[i] = add_result[i];      // ADD, LUI, AUIPC
                3'b001: alu_result[i] = sub_slt_br_result;  // SUB, SLTU, SLTI, BR*
                3'b010: alu_result[i] = shr_zic_result[i];  // SRL, SRA, SRLI, SRAI, CZERO*
                3'b011: alu_result[i] = msc_result[i];      // AND, OR, XOR, SLL, SLLI
                3'b100: alu_result[i] = add_result_w[i];    // ADDIW, ADDW
                3'b101: alu_result[i] = sub_result_w[i];    // SUBW
                3'b110: alu_result[i] = shr_result_w[i];    // SRLW, SRAW, SRLIW, SRAIW
                3'b111: alu_result[i] = msc_result_w[i];    // SLLW
            endcase
        end
    end

    // branch

    wire [`PC_BITS-1:0] PC_r;
    wire [`INST_BR_BITS-1:0] br_op_r;
    wire [`PC_BITS-1:0] cbr_dest, cbr_dest_r;
    wire [LANE_WIDTH-1:0] tid, tid_r;
    wire is_br_op_r;

    assign cbr_dest = add_result[0][1 +: `PC_BITS];

    if (LANE_BITS != 0) begin
        assign tid = execute_if.data.tid[0 +: LANE_BITS];
    end else begin
        assign tid = 0;
    end

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `NR_BITS + 1 + PID_WIDTH + 1 + 1 + (NUM_LANES * `XLEN) + `PC_BITS + `PC_BITS + 1 + `INST_BR_BITS + LANE_WIDTH)
    ) rsp_buf (
        .clk      (clk),
        .reset    (reset),
        .valid_in (execute_if.valid),
        .ready_in (execute_if.ready),
        .data_in  ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, alu_result, execute_if.data.PC, cbr_dest, is_br_op, br_op, tid}),
        .data_out ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.rd, commit_if.data.wb, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop, alu_result_r, PC_r, cbr_dest_r, is_br_op_r, br_op_r, tid_r}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    `UNUSED_VAR (br_op_r)
    wire is_br_neg  = `INST_BR_IS_NEG(br_op_r);
    wire is_br_less = `INST_BR_IS_LESS(br_op_r);
    wire is_br_static = `INST_BR_IS_STATIC(br_op_r);

    wire [`XLEN-1:0] br_result = alu_result_r[tid_r];
    wire is_less  = br_result[0];
    wire is_equal = br_result[1];

    wire br_enable = is_br_op_r && commit_if.valid && commit_if.ready && commit_if.data.eop;
    wire br_taken = ((is_br_less ? is_less : is_equal) ^ is_br_neg) | is_br_static;
    wire [`PC_BITS-1:0] br_dest = is_br_static ? br_result[1 +: `PC_BITS] : cbr_dest_r;
    wire [`NW_WIDTH-1:0] br_wid;
    `ASSIGN_BLOCKED_WID (br_wid, commit_if.data.wid, BLOCK_IDX, `NUM_ALU_BLOCKS)

    VX_pipe_register #(
        .DATAW (1 + `NW_WIDTH + 1 + `PC_BITS)
    ) branch_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  ({br_enable,           br_wid,            br_taken,            br_dest}),
        .data_out ({branch_ctl_if.valid, branch_ctl_if.wid, branch_ctl_if.taken, branch_ctl_if.dest})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        assign commit_if.data.data[i] = (is_br_op_r && is_br_static) ? {(PC_r + `PC_BITS'(2)), 1'd0} : alu_result_r[i];
    end

    assign commit_if.data.PC = PC_r;

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (br_enable) begin
            `TRACE(1, ("%d: %s-branch: wid=%0d, PC=0x%0h, taken=%b, dest=0x%0h (#%0d)\n",
                $time, INSTANCE_ID, br_wid, {commit_if.data.PC, 1'b0}, br_taken, {br_dest, 1'b0}, commit_if.data.uuid));
        end
    end
`endif

endmodule
