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

module VX_vint_unit #(
    parameter CORE_ID   = 0,
    parameter BLOCK_IDX = 0,
    parameter NUM_LANES = 1,
    parameter NUM_VECTOR_LANES = 1
) (
    input wire              clk,
    input wire              reset,
    
    // Inputs
    VX_vexecute_if.slave     execute_if,

    // Outputs    
    VX_vcommit_if.master     commit_if
);   
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_PARAM(BLOCK_IDX)
    localparam PID_BITS       = `CLOG2(`NUM_THREADS / NUM_VECTOR_LANES);
    localparam PID_WIDTH      = `UP(PID_BITS);
    localparam SHIFT_IMM_BITS = `CLOG2(`XLEN);

    `UNUSED_VAR (execute_if.data.rs3_data)

    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] add_result;
    wire [NUM_VECTOR_LANES-1:0][`XLEN:0] sub_result; // +1 bit for branch compare
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] shr_result;
    reg  [NUM_VECTOR_LANES-1:0][`XLEN-1:0] msc_result;
    
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] add_result_w;
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] sub_result_w;
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] shr_result_w;
    reg  [NUM_VECTOR_LANES-1:0][`XLEN-1:0] msc_result_w;

    reg [NUM_VECTOR_LANES-1:0][`XLEN-1:0] alu_result;
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] alu_result_r;

`ifdef XLEN_64
    wire is_alu_w = `INST_ALU_IS_W(execute_if.data.op_mod);
`else
    wire is_alu_w = 0;
`endif

    `UNUSED_VAR (execute_if.data.op_mod)

    wire [`INST_ALU_BITS-1:0] alu_op = `INST_ALU_BITS'(execute_if.data.op_type);
    // wire                   is_sub_op = `INST_ALU_IS_SUB(alu_op);
    wire                   is_signed = `INST_ALU_SIGNED(alu_op);   
    wire [1:0]              op_class = `INST_ALU_CLASS(alu_op);
    
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] alu_in1 = execute_if.data.vs1_data;
    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] alu_in2 = execute_if.data.vs2_data;

    wire [NUM_VECTOR_LANES-1:0][`XLEN-1:0] alu_in2_imm = execute_if.data.use_imm ? {NUM_VECTOR_LANES{execute_if.data.imm}} : alu_in2;

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin
        assign add_result[i] = alu_in1[i] + alu_in2_imm[i];
        assign add_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] + alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin
        wire [`XLEN:0] sub_in1 = {is_signed & alu_in1[i][`XLEN-1], alu_in1[i]};
        wire [`XLEN:0] sub_in2 = {is_signed & alu_in2_imm[i][`XLEN-1], alu_in2_imm[i]};
        assign sub_result[i] = sub_in1 - sub_in2;
        assign sub_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] - alu_in2_imm[i][31:0]));
    end

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin    
        wire [`XLEN:0] shr_in1 = {is_signed && alu_in1[i][`XLEN-1], alu_in1[i]};        
        assign shr_result[i] = `XLEN'($signed(shr_in1) >>> alu_in2_imm[i][SHIFT_IMM_BITS-1:0]);
        wire [32:0] shr_in1_w = {is_signed && alu_in1[i][31], alu_in1[i][31:0]};
        wire [31:0] shr_res_w = 32'($signed(shr_in1_w) >>> alu_in2_imm[i][4:0]);
        assign shr_result_w[i] = `XLEN'($signed(shr_res_w));
    end

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin
        always @(*) begin
            case (alu_op[1:0])
                2'b00: msc_result[i] = alu_in1[i] & alu_in2_imm[i]; // AND
                2'b01: msc_result[i] = alu_in1[i] | alu_in2_imm[i]; // OR
                2'b10: msc_result[i] = alu_in1[i] ^ alu_in2_imm[i]; // XOR
                2'b11: msc_result[i] = alu_in1[i] << alu_in2_imm[i][SHIFT_IMM_BITS-1:0]; // SLL
            endcase
        end
        assign msc_result_w[i] = `XLEN'($signed(alu_in1[i][31:0] << alu_in2_imm[i][4:0]));
    end

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin
        always @(*) begin
            case ({is_alu_w, op_class})                        
                3'b000: alu_result[i] = add_result[i];      // ADD, LUI, AUIPC
                3'b001: alu_result[i] =  sub_result[i][`XLEN-1:0];    // SUB, SLTU, SLTI
                3'b010: alu_result[i] = shr_result[i];      // SRL, SRA, SRLI, SRAI
                3'b011: alu_result[i] = msc_result[i];      // AND, OR, XOR, SLL, SLLI
                3'b100: alu_result[i] = add_result_w[i];    // ADDIW, ADDW
                3'b101: alu_result[i] = sub_result_w[i];    // SUBW
                3'b110: alu_result[i] = shr_result_w[i];    // SRLW, SRAW, SRLIW, SRAIW
                3'b111: alu_result[i] = msc_result_w[i];    // SLLW
            endcase
        end       
    end


    wire [`XLEN-1:0] PC_r;

    VX_elastic_buffer #(
        .DATAW (`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `NR_BITS + 1 + PID_WIDTH + 1+ 1 +  (NUM_VECTOR_LANES * `XLEN) + `XLEN)
    ) rsp_buf (
        .clk      (clk),
        .reset    (reset),
        .valid_in (execute_if.valid),
        .ready_in (execute_if.ready),
        .data_in  ({execute_if.data.uuid, execute_if.data.wid, execute_if.data.tmask, execute_if.data.rd, execute_if.data.wb, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop, alu_result, execute_if.data.PC}),
        .data_out ({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.rd, commit_if.data.wb, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop, alu_result_r, PC_r}),
        .valid_out (commit_if.valid),
        .ready_out (commit_if.ready)
    );

    for (genvar i = 0; i < NUM_VECTOR_LANES; ++i) begin    
        assign commit_if.data.data[i] = alu_result_r[i];
    end

    assign commit_if.data.PC = PC_r;
 

endmodule
