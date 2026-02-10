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

module VX_decode import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_fetch_if.slave       fetch_if,

    // outputs
    VX_decode_if.master     decode_if,
    VX_decode_sched_if.master decode_sched_if
);

    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    localparam OUT_DATAW = $bits(decode_t);

    reg [EX_BITS-1:0] ex_type;
    reg [INST_OP_BITS-1:0] op_type;
    op_args_t op_args;
    reg [NUM_SRC_OPDS:0][NUM_REGS_BITS-1:0] reg_ids;
    reg [NUM_SRC_OPDS:0] use_regs;
    reg [NUM_XREGS-1:0] rd_xregs;
    reg [NUM_XREGS-1:0] wr_xregs;
    reg is_wstall;

    wire [31:0] instr = fetch_if.data.instr;
    wire [6:0] opcode = instr[6:0];
    wire [1:0] funct2 = instr[26:25];
    wire [2:0] funct3 = instr[14:12];
    wire [4:0] funct5 = instr[31:27];
    wire [6:0] funct7 = instr[31:25];
    wire [11:0] u_12  = instr[31:20];

    wire [4:0] rd  = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];
    wire [4:0] rs3 = instr[31:27];

    `UNUSED_VAR (funct2)
    `UNUSED_VAR (funct5)

    wire is_itype_sh   = funct3[0] && ~funct3[1];
    wire is_csr_fflags = (u_12 == `VX_CSR_FFLAGS);
    wire is_csr_frm    = (u_12 == `VX_CSR_FRM);
    wire is_csr_fcsr   = (u_12 == `VX_CSR_FCSR);
    wire frm_is_dyn    = (funct3 == 3'b111);

    reg csr_write;
    always @(*) begin
        csr_write = 1'b0;
        unique case (funct3)
            3'b001, // CSRRW
            3'b101: // CSRRWI
                csr_write = 1'b1;
            3'b010, // CSRRS
            3'b011: // CSRRC
                csr_write = (rs1 != 0);
            3'b110, // CSRRSI
            3'b111: // CSRRCI
                csr_write = (rs1 != 0);
            default: csr_write = 1'b0;
        endcase
    end

    wire [19:0] ui_imm  = instr[31:12];
`ifdef XLEN_64
    wire [11:0] i_imm   = is_itype_sh ? {6'b0, instr[25:20]} : u_12;
    wire [11:0] iw_imm  = is_itype_sh ? {7'b0, instr[24:20]} : u_12;
`else
    wire [11:0] i_imm   = is_itype_sh ? {7'b0, instr[24:20]} : u_12;
`endif
    wire [11:0] s_imm   = {funct7, rd};
    wire [11:0] b_imm   = {instr[31], instr[7], instr[30:25], instr[11:8]};
    wire [19:0] jal_imm = {instr[31], instr[19:12], instr[20], instr[30:21]};

    reg [INST_ALU_BITS-1:0] r_type;
    always @(*) begin
        case (funct3)
            3'h0: r_type = (opcode[5] && funct7[5]) ? INST_ALU_SUB : INST_ALU_ADD;
            3'h1: r_type = INST_ALU_SLL;
            3'h2: r_type = INST_ALU_SLT;
            3'h3: r_type = INST_ALU_SLTU;
            3'h4: r_type = INST_ALU_XOR;
            3'h5: r_type = funct7[5] ? INST_ALU_SRA : INST_ALU_SRL;
            3'h6: r_type = INST_ALU_OR;
            3'h7: r_type = INST_ALU_AND;
        endcase
    end

    reg [INST_BR_BITS-1:0] b_type;
    always @(*) begin
        case (funct3)
            3'h0: b_type = INST_BR_BEQ;
            3'h1: b_type = INST_BR_BNE;
            3'h4: b_type = INST_BR_BLT;
            3'h5: b_type = INST_BR_BGE;
            3'h6: b_type = INST_BR_BLTU;
            3'h7: b_type = INST_BR_BGEU;
            default: b_type = 'x;
        endcase
    end

    reg [INST_BR_BITS-1:0] s_type;
    always @(*) begin
        case (u_12)
            12'h000: s_type = INST_OP_BITS'(INST_BR_ECALL);
            12'h001: s_type = INST_OP_BITS'(INST_BR_EBREAK);
            12'h002: s_type = INST_OP_BITS'(INST_BR_URET);
            12'h102: s_type = INST_OP_BITS'(INST_BR_SRET);
            12'h302: s_type = INST_OP_BITS'(INST_BR_MRET);
            default: s_type = 'x;
        endcase
    end

`ifdef EXT_M_ENABLE
    reg [INST_M_BITS-1:0] m_type;
    always @(*) begin
        case (funct3)
            3'h0: m_type = INST_M_MUL;
            3'h1: m_type = INST_M_MULH;
            3'h2: m_type = INST_M_MULHSU;
            3'h3: m_type = INST_M_MULHU;
            3'h4: m_type = INST_M_DIV;
            3'h5: m_type = INST_M_DIVU;
            3'h6: m_type = INST_M_REM;
            3'h7: m_type = INST_M_REMU;
        endcase
    end
`endif

    always @(*) begin

        ex_type   = 'x;
        op_type   = 'x;
        op_args   = 'x;
        reg_ids   = 'x;
        use_regs  = '0;
        rd_xregs  = '0;
        wr_xregs  = '0;
        is_wstall = 0;

        case (opcode)
            INST_I: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(r_type);
                op_args.alu.xtype = ALU_TYPE_ARITH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = `SEXT(20, i_imm);
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            INST_R: begin
                ex_type = EX_ALU;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 0;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
                case (funct7)
                `ifdef EXT_M_ENABLE
                    INST_R_F7_MUL: begin
                        // MUL, MULH, MULHSU, MULHU
                        op_type = INST_OP_BITS'(m_type);
                        op_args.alu.xtype = ALU_TYPE_MULDIV;
                    end
                `endif
                `ifdef EXT_ZICOND_ENABLE
                    INST_R_F7_ZICOND: begin
                        // CZERO-EQZ, CZERO-NEZ
                        op_type = funct3[1] ? INST_OP_BITS'(INST_ALU_CZNE) : INST_OP_BITS'(INST_ALU_CZEQ);
                        op_args.alu.xtype = ALU_TYPE_ARITH;
                    end
                `endif
                    default: begin
                        op_type = INST_OP_BITS'(r_type);
                        op_args.alu.xtype = ALU_TYPE_ARITH;
                    end
                endcase
            end
        `ifdef XLEN_64
            INST_I_W: begin
                // ADDIW, SLLIW, SRLIW, SRAIW
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(r_type);
                op_args.alu.xtype = ALU_TYPE_ARITH;
                op_args.alu.is_w = 1;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = `SEXT(20, iw_imm);
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            INST_R_W: begin
                ex_type = EX_ALU;
                op_args.alu.is_w = 1;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 0;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
                case (funct7)
                `ifdef EXT_M_ENABLE
                    INST_R_F7_MUL: begin
                        // MULW, DIVW, DIVUW, REMW, REMUW
                        op_type = INST_OP_BITS'(m_type);
                        op_args.alu.xtype = ALU_TYPE_MULDIV;
                    end
                `endif
                    default: begin
                        // ADDW, SUBW, SLLW, SRLW, SRAW
                        op_type = INST_OP_BITS'(r_type);
                        op_args.alu.xtype = ALU_TYPE_ARITH;
                    end
                endcase
            end
        `endif
            INST_LUI: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(INST_ALU_LUI);
                op_args.alu.xtype = ALU_TYPE_ARITH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = ui_imm;
                `USED_IREG (rd);
            end
            INST_AUIPC: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(INST_ALU_AUIPC);
                op_args.alu.xtype = ALU_TYPE_ARITH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 1;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = ui_imm;
                `USED_IREG (rd);
            end
            INST_JAL: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(INST_BR_JAL);
                op_args.alu.xtype = ALU_TYPE_BRANCH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 1;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = jal_imm;
                is_wstall = 1;
                `USED_IREG (rd);
            end
            INST_JALR: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(INST_BR_JALR);
                op_args.alu.xtype = ALU_TYPE_BRANCH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 0;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = `SEXT(20, u_12);
                is_wstall = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            INST_B: begin
                ex_type = EX_ALU;
                op_type = INST_OP_BITS'(b_type);
                op_args.alu.xtype = ALU_TYPE_BRANCH;
                op_args.alu.is_w = 0;
                op_args.alu.use_PC = 1;
                op_args.alu.use_imm = 1;
                op_args.alu.imm20 = `SEXT(20, b_imm);
                is_wstall = 1;
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            INST_FENCE: begin
                ex_type = EX_LSU;
                op_type = INST_LSU_FENCE;
                op_args.lsu.is_store = 0;
                op_args.lsu.is_float = 0;
                op_args.lsu.offset = 0;
            end
            INST_SYS : begin
                if (funct3[1:0] != 0) begin
                    ex_type = EX_SFU;
                    op_type = INST_OP_BITS'(inst_sfu_csr(funct3));
                    op_args.csr.addr = u_12;
                    op_args.csr.use_imm = funct3[2];
                    op_args.csr.imm5 = rs1;
                    rd_xregs[XREG_FFLAGS] = is_csr_fcsr || is_csr_fflags;
                    rd_xregs[XREG_FRM]    = is_csr_fcsr || is_csr_frm;
                    wr_xregs[XREG_FFLAGS] = csr_write && (is_csr_fcsr || is_csr_fflags);
                    wr_xregs[XREG_FRM]    = csr_write && (is_csr_fcsr || is_csr_frm);
                    `USED_IREG (rd);
                    `USED_REG (REG_TYPE_I, rs1, ~funct3[2]);
                end else begin
                    ex_type = EX_ALU;
                    op_type = INST_OP_BITS'(s_type);
                    op_args.alu.xtype = ALU_TYPE_BRANCH;
                    op_args.alu.is_w = 0;
                    op_args.alu.use_imm = 1;
                    op_args.alu.use_PC  = 1;
                    op_args.alu.imm20 = 20'd4;
                    is_wstall = 1;
                    `USED_IREG (rd);
                end
            end
        `ifdef EXT_F_ENABLE
            INST_FL,
        `endif
            INST_L: begin
                ex_type = EX_LSU;
                op_type = INST_OP_BITS'({1'b0, funct3});
                op_args.lsu.is_store = 0;
                op_args.lsu.is_float = opcode[2];
                op_args.lsu.offset = u_12;
                `USED_IREG (rs1);
            `ifdef EXT_F_ENABLE
                `USED_REG (opcode[2], rd, 1'b1);
            `else
                `USED_IREG (rd);
            `endif
            end
        `ifdef EXT_F_ENABLE
            INST_FS,
        `endif
            INST_S: begin
                ex_type = EX_LSU;
                op_type = INST_OP_BITS'({1'b1, funct3});
                op_args.lsu.is_store = 1;
                op_args.lsu.is_float = opcode[2];
                op_args.lsu.offset = s_imm;
                `USED_IREG (rs1);
            `ifdef EXT_F_ENABLE
                `USED_REG (opcode[2], rs2, 1'b1);
            `else
                `USED_IREG (rs2);
            `endif
            end
        `ifdef EXT_F_ENABLE
            INST_FMADD,  // 7'b1000011
            INST_FMSUB,  // 7'b1000111
            INST_FNMSUB, // 7'b1001011
            INST_FNMADD: // 7'b1001111
            begin
                ex_type = EX_FPU;
                op_type = INST_OP_BITS'({2'b00, 1'b1, opcode[3]});
                op_args.fpu.frm = funct3;
                op_args.fpu.fmt[0] = funct2[0]; // float/double
                op_args.fpu.fmt[1] = opcode[3] ^ opcode[2]; // SUB

                // track FCSR dependencies
                rd_xregs[XREG_FRM] = frm_is_dyn;
                wr_xregs[XREG_FFLAGS] = 1'b1;

                `USED_FREG (rd);
                `USED_FREG (rs1);
                `USED_FREG (rs2);
                `USED_FREG (rs3);
            end
            INST_FCI: begin
                ex_type = EX_FPU;
                op_args.fpu.frm = funct3;
                op_args.fpu.fmt[0] = funct2[0]; // float/double
                op_args.fpu.fmt[1] = rs2[1]; // CVT W/L
                // Most FP operations may set exception flags
                wr_xregs[XREG_FFLAGS] = 1'b1;
                case (funct5)
                    5'b00000, // FADD
                    5'b00001, // FSUB
                    5'b00010: // FMUL
                    begin
                        op_type = INST_OP_BITS'({2'b00, 1'b0, funct5[1]});
                        op_args.fpu.fmt[1] = funct5[0]; // SUB
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b00100: begin
                        // NCP: FSGNJ=0, FSGNJN=1, FSGNJX=2
                        op_type = INST_OP_BITS'(INST_FPU_MISC);
                        op_args.fpu.frm = INST_FRM_BITS'(funct3[1:0]);
                        wr_xregs[XREG_FFLAGS] = 1'b0;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b00101: begin
                        // NCP: FMIN=6, FMAX=7
                        op_type = INST_OP_BITS'(INST_FPU_MISC);
                        op_args.fpu.frm = INST_FRM_BITS'(funct3[0] ? 7 : 6);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                `ifdef FLEN_64
                    5'b01000: begin
                        // FCVT.S.D, FCVT.D.S
                        op_type = INST_OP_BITS'(INST_FPU_F2F);
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end
                `endif
                    5'b00011: begin
                        // FDIV
                        op_type = INST_OP_BITS'(INST_FPU_DIV);
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b01011: begin
                        // FSQRT
                        op_type = INST_OP_BITS'(INST_FPU_SQRT);
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end
                    5'b10100: begin
                        // FCMP
                        op_type = INST_OP_BITS'(INST_FPU_CMP);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b11000: begin
                        // FCVT.W.X, FCVT.WU.X
                        op_type = (rs2[0]) ? INST_OP_BITS'(INST_FPU_F2U) : INST_OP_BITS'(INST_FPU_F2I);
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                    end
                    5'b11010: begin
                        // FCVT.X.W, FCVT.X.WU
                        op_type = (rs2[0]) ? INST_OP_BITS'(INST_FPU_U2F) : INST_OP_BITS'(INST_FPU_I2F);
                        rd_xregs[XREG_FRM] = frm_is_dyn;
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                    5'b11100: begin
                        if (funct3[0]) begin
                            // NCP: FCLASS=3
                            op_type = INST_OP_BITS'(INST_FPU_MISC);
                            op_args.fpu.frm = INST_FRM_BITS'(3);
                        end else begin
                            // NCP: FMV.X.W=4
                            op_type = INST_OP_BITS'(INST_FPU_MISC);
                            op_args.fpu.frm = INST_FRM_BITS'(4);
                        end
                        wr_xregs[XREG_FFLAGS] = 1'b0;
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                    end
                    5'b11110: begin
                        // NCP: FMV.W.X=5
                        op_type = INST_OP_BITS'(INST_FPU_MISC);
                        op_args.fpu.frm = INST_FRM_BITS'(5);
                        wr_xregs[XREG_FFLAGS] = 1'b0;
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                default:;
                endcase
            end
        `endif
            INST_EXT1: begin
                case (funct7)
                    7'h00: begin
                        ex_type = EX_SFU;
                        is_wstall = 1;
                        case (funct3)
                            3'h0: begin // TMC
                                op_type = INST_OP_BITS'(INST_SFU_TMC);
                                `USED_IREG (rs1);
                            end
                            3'h1: begin // WSPAWN
                                op_type = INST_OP_BITS'(INST_SFU_WSPAWN);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h2: begin // SPLIT
                                op_type = INST_OP_BITS'(INST_SFU_SPLIT);
                                op_args.wctl.is_cond_neg = rs2[0];
                                `USED_IREG (rs1);
                                `USED_IREG (rd);
                            end
                            3'h3: begin // JOIN
                                op_type = INST_OP_BITS'(INST_SFU_JOIN);
                                `USED_IREG (rs1);
                            end
                            3'h4: begin // BAR
                                op_type = INST_OP_BITS'(INST_SFU_BAR);
                                op_args.wctl.is_async_bar = 0;
                                op_args.wctl.is_bar_arrive = 0;
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h5: begin // PRED
                                op_type = INST_OP_BITS'(INST_SFU_PRED);
                                op_args.wctl.is_cond_neg = rd[0];
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h6: begin // Asynchronous arrive/wait barrier
                                logic is_wait = (rd == 0);
                                op_type = INST_OP_BITS'(INST_SFU_BAR);
                                op_args.wctl.is_async_bar = 1;
                                op_args.wctl.is_bar_arrive = ~is_wait;
                                is_wstall = is_wait;
                                `USED_IREG (rd);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            default:;
                        endcase
                    end
                    7'h01: begin // VOTE, SHFL
                        ex_type = EX_ALU;
                        op_args.alu.xtype = ALU_TYPE_OTHER;
                        `USED_IREG (rd);
                        `USED_IREG (rs1);
                        if (funct3[2]) begin
                            `USED_IREG (rs2);
                        end
                        op_type = INST_OP_BITS'(funct3);
                    end
                `ifdef EXT_TCU_ENABLE
                    7'h02: begin
                        case (funct3)
                            3'h0: begin // WMMA_SYNC
                                ex_type = EX_TCU;
                                op_type = INST_OP_BITS'(INST_TCU_WMMA);
                                op_args.tcu.fmt_s  = rs1[3:0];
                                op_args.tcu.fmt_d  = rd[3:0];
                                op_args.tcu.step_m = '0;
                                op_args.tcu.step_n = '0;
                                `USED_FREG (rd);
                                `USED_FREG (rs1);
                                `USED_FREG (rs2);
                                `USED_FREG (rs3);
                            end
                            default:;
                        endcase
                    end
                `endif
                    default:;
                endcase
            end
            default:;
        endcase
    end

    // disable writes to x0
    wire wb = use_regs[RV_RD] && (reg_ids[RV_RD] != 0);

    VX_elastic_buffer #(
        .DATAW (OUT_DATAW),
        .SIZE  (0)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fetch_if.valid),
        .ready_in  (fetch_if.ready),
        .data_in   ({fetch_if.data.uuid,  fetch_if.data.wid,  fetch_if.data.tmask,  fetch_if.data.PC,  ex_type,                op_type,                op_args,                wb,                rd_xregs,                wr_xregs,                use_regs[3:1],          reg_ids[RV_RD],    reg_ids[RV_RS1],    reg_ids[RV_RS2],    reg_ids[RV_RS3]}),
        .data_out  ({decode_if.data.uuid, decode_if.data.wid, decode_if.data.tmask, decode_if.data.PC, decode_if.data.ex_type, decode_if.data.op_type, decode_if.data.op_args, decode_if.data.wb, decode_if.data.rd_xregs, decode_if.data.wr_xregs, decode_if.data.used_rs, decode_if.data.rd, decode_if.data.rs1, decode_if.data.rs2, decode_if.data.rs3}),
        .valid_out (decode_if.valid),
        .ready_out (decode_if.ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire fetch_fire = fetch_if.valid && fetch_if.ready;

    assign decode_sched_if.valid  = fetch_fire;
    assign decode_sched_if.wid    = fetch_if.data.wid;
    assign decode_sched_if.unlock = ~is_wstall;

`ifndef L1_ENABLE
    assign fetch_if.ibuf_pop = decode_if.ibuf_pop;
`endif

`ifdef DBG_TRACE_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && decode_if.ready) begin
            `TRACE(1, ("%t: %s decode: wid=%0d, PC=0x%0h, ex=", $time, INSTANCE_ID, decode_if.data.wid, to_fullPC(decode_if.data.PC)))
            VX_trace_pkg::trace_ex_type(1, decode_if.data.ex_type);
            `TRACE(1, (", op="))
            VX_trace_pkg::trace_ex_op(1, decode_if.data.ex_type, decode_if.data.op_type, decode_if.data.op_args);
            `TRACE(1, (", tmask=%b, wb=%b, rd_xregs=%b, wr_xregs=%b, used_rs=%b, rd=", decode_if.data.tmask, decode_if.data.wb, decode_if.data.rd_xregs, decode_if.data.wr_xregs, decode_if.data.used_rs))
            VX_trace_pkg::trace_reg_idx(1, decode_if.data.rd);
            `TRACE(1, (", rs1="))
            VX_trace_pkg::trace_reg_idx(1, decode_if.data.rs1);
            `TRACE(1, (", rs2="))
            VX_trace_pkg::trace_reg_idx(1, decode_if.data.rs2);
            `TRACE(1, (", rs3="))
            VX_trace_pkg::trace_reg_idx(1, decode_if.data.rs3);
            VX_trace_pkg::trace_op_args(1, decode_if.data.ex_type, decode_if.data.op_type, decode_if.data.op_args);
            `TRACE(1, (" (#%0d)\n", decode_if.data.uuid))
        end
    end
`endif

endmodule
