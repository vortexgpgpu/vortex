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

`ifndef VX_TRACE_PKG_VH
`define VX_TRACE_PKG_VH

`include "VX_define.vh"

package VX_trace_pkg;

`ifdef SIMULATION

    import VX_gpu_pkg::*;

`ifdef SV_DPI
    import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
`endif

    task trace_reg_idx(input int level, input logic [NUM_REGS_BITS-1:0] reg_id);
        `TRACE(level, ("%0d", reg_id));
    endtask

    task trace_ex_type(input int level, input [EX_BITS-1:0] ex_type);
        case (ex_type)
            EX_ALU: `TRACE(level, ("ALU"))
            EX_LSU: `TRACE(level, ("LSU"))
            EX_SFU: `TRACE(level, ("SFU"))
        `ifdef EXT_F_ENABLE
            EX_FPU: `TRACE(level, ("FPU"))
        `endif
        `ifdef EXT_TCU_ENABLE
            EX_TCU: `TRACE(level, ("TCU"))
        `endif
            default: `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [EX_BITS-1:0] ex_type,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        case (ex_type)
        EX_ALU: begin
            case (op_args.alu.xtype)
                ALU_TYPE_ARITH: begin
                    if (op_args.alu.is_w) begin
                        if (op_args.alu.use_imm) begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD: `TRACE(level, ("ADDIW"))
                                INST_ALU_SLL: `TRACE(level, ("SLLIW"))
                                INST_ALU_SRL: `TRACE(level, ("SRLIW"))
                                INST_ALU_SRA: `TRACE(level, ("SRAIW"))
                                default:      `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD: `TRACE(level, ("ADDW"))
                                INST_ALU_SUB: `TRACE(level, ("SUBW"))
                                INST_ALU_SLL: `TRACE(level, ("SLLW"))
                                INST_ALU_SRL: `TRACE(level, ("SRLW"))
                                INST_ALU_SRA: `TRACE(level, ("SRAW"))
                                default:      `TRACE(level, ("?"))
                            endcase
                        end
                    end else begin
                        if (op_args.alu.use_imm) begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD:   `TRACE(level, ("ADDI"))
                                INST_ALU_SLL:   `TRACE(level, ("SLLI"))
                                INST_ALU_SRL:   `TRACE(level, ("SRLI"))
                                INST_ALU_SRA:   `TRACE(level, ("SRAI"))
                                INST_ALU_SLT:   `TRACE(level, ("SLTI"))
                                INST_ALU_SLTU:  `TRACE(level, ("SLTIU"))
                                INST_ALU_XOR:   `TRACE(level, ("XORI"))
                                INST_ALU_OR:    `TRACE(level, ("ORI"))
                                INST_ALU_AND:   `TRACE(level, ("ANDI"))
                                INST_ALU_LUI:   `TRACE(level, ("LUI"))
                                INST_ALU_AUIPC: `TRACE(level, ("AUIPC"))
                                default:        `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD:   `TRACE(level, ("ADD"))
                                INST_ALU_SUB:   `TRACE(level, ("SUB"))
                                INST_ALU_SLL:   `TRACE(level, ("SLL"))
                                INST_ALU_SRL:   `TRACE(level, ("SRL"))
                                INST_ALU_SRA:   `TRACE(level, ("SRA"))
                                INST_ALU_SLT:   `TRACE(level, ("SLT"))
                                INST_ALU_SLTU:  `TRACE(level, ("SLTU"))
                                INST_ALU_XOR:   `TRACE(level, ("XOR"))
                                INST_ALU_OR:    `TRACE(level, ("OR"))
                                INST_ALU_AND:   `TRACE(level, ("AND"))
                                INST_ALU_CZEQ:  `TRACE(level, ("CZERO.EQZ"))
                                INST_ALU_CZNE:  `TRACE(level, ("CZERO.NEZ"))
                                default:        `TRACE(level, ("?"))
                            endcase
                        end
                    end
                end
                ALU_TYPE_OTHER: begin
                    if (op_type[2]) begin
                        case (INST_SHFL_BITS'(op_type))
                            INST_SHFL_UP:  `TRACE(level, ("SHFL.UP"))
                            INST_SHFL_DOWN:`TRACE(level, ("SHFL.DOWN"))
                            INST_SHFL_BFLY:`TRACE(level, ("SHFL.BFLY"))
                            INST_SHFL_IDX: `TRACE(level, ("SHFL.IDX"))
                            default:       `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (INST_VOTE_BITS'(op_type))
                            INST_VOTE_ALL: `TRACE(level, ("VOTE.ALL"))
                            INST_VOTE_ANY: `TRACE(level, ("VOTE.ANY"))
                            INST_VOTE_UNI: `TRACE(level, ("VOTE.UNI"))
                            INST_VOTE_BAL: `TRACE(level, ("VOTE.BAL"))
                            default:       `TRACE(level, ("?"))
                        endcase
                    end
                end
                ALU_TYPE_BRANCH: begin
                    case (INST_BR_BITS'(op_type))
                        INST_BR_BEQ:   `TRACE(level, ("BEQ"))
                        INST_BR_BNE:   `TRACE(level, ("BNE"))
                        INST_BR_BLT:   `TRACE(level, ("BLT"))
                        INST_BR_BGE:   `TRACE(level, ("BGE"))
                        INST_BR_BLTU:  `TRACE(level, ("BLTU"))
                        INST_BR_BGEU:  `TRACE(level, ("BGEU"))
                        INST_BR_JAL:   `TRACE(level, ("JAL"))
                        INST_BR_JALR:  `TRACE(level, ("JALR"))
                        INST_BR_ECALL: `TRACE(level, ("ECALL"))
                        INST_BR_EBREAK:`TRACE(level, ("EBREAK"))
                        INST_BR_URET:  `TRACE(level, ("URET"))
                        INST_BR_SRET:  `TRACE(level, ("SRET"))
                        INST_BR_MRET:  `TRACE(level, ("MRET"))
                        default:       `TRACE(level, ("?"))
                    endcase
                end
                ALU_TYPE_MULDIV: begin
                    if (op_args.alu.is_w) begin
                        case (INST_M_BITS'(op_type))
                            INST_M_MUL:  `TRACE(level, ("MULW"))
                            INST_M_DIV:  `TRACE(level, ("DIVW"))
                            INST_M_DIVU: `TRACE(level, ("DIVUW"))
                            INST_M_REM:  `TRACE(level, ("REMW"))
                            INST_M_REMU: `TRACE(level, ("REMUW"))
                            default:      `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (INST_M_BITS'(op_type))
                            INST_M_MUL:   `TRACE(level, ("MUL"))
                            INST_M_MULH:  `TRACE(level, ("MULH"))
                            INST_M_MULHSU:`TRACE(level, ("MULHSU"))
                            INST_M_MULHU: `TRACE(level, ("MULHU"))
                            INST_M_DIV:   `TRACE(level, ("DIV"))
                            INST_M_DIVU:  `TRACE(level, ("DIVU"))
                            INST_M_REM:   `TRACE(level, ("REM"))
                            INST_M_REMU:  `TRACE(level, ("REMU"))
                            default:      `TRACE(level, ("?"))
                        endcase
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
        EX_LSU: begin
            if (op_args.lsu.is_float) begin
                case (INST_LSU_BITS'(op_type))
                    INST_LSU_LW: `TRACE(level, ("FLW"))
                    INST_LSU_LD: `TRACE(level, ("FLD"))
                    INST_LSU_SW: `TRACE(level, ("FSW"))
                    INST_LSU_SD: `TRACE(level, ("FSD"))
                    default:     `TRACE(level, ("?"))
                endcase
            end else begin
                case (INST_LSU_BITS'(op_type))
                    INST_LSU_LB: `TRACE(level, ("LB"))
                    INST_LSU_LH: `TRACE(level, ("LH"))
                    INST_LSU_LW: `TRACE(level, ("LW"))
                    INST_LSU_LD: `TRACE(level, ("LD"))
                    INST_LSU_LBU:`TRACE(level, ("LBU"))
                    INST_LSU_LHU:`TRACE(level, ("LHU"))
                    INST_LSU_LWU:`TRACE(level, ("LWU"))
                    INST_LSU_SB: `TRACE(level, ("SB"))
                    INST_LSU_SH: `TRACE(level, ("SH"))
                    INST_LSU_SW: `TRACE(level, ("SW"))
                    INST_LSU_SD: `TRACE(level, ("SD"))
                    INST_LSU_FENCE:`TRACE(level,("FENCE"))
                    default:     `TRACE(level, ("?"))
                endcase
            end
        end
        EX_SFU: begin
            case (INST_SFU_BITS'(op_type))
                INST_SFU_TMC:   `TRACE(level, ("TMC"))
                INST_SFU_WSPAWN:`TRACE(level, ("WSPAWN"))
                INST_SFU_SPLIT: begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("SPLIT.N"))
                    end else begin
                        `TRACE(level, ("SPLIT"))
                    end
                end
                INST_SFU_JOIN:  `TRACE(level, ("JOIN"))
                INST_SFU_BAR:   `TRACE(level, ("BAR"))
                INST_SFU_PRED:  begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("PRED.N"))
                    end else begin
                        `TRACE(level, ("PRED"))
                    end
                end
                INST_SFU_CSRRW: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRWI"))
                    end else begin
                        `TRACE(level, ("CSRRW"))
                    end
                end
                INST_SFU_CSRRS: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRSI"))
                    end else begin
                        `TRACE(level, ("CSRRS"))
                    end
                end
                INST_SFU_CSRRC: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRCI"))
                    end else begin
                        `TRACE(level, ("CSRRC"))
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
    `ifdef EXT_F_ENABLE
        EX_FPU: begin
            case (INST_FPU_BITS'(op_type))
                INST_FPU_ADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FSUB.D"))
                        end else begin
                            `TRACE(level, ("FSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FADD.D"))
                        end else begin
                            `TRACE(level, ("FADD.S"))
                        end
                    end
                end
                INST_FPU_MADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FMSUB.D"))
                        end else begin
                            `TRACE(level, ("FMSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FMADD.D"))
                        end else begin
                            `TRACE(level, ("FMADD.S"))
                        end
                    end
                end
                INST_FPU_NMADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FNMSUB.D"))
                        end else begin
                            `TRACE(level, ("FNMSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FNMADD.D"))
                        end else begin
                            `TRACE(level, ("FNMADD.S"))
                        end
                    end
                end
                INST_FPU_MUL: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FMUL.D"))
                    end else begin
                        `TRACE(level, ("FMUL.S"))
                        end
                end
                INST_FPU_DIV: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FDIV.D"))
                    end else begin
                        `TRACE(level, ("FDIV.S"))
                        end
                end
                INST_FPU_SQRT: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FSQRT.D"))
                    end else begin
                        `TRACE(level, ("FSQRT.S"))
                    end
                end
                INST_FPU_CMP: begin
                    if (op_args.fpu.fmt[0]) begin
                        case (op_args.fpu.frm[1:0])
                        0:       `TRACE(level, ("FLE.D"))
                        1:       `TRACE(level, ("FLT.D"))
                        2:       `TRACE(level, ("FEQ.D"))
                        default: `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (op_args.fpu.frm[1:0])
                        0:       `TRACE(level, ("FLE.S"))
                        1:       `TRACE(level, ("FLT.S"))
                        2:       `TRACE(level, ("FEQ.S"))
                        default: `TRACE(level, ("?"))
                        endcase
                    end
                end
                INST_FPU_F2F: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FCVT.D.S"))
                    end else begin
                        `TRACE(level, ("FCVT.S.D"))
                    end
                end
                INST_FPU_F2I: begin
                    case (op_args.fpu.fmt)
                    2'b00: `TRACE(level, ("FCVT.W.S"))
                    2'b01: `TRACE(level, ("FCVT.W.D"))
                    2'b10: `TRACE(level, ("FCVT.L.S"))
                    2'b11: `TRACE(level, ("FCVT.L.D"))
                    endcase
                end
                INST_FPU_F2U: begin
                    case (op_args.fpu.fmt)
                    2'b00: `TRACE(level, ("FCVT.WU.S"))
                    2'b01: `TRACE(level, ("FCVT.WU.D"))
                    2'b10: `TRACE(level, ("FCVT.LU.S"))
                    2'b11: `TRACE(level, ("FCVT.LU.D"))
                    endcase
                end
                INST_FPU_I2F: begin
                    case (op_args.fpu.fmt)
                    2'b00: `TRACE(level, ("FCVT.S.W"))
                    2'b01: `TRACE(level, ("FCVT.D.W"))
                    2'b10: `TRACE(level, ("FCVT.S.L"))
                    2'b11: `TRACE(level, ("FCVT.D.L"))
                    endcase
                end
                INST_FPU_U2F: begin
                    case (op_args.fpu.fmt)
                    2'b00: `TRACE(level, ("FCVT.S.WU"))
                    2'b01: `TRACE(level, ("FCVT.D.WU"))
                    2'b10: `TRACE(level, ("FCVT.S.LU"))
                    2'b11: `TRACE(level, ("FCVT.D.LU"))
                    endcase
                end
                INST_FPU_MISC: begin
                    if (op_args.fpu.fmt[0]) begin
                        case (op_args.fpu.frm)
                            0: `TRACE(level, ("FSGNJ.D"))
                            1: `TRACE(level, ("FSGNJN.D"))
                            2: `TRACE(level, ("FSGNJX.D"))
                            3: `TRACE(level, ("FCLASS.D"))
                            4: `TRACE(level, ("FMV.X.D"))
                            5: `TRACE(level, ("FMV.D.X"))
                            6: `TRACE(level, ("FMIN.D"))
                            7: `TRACE(level, ("FMAX.D"))
                        endcase
                    end else begin
                        case (op_args.fpu.frm)
                            0: `TRACE(level, ("FSGNJ.S"))
                            1: `TRACE(level, ("FSGNJN.S"))
                            2: `TRACE(level, ("FSGNJX.S"))
                            3: `TRACE(level, ("FCLASS.S"))
                            4: `TRACE(level, ("FMV.X.S"))
                            5: `TRACE(level, ("FMV.S.X"))
                            6: `TRACE(level, ("FMIN.S"))
                            7: `TRACE(level, ("FMAX.S"))
                        endcase
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
    `endif
    `ifdef EXT_TCU_ENABLE
        EX_TCU: begin
            VX_tcu_pkg::trace_ex_op(level, op_type, op_args);
        end
    `endif
        default: `TRACE(level, ("?"))
        endcase
    endtask

    task trace_op_args(input int level,
                       input [EX_BITS-1:0] ex_type,
                       input [INST_OP_BITS-1:0] op_type,
                       input op_args_t op_args
    );
        case (ex_type)
        EX_ALU: begin
            `TRACE(level, ("use_PC=%b, use_imm=%b, imm=0x%0h", op_args.alu.use_PC, op_args.alu.use_imm, op_args.alu.imm))
        end
        EX_LSU: begin
            `TRACE(level, ("offset=0x%0h", op_args.lsu.offset))
        end
        EX_SFU: begin
            if (inst_sfu_is_csr(op_type)) begin
                `TRACE(level, ("addr=0x%0h, use_imm=%b, imm=0x%0h", op_args.csr.addr, op_args.csr.use_imm, op_args.csr.imm))
            end
        end
    `ifdef EXT_F_ENABLE
        EX_FPU: begin
            `TRACE(level, ("fmt=0x%0h, frm=0x%0h", op_args.fpu.fmt, op_args.fpu.frm))
        end
    `endif
        default:;
        endcase
    endtask

    task trace_base_dcr(input int level, input [VX_DCR_ADDR_WIDTH-1:0] addr);
        case (addr)
            `VX_DCR_BASE_STARTUP_ADDR0: `TRACE(level, ("STARTUP_ADDR0"))
            `VX_DCR_BASE_STARTUP_ADDR1: `TRACE(level, ("STARTUP_ADDR1"))
            `VX_DCR_BASE_STARTUP_ARG0:  `TRACE(level, ("STARTUP_ARG0"))
            `VX_DCR_BASE_STARTUP_ARG1:  `TRACE(level, ("STARTUP_ARG1"))
            `VX_DCR_BASE_MPM_CLASS:     `TRACE(level, ("MPM_CLASS"))
            default:                    `TRACE(level, ("?"))
        endcase
    endtask

`endif

endpackage

`endif // VX_TRACE_PKG_VH
