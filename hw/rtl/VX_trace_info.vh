`ifndef VX_TRACE_INFO_VH
`define VX_TRACE_INFO_VH

`include "VX_define.vh"

`define TRACE_EX_TYPE(level, ex_type) \
    case (ex_type) \
        `EX_ALU: `TRACE(level, ("ALU")); \
        `EX_LSU: `TRACE(level, ("LSU")); \
        `EX_CSR: `TRACE(level, ("CSR")); \
        `EX_FPU: `TRACE(level, ("FPU")); \
        `EX_GPU: `TRACE(level, ("GPU")); \
        default: `TRACE(level, ("?")); \
    endcase

`define TRACE_EX_OP(level, ex_type, op_type, op_mod) \
    case (ex_type)  \
    `EX_ALU: begin  \
        if (`INST_ALU_IS_BR(op_mod)) begin \
            case (`INST_BR_BITS'(op_type)) \
                `INST_BR_EQ:    `TRACE(level, ("BEQ")); \
                `INST_BR_NE:    `TRACE(level, ("BNE")); \
                `INST_BR_LT:    `TRACE(level, ("BLT")); \
                `INST_BR_GE:    `TRACE(level, ("BGE")); \
                `INST_BR_LTU:   `TRACE(level, ("BLTU")); \
                `INST_BR_GEU:   `TRACE(level, ("BGEU")); \
                `INST_BR_JAL:   `TRACE(level, ("JAL")); \
                `INST_BR_JALR:  `TRACE(level, ("JALR")); \
                `INST_BR_ECALL: `TRACE(level, ("ECALL")); \
                `INST_BR_EBREAK:`TRACE(level, ("EBREAK")); \
                `INST_BR_URET:  `TRACE(level, ("URET")); \
                `INST_BR_SRET:  `TRACE(level, ("SRET")); \
                `INST_BR_MRET:  `TRACE(level, ("MRET")); \
                default:        `TRACE(level, ("?")); \
            endcase \
        end else if (`INST_ALU_IS_M(op_mod)) begin \
            if (`INST_ALU_IS_W(op_mod)) begin \
                case (`INST_M_BITS'(op_type)) \
                    `INST_M_MUL:  `TRACE(level, ("MULW")); \
                    `INST_M_DIV:  `TRACE(level, ("DIVW")); \
                    `INST_M_DIVU: `TRACE(level, ("DIVUW")); \
                    `INST_M_REM:  `TRACE(level, ("REMW")); \
                    `INST_M_REMU: `TRACE(level, ("REMUW")); \
                    default:      `TRACE(level, ("?")); \
                endcase \
            end else begin \
                case (`INST_M_BITS'(op_type)) \
                    `INST_M_MUL:   `TRACE(level, ("MUL")); \
                    `INST_M_MULH:  `TRACE(level, ("MULH")); \
                    `INST_M_MULHSU:`TRACE(level, ("MULHSU")); \
                    `INST_M_MULHU: `TRACE(level, ("MULHU")); \
                    `INST_M_DIV:   `TRACE(level, ("DIV")); \
                    `INST_M_DIVU:  `TRACE(level, ("DIVU")); \
                    `INST_M_REM:   `TRACE(level, ("REM")); \
                    `INST_M_REMU:  `TRACE(level, ("REMU")); \
                    default:       `TRACE(level, ("?")); \
                endcase \
            end \
        end else begin \
            if (`INST_ALU_IS_W(op_mod)) begin \
                case (`INST_ALU_BITS'(op_type)) \
                    `INST_ALU_ADD: `TRACE(level, ("ADD_W")); \
                    `INST_ALU_SUB: `TRACE(level, ("SUB_W")); \
                    `INST_ALU_SLL: `TRACE(level, ("SLL_W")); \
                    `INST_ALU_SRL: `TRACE(level, ("SRL_W")); \
                    `INST_ALU_SRA: `TRACE(level, ("SRA_W")); \
                    default:       `TRACE(level, ("?")); \
                endcase \
            end else begin \
                case (`INST_ALU_BITS'(op_type)) \
                    `INST_ALU_ADD:   `TRACE(level, ("ADD")); \
                    `INST_ALU_SUB:   `TRACE(level, ("SUB")); \
                    `INST_ALU_SLL:   `TRACE(level, ("SLL")); \
                    `INST_ALU_SRL:   `TRACE(level, ("SRL")); \
                    `INST_ALU_SRA:   `TRACE(level, ("SRA")); \
                    `INST_ALU_SLT:   `TRACE(level, ("SLT")); \
                    `INST_ALU_SLTU:  `TRACE(level, ("SLTU")); \
                    `INST_ALU_XOR:   `TRACE(level, ("XOR")); \
                    `INST_ALU_OR:    `TRACE(level, ("OR")); \
                    `INST_ALU_AND:   `TRACE(level, ("AND")); \
                    `INST_ALU_LUI:   `TRACE(level, ("LUI")); \
                    `INST_ALU_AUIPC: `TRACE(level, ("AUIPC")); \
                    default:         `TRACE(level, ("?")); \
                endcase \
            end \
        end \
    end \
    `EX_LSU: begin \
        if (op_mod == 0) begin \
            case (`INST_LSU_BITS'(op_type)) \
                `INST_LSU_LB: `TRACE(level, ("LB")); \
                `INST_LSU_LH: `TRACE(level, ("LH")); \
                `INST_LSU_LW: `TRACE(level, ("LW")); \
                `INST_LSU_LD: `TRACE(level, ("LD")); \
                `INST_LSU_LBU:`TRACE(level, ("LBU")); \
                `INST_LSU_LHU:`TRACE(level, ("LHU")); \
                `INST_LSU_LWU:`TRACE(level, ("LWU")); \
                `INST_LSU_SB: `TRACE(level, ("SB")); \
                `INST_LSU_SH: `TRACE(level, ("SH")); \
                `INST_LSU_SW: `TRACE(level, ("SW")); \
                `INST_LSU_SD: `TRACE(level, ("SD")); \
                default:      `TRACE(level, ("?")); \
            endcase \
        end else if (op_mod == 1) begin \
            case (`INST_FENCE_BITS'(op_type)) \
                `INST_FENCE_D: `TRACE(level, ("DFENCE")); \
                `INST_FENCE_I: `TRACE(level, ("IFENCE")); \
                default:       `TRACE(level, ("?")); \
            endcase \
        end \
    end \
    `EX_CSR: begin \
        case (`INST_CSR_BITS'(op_type)) \
            `INST_CSR_RW: `TRACE(level, ("CSRW")); \
            `INST_CSR_RS: `TRACE(level, ("CSRS")); \
            `INST_CSR_RC: `TRACE(level, ("CSRC")); \
            default:      `TRACE(level, ("?")); \
        endcase \
    end \
    `EX_FPU: begin \
        case (`INST_FPU_BITS'(op_type)) \
            `INST_FPU_ADD:   `TRACE(level, ("ADD")); \
            `INST_FPU_SUB:   `TRACE(level, ("SUB")); \
            `INST_FPU_MUL:   `TRACE(level, ("MUL")); \
            `INST_FPU_DIV:   `TRACE(level, ("DIV")); \
            `INST_FPU_SQRT:  `TRACE(level, ("SQRT")); \
            `INST_FPU_MADD:  `TRACE(level, ("MADD")); \
            `INST_FPU_NMSUB: `TRACE(level, ("NMSUB")); \
            `INST_FPU_NMADD: `TRACE(level, ("NMADD")); \
            `INST_FPU_CVTWX: `TRACE(level, ("CVT.W.X")); \
            `INST_FPU_CVTWUX:`TRACE(level, ("CVT.WU.X")); \
            `INST_FPU_CVTXW: `TRACE(level, ("CVT.X.W")); \
            `INST_FPU_CVTXWU:`TRACE(level, ("CVT.X.WU")); \
            `INST_FPU_NCP: begin \
                case (op_mod) \
                    0: `TRACE(level, ("SGNJ")); \
                    1: `TRACE(level, ("SGNJN")); \
                    2: `TRACE(level, ("SGNJX")); \
                    3: `TRACE(level, ("MIN")); \
                    4: `TRACE(level, ("MAX")); \
                    5: `TRACE(level, ("MVXW")); \
                    6: `TRACE(level, ("MVWX")); \
                    7: `TRACE(level, ("CLASS")); \
                    8: `TRACE(level, ("FLE")); \
                    9: `TRACE(level, ("FLT")); \
                   10: `TRACE(level, ("FEQ")); \
                endcase \
            end \
            default:   `TRACE(level, ("?")); \
        endcase \
    end \
    `EX_GPU: begin \
        case (`INST_GPU_BITS'(op_type)) \
            `INST_GPU_TMC:   `TRACE(level, ("TMC")); \
            `INST_GPU_WSPAWN:`TRACE(level, ("WSPAWN")); \
            `INST_GPU_SPLIT: `TRACE(level, ("SPLIT")); \
            `INST_GPU_JOIN:  `TRACE(level, ("JOIN")); \
            `INST_GPU_BAR:   `TRACE(level, ("BAR")); \
            `INST_GPU_PRED:  `TRACE(level, ("PRED")); \
            `INST_GPU_TEX:   `TRACE(level, ("TEX")); \
            `INST_GPU_RASTER:`TRACE(level, ("RASTER")); \
            `INST_GPU_ROP:   `TRACE(level, ("ROP")); \
            `INST_GPU_IMADD: `TRACE(level, ("IMADD")); \
            default:         `TRACE(level, ("?")); \
        endcase \
    end \
    default: `TRACE(level, ("?")); \
    endcase

`define TRACE_BASE_DCR(level, addr) \
    case (addr) \
        `DCR_BASE_STARTUP_ADDR0: `TRACE(level, ("STARTUP_ADDR0")); \
        `DCR_BASE_STARTUP_ADDR1: `TRACE(level, ("STARTUP_ADDR1")); \
        `DCR_BASE_MPM_CLASS:     `TRACE(level, ("MPM_CLASS")); \
        default:                 `TRACE(level, ("?")); \
    endcase

`endif // VX_TRACE_INFO_VH
