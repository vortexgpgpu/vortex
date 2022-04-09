`ifndef VX_TRACE_INSTR_VH
`define VX_TRACE_INSTR_VH

`include "VX_define.vh"

task trace_ex_type (
    input int            level,
    input [`EX_BITS-1:0] ex_type
);
    case (ex_type)
        `EX_ALU: dpi_trace(level, "ALU");     
        `EX_LSU: dpi_trace(level, "LSU");
        `EX_CSR: dpi_trace(level, "CSR");
        `EX_FPU: dpi_trace(level, "FPU");
        `EX_GPU: dpi_trace(level, "GPU");
        default: dpi_trace(level, "NOP");
    endcase  
endtask

task trace_ex_op (
  input int                  level,
  input [`EX_BITS-1:0]       ex_type,
  input [`INST_OP_BITS-1:0]  op_type,
  input [`INST_MOD_BITS-1:0] op_mod
);
    case (ex_type)        
    `EX_ALU: begin
        if (`INST_ALU_IS_BR(op_mod)) begin
            case (`INST_BR_BITS'(op_type))
                `INST_BR_EQ:    dpi_trace(level, "BEQ");
                `INST_BR_NE:    dpi_trace(level, "BNE");
                `INST_BR_LT:    dpi_trace(level, "BLT");
                `INST_BR_GE:    dpi_trace(level, "BGE");
                `INST_BR_LTU:   dpi_trace(level, "BLTU");
                `INST_BR_GEU:   dpi_trace(level, "BGEU");           
                `INST_BR_JAL:   dpi_trace(level, "JAL");
                `INST_BR_JALR:  dpi_trace(level, "JALR");
                `INST_BR_ECALL: dpi_trace(level, "ECALL");
                `INST_BR_EBREAK:dpi_trace(level, "EBREAK");    
                `INST_BR_URET:  dpi_trace(level, "URET");    
                `INST_BR_SRET:  dpi_trace(level, "SRET");    
                `INST_BR_MRET:  dpi_trace(level, "MRET");    
                default:   dpi_trace(level, "?");
            endcase                
        end else if (`INST_ALU_IS_MUL(op_mod)) begin
            case (`INST_MUL_BITS'(op_type))
                `INST_MUL_MUL:   dpi_trace(level, "MUL");
                `INST_MUL_MULH:  dpi_trace(level, "MULH");
                `INST_MUL_MULHSU:dpi_trace(level, "MULHSU");
                `INST_MUL_MULHU: dpi_trace(level, "MULHU");
                `INST_MUL_DIV:   dpi_trace(level, "DIV");
                `INST_MUL_DIVU:  dpi_trace(level, "DIVU");
                `INST_MUL_REM:   dpi_trace(level, "REM");
                `INST_MUL_REMU:  dpi_trace(level, "REMU");
                default:    dpi_trace(level, "?");
            endcase
        end else begin
            case (`INST_ALU_BITS'(op_type))
                `INST_ALU_ADD:   dpi_trace(level, "ADD");
                `INST_ALU_SUB:   dpi_trace(level, "SUB");
                `INST_ALU_SLL:   dpi_trace(level, "SLL");
                `INST_ALU_SRL:   dpi_trace(level, "SRL");
                `INST_ALU_SRA:   dpi_trace(level, "SRA");
                `INST_ALU_SLT:   dpi_trace(level, "SLT");
                `INST_ALU_SLTU:  dpi_trace(level, "SLTU");
                `INST_ALU_XOR:   dpi_trace(level, "XOR");
                `INST_ALU_OR:    dpi_trace(level, "OR");
                `INST_ALU_AND:   dpi_trace(level, "AND");
                `INST_ALU_LUI:   dpi_trace(level, "LUI");
                `INST_ALU_AUIPC: dpi_trace(level, "AUIPC");
                default:    dpi_trace(level, "?");
            endcase         
        end
    end
    `EX_LSU: begin
        if (op_mod == 0) begin
            case (`INST_LSU_BITS'(op_type))
                `INST_LSU_LB: dpi_trace(level, "LB");
                `INST_LSU_LH: dpi_trace(level, "LH");
                `INST_LSU_LW: dpi_trace(level, "LW");
                `INST_LSU_LBU:dpi_trace(level, "LBU");
                `INST_LSU_LHU:dpi_trace(level, "LHU");
                `INST_LSU_SB: dpi_trace(level, "SB");
                `INST_LSU_SH: dpi_trace(level, "SH");
                `INST_LSU_SW: dpi_trace(level, "SW");
                default: dpi_trace(level, "?");
            endcase
        end else if (op_mod == 1) begin
            case (`INST_FENCE_BITS'(op_type))
                `INST_FENCE_D: dpi_trace(level, "DFENCE");
                `INST_FENCE_I: dpi_trace(level, "IFENCE");
                default: dpi_trace(level, "?");
            endcase
        end
    end
    `EX_CSR: begin
        case (`INST_CSR_BITS'(op_type))
            `INST_CSR_RW: dpi_trace(level, "CSRW");
            `INST_CSR_RS: dpi_trace(level, "CSRS");
            `INST_CSR_RC: dpi_trace(level, "CSRC");
            default: dpi_trace(level, "?");
        endcase
    end
    `EX_FPU: begin
        case (`INST_FPU_BITS'(op_type))
            `INST_FPU_ADD:   dpi_trace(level, "ADD");
            `INST_FPU_SUB:   dpi_trace(level, "SUB");
            `INST_FPU_MUL:   dpi_trace(level, "MUL");
            `INST_FPU_DIV:   dpi_trace(level, "DIV");
            `INST_FPU_SQRT:  dpi_trace(level, "SQRT");
            `INST_FPU_MADD:  dpi_trace(level, "MADD");
            `INST_FPU_NMSUB: dpi_trace(level, "NMSUB");
            `INST_FPU_NMADD: dpi_trace(level, "NMADD");                
            `INST_FPU_CVTWS: dpi_trace(level, "CVTWS");
            `INST_FPU_CVTWUS:dpi_trace(level, "CVTWUS");
            `INST_FPU_CVTSW: dpi_trace(level, "CVTSW");
            `INST_FPU_CVTSWU:dpi_trace(level, "CVTSWU");
            `INST_FPU_CLASS: dpi_trace(level, "CLASS");
            `INST_FPU_CMP:   dpi_trace(level, "CMP");
            `INST_FPU_MISC: begin
                case (op_mod)
                    0: dpi_trace(level, "SGNJ");
                    1: dpi_trace(level, "SGNJN");
                    2: dpi_trace(level, "SGNJX");
                    3: dpi_trace(level, "MIN");
                    4: dpi_trace(level, "MAX");
                    5: dpi_trace(level, "MVXW");
                    6: dpi_trace(level, "MVWX");
                endcase
            end 
            default:   dpi_trace(level, "?");
        endcase
    end
    `EX_GPU: begin
        case (`INST_GPU_BITS'(op_type))
            `INST_GPU_TMC:   dpi_trace(level, "TMC");
            `INST_GPU_WSPAWN:dpi_trace(level, "WSPAWN");
            `INST_GPU_SPLIT: dpi_trace(level, "SPLIT");
            `INST_GPU_JOIN:  dpi_trace(level, "JOIN");
            `INST_GPU_BAR:   dpi_trace(level, "BAR");
            `INST_GPU_PRED:  dpi_trace(level, "PRED");
            `INST_GPU_TEX:   dpi_trace(level, "TEX");
            `INST_GPU_RASTER:dpi_trace(level, "RASTER");
            `INST_GPU_ROP:   dpi_trace(level, "ROP");
            `INST_GPU_IMADD: dpi_trace(level, "IMADD");
            default:         dpi_trace(level, "?");
        endcase
    end    
    default: dpi_trace(level, "?");
    endcase 
endtask

`endif
