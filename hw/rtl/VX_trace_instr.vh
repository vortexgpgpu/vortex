`ifndef VX_TRACE_INSTR
`define VX_TRACE_INSTR

`include "VX_define.vh"

task trace_ex_type (
    input [`EX_BITS-1:0] ex_type
);
    case (ex_type)
        `EX_ALU: dpi_trace("ALU");     
        `EX_LSU: dpi_trace("LSU");
        `EX_CSR: dpi_trace("CSR");
        `EX_FPU: dpi_trace("FPU");
        `EX_GPU: dpi_trace("GPU");
        default: dpi_trace("NOP");
    endcase  
endtask

task trace_ex_op (
  input [`EX_BITS-1:0] ex_type,
  input [`INST_OP_BITS-1:0] op_type,
  input [`INST_MOD_BITS-1:0] op_mod
);
    case (ex_type)        
    `EX_ALU: begin
        if (`INST_ALU_IS_BR(op_mod)) begin
            case (`INST_BR_BITS'(op_type))
                `INST_BR_EQ:    dpi_trace("BEQ");
                `INST_BR_NE:    dpi_trace("BNE");
                `INST_BR_LT:    dpi_trace("BLT");
                `INST_BR_GE:    dpi_trace("BGE");
                `INST_BR_LTU:   dpi_trace("BLTU");
                `INST_BR_GEU:   dpi_trace("BGEU");           
                `INST_BR_JAL:   dpi_trace("JAL");
                `INST_BR_JALR:  dpi_trace("JALR");
                `INST_BR_ECALL: dpi_trace("ECALL");
                `INST_BR_EBREAK:dpi_trace("EBREAK");    
                `INST_BR_MRET:  dpi_trace("MRET");    
                `INST_BR_SRET:  dpi_trace("SRET");    
                `INST_BR_DRET:  dpi_trace("DRET");    
                default:   dpi_trace("?");
            endcase                
        end else if (`INST_ALU_IS_MUL(op_mod)) begin
            case (`INST_MUL_BITS'(op_type))
                `INST_MUL_MUL:   dpi_trace("MUL");
                `INST_MUL_MULH:  dpi_trace("MULH");
                `INST_MUL_MULHSU:dpi_trace("MULHSU");
                `INST_MUL_MULHU: dpi_trace("MULHU");
                `INST_MUL_DIV:   dpi_trace("DIV");
                `INST_MUL_DIVU:  dpi_trace("DIVU");
                `INST_MUL_REM:   dpi_trace("REM");
                `INST_MUL_REMU:  dpi_trace("REMU");
                default:    dpi_trace("?");
            endcase
        end else begin
            case (`INST_ALU_BITS'(op_type))
                `INST_ALU_ADD:   dpi_trace("ADD");
                `INST_ALU_SUB:   dpi_trace("SUB");
                `INST_ALU_SLL:   dpi_trace("SLL");
                `INST_ALU_SRL:   dpi_trace("SRL");
                `INST_ALU_SRA:   dpi_trace("SRA");
                `INST_ALU_SLT:   dpi_trace("SLT");
                `INST_ALU_SLTU:  dpi_trace("SLTU");
                `INST_ALU_XOR:   dpi_trace("XOR");
                `INST_ALU_OR:    dpi_trace("OR");
                `INST_ALU_AND:   dpi_trace("AND");
                `INST_ALU_LUI:   dpi_trace("LUI");
                `INST_ALU_AUIPC: dpi_trace("AUIPC");
                default:    dpi_trace("?");
            endcase         
        end
    end
    `EX_LSU: begin
        if (op_mod == 0) begin
            case (`INST_LSU_BITS'(op_type))
                `INST_LSU_LB: dpi_trace("LB");
                `INST_LSU_LH: dpi_trace("LH");
                `INST_LSU_LW: dpi_trace("LW");
                `INST_LSU_LBU:dpi_trace("LBU");
                `INST_LSU_LHU:dpi_trace("LHU");
                `INST_LSU_SB: dpi_trace("SB");
                `INST_LSU_SH: dpi_trace("SH");
                `INST_LSU_SW: dpi_trace("SW");
                default: dpi_trace("?");
            endcase
        end else if (op_mod == 1) begin
            case (`INST_FENCE_BITS'(op_type))
                `INST_FENCE_D: dpi_trace("DFENCE");
                `INST_FENCE_I: dpi_trace("IFENCE");
                default: dpi_trace("?");
            endcase
        end
    end
    `EX_CSR: begin
        case (`INST_CSR_BITS'(op_type))
            `INST_CSR_RW: dpi_trace("CSRW");
            `INST_CSR_RS: dpi_trace("CSRS");
            `INST_CSR_RC: dpi_trace("CSRC");
            default: dpi_trace("?");
        endcase
    end
    `EX_FPU: begin
        case (`INST_FPU_BITS'(op_type))
            `INST_FPU_ADD:   dpi_trace("ADD");
            `INST_FPU_SUB:   dpi_trace("SUB");
            `INST_FPU_MUL:   dpi_trace("MUL");
            `INST_FPU_DIV:   dpi_trace("DIV");
            `INST_FPU_SQRT:  dpi_trace("SQRT");
            `INST_FPU_MADD:  dpi_trace("MADD");
            `INST_FPU_NMSUB: dpi_trace("NMSUB");
            `INST_FPU_NMADD: dpi_trace("NMADD");                
            `INST_FPU_CVTWS: dpi_trace("CVTWS");
            `INST_FPU_CVTWUS:dpi_trace("CVTWUS");
            `INST_FPU_CVTSW: dpi_trace("CVTSW");
            `INST_FPU_CVTSWU:dpi_trace("CVTSWU");
            `INST_FPU_CLASS: dpi_trace("CLASS");
            `INST_FPU_CMP:   dpi_trace("CMP");
            `INST_FPU_MISC: begin
                case (op_mod)
                0: dpi_trace("SGNJ");   
                1: dpi_trace("SGNJN");
                2: dpi_trace("SGNJX");
                3: dpi_trace("MIN");
                4: dpi_trace("MAX");
                5: dpi_trace("MVXW");
                6: dpi_trace("MVWX");
                endcase
            end 
            default:    dpi_trace("?");
        endcase
    end
    `EX_GPU: begin
        case (`INST_GPU_BITS'(op_type))
            `INST_GPU_TMC:   dpi_trace("TMC");
            `INST_GPU_WSPAWN:dpi_trace("WSPAWN");
            `INST_GPU_SPLIT: dpi_trace("SPLIT");
            `INST_GPU_JOIN:  dpi_trace("JOIN");
            `INST_GPU_BAR:   dpi_trace("BAR");
            `INST_GPU_PRED:  dpi_trace("PRED");
            `INST_GPU_TEX:   dpi_trace("TEX");
            default:    dpi_trace("?");
        endcase
    end    
    default: dpi_trace("?");
    endcase 
endtask

`endif
