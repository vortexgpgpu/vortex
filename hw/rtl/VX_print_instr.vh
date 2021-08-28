`ifndef VX_PRINT_INSTR
`define VX_PRINT_INSTR

`include "VX_define.vh"

task print_ex_type (
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

task print_ex_op (
  input [`EX_BITS-1:0] ex_type,
  input [`OP_BITS-1:0] op_type,
  input [`MOD_BITS-1:0] op_mod
);
    case (ex_type)        
    `EX_ALU: begin
        if (`ALU_IS_BR(op_mod)) begin
            case (`BR_BITS'(op_type))
                `BR_EQ:    dpi_trace("BEQ");
                `BR_NE:    dpi_trace("BNE");
                `BR_LT:    dpi_trace("BLT");
                `BR_GE:    dpi_trace("BGE");
                `BR_LTU:   dpi_trace("BLTU");
                `BR_GEU:   dpi_trace("BGEU");           
                `BR_JAL:   dpi_trace("JAL");
                `BR_JALR:  dpi_trace("JALR");
                `BR_ECALL: dpi_trace("ECALL");
                `BR_EBREAK:dpi_trace("EBREAK");    
                `BR_MRET:  dpi_trace("MRET");    
                `BR_SRET:  dpi_trace("SRET");    
                `BR_DRET:  dpi_trace("DRET");    
                default:   dpi_trace("?");
            endcase                
        end else if (`ALU_IS_MUL(op_mod)) begin
            case (`MUL_BITS'(op_type))
                `MUL_MUL:   dpi_trace("MUL");
                `MUL_MULH:  dpi_trace("MULH");
                `MUL_MULHSU:dpi_trace("MULHSU");
                `MUL_MULHU: dpi_trace("MULHU");
                `MUL_DIV:   dpi_trace("DIV");
                `MUL_DIVU:  dpi_trace("DIVU");
                `MUL_REM:   dpi_trace("REM");
                `MUL_REMU:  dpi_trace("REMU");
                default:    dpi_trace("?");
            endcase
        end else begin
            case (`ALU_BITS'(op_type))
                `ALU_ADD:   dpi_trace("ADD");
                `ALU_SUB:   dpi_trace("SUB");
                `ALU_SLL:   dpi_trace("SLL");
                `ALU_SRL:   dpi_trace("SRL");
                `ALU_SRA:   dpi_trace("SRA");
                `ALU_SLT:   dpi_trace("SLT");
                `ALU_SLTU:  dpi_trace("SLTU");
                `ALU_XOR:   dpi_trace("XOR");
                `ALU_OR:    dpi_trace("OR");
                `ALU_AND:   dpi_trace("AND");
                `ALU_LUI:   dpi_trace("LUI");
                `ALU_AUIPC: dpi_trace("AUIPC");
                default:    dpi_trace("?");
            endcase         
        end
    end
    `EX_LSU: begin
        if (op_mod == 0) begin
            case (`LSU_BITS'(op_type))
                `LSU_LB: dpi_trace("LB");
                `LSU_LH: dpi_trace("LH");
                `LSU_LW: dpi_trace("LW");
                `LSU_LBU:dpi_trace("LBU");
                `LSU_LHU:dpi_trace("LHU");
                `LSU_SB: dpi_trace("SB");
                `LSU_SH: dpi_trace("SH");
                `LSU_SW: dpi_trace("SW");
                default: dpi_trace("?");
            endcase
        end else if (op_mod == 1) begin
            case (`FENCE_BITS'(op_type))
                `FENCE_D: dpi_trace("DFENCE");
                `FENCE_I: dpi_trace("IFENCE");
                default: dpi_trace("?");
            endcase
        end
    end
    `EX_CSR: begin
        case (`CSR_BITS'(op_type))
            `CSR_RW: dpi_trace("CSRW");
            `CSR_RS: dpi_trace("CSRS");
            `CSR_RC: dpi_trace("CSRC");
            default: dpi_trace("?");
        endcase
    end
    `EX_FPU: begin
        case (`FPU_BITS'(op_type))
            `FPU_ADD:   dpi_trace("ADD");
            `FPU_SUB:   dpi_trace("SUB");
            `FPU_MUL:   dpi_trace("MUL");
            `FPU_DIV:   dpi_trace("DIV");
            `FPU_SQRT:  dpi_trace("SQRT");
            `FPU_MADD:  dpi_trace("MADD");
            `FPU_NMSUB: dpi_trace("NMSUB");
            `FPU_NMADD: dpi_trace("NMADD");                
            `FPU_CVTWS: dpi_trace("CVTWS");
            `FPU_CVTWUS:dpi_trace("CVTWUS");
            `FPU_CVTSW: dpi_trace("CVTSW");
            `FPU_CVTSWU:dpi_trace("CVTSWU");
            `FPU_CLASS: dpi_trace("CLASS");
            `FPU_CMP:   dpi_trace("CMP");
            `FPU_MISC: begin
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
        case (`GPU_BITS'(op_type))
            `GPU_TMC:   dpi_trace("TMC");
            `GPU_WSPAWN:dpi_trace("WSPAWN");
            `GPU_SPLIT: dpi_trace("SPLIT");
            `GPU_JOIN:  dpi_trace("JOIN");
            `GPU_BAR:   dpi_trace("BAR");
            default:    dpi_trace("?");
        endcase
    end    
    default: dpi_trace("?");
    endcase 
endtask

`endif
