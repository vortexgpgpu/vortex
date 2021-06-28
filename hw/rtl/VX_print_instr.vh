`ifndef VX_PRINT_INSTR
`define VX_PRINT_INSTR

`include "VX_define.vh"

task print_ex_type (
    input [`EX_BITS-1:0] ex_type
);
    case (ex_type)
        `EX_ALU: $write("ALU");     
        `EX_LSU: $write("LSU");
        `EX_CSR: $write("CSR");
        `EX_FPU: $write("FPU");
        `EX_GPU: $write("GPU");
        default: $write("NOP");
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
                `BR_EQ:    $write("BEQ");
                `BR_NE:    $write("BNE");
                `BR_LT:    $write("BLT");
                `BR_GE:    $write("BGE");
                `BR_LTU:   $write("BLTU");
                `BR_GEU:   $write("BGEU");           
                `BR_JAL:   $write("JAL");
                `BR_JALR:  $write("JALR");
                `BR_ECALL: $write("ECALL");
                `BR_EBREAK:$write("EBREAK");    
                `BR_MRET:  $write("MRET");    
                `BR_SRET:  $write("SRET");    
                `BR_DRET:  $write("DRET");    
                default:    $write("?");
            endcase                
        end else if (`ALU_IS_MUL(op_mod)) begin
            case (`MUL_BITS'(op_type))
                `MUL_MUL:   $write("MUL");
                `MUL_MULH:  $write("MULH");
                `MUL_MULHSU:$write("MULHSU");
                `MUL_MULHU: $write("MULHU");
                `MUL_DIV:   $write("DIV");
                `MUL_DIVU:  $write("DIVU");
                `MUL_REM:   $write("REM");
                `MUL_REMU:  $write("REMU");
                default:    $write("?");
            endcase
        end else begin
            case (`ALU_BITS'(op_type))
                `ALU_ADD:   $write("ADD");
                `ALU_SUB:   $write("SUB");
                `ALU_SLL:   $write("SLL");
                `ALU_SRL:   $write("SRL");
                `ALU_SRA:   $write("SRA");
                `ALU_SLT:   $write("SLT");
                `ALU_SLTU:  $write("SLTU");
                `ALU_XOR:   $write("XOR");
                `ALU_OR:    $write("OR");
                `ALU_AND:   $write("AND");
                `ALU_LUI:   $write("LUI");
                `ALU_AUIPC: $write("AUIPC");
                default:    $write("?");
            endcase         
        end
    end
    `EX_LSU: begin
        case (`LSU_BITS'(op_type))
            `LSU_LB: $write("LB");
            `LSU_LH: $write("LH");
            `LSU_LW: $write("LW");
            `LSU_LBU:$write("LBU");
            `LSU_LHU:$write("LHU");
            `LSU_SB: $write("SB");
            `LSU_SH: $write("SH");
            `LSU_SW: $write("SW");
            default: $write("?");
        endcase
    end
    `EX_CSR: begin
        case (`CSR_BITS'(op_type))
            `CSR_RW: $write("CSRW");
            `CSR_RS: $write("CSRS");
            `CSR_RC: $write("CSRC");
            default: $write("?");
        endcase
    end
    `EX_FPU: begin
        case (`FPU_BITS'(op_type))
            `FPU_ADD:   $write("ADD");
            `FPU_SUB:   $write("SUB");
            `FPU_MUL:   $write("MUL");
            `FPU_DIV:   $write("DIV");
            `FPU_SQRT:  $write("SQRT");
            `FPU_MADD:  $write("MADD");
            `FPU_NMSUB: $write("NMSUB");
            `FPU_NMADD: $write("NMADD");                
            `FPU_CVTWS: $write("CVTWS");
            `FPU_CVTWUS:$write("CVTWUS");
            `FPU_CVTSW: $write("CVTSW");
            `FPU_CVTSWU:$write("CVTSWU");
            `FPU_CLASS: $write("CLASS");
            `FPU_CMP:   $write("CMP");
            `FPU_MISC: begin
                case (op_mod)
                0: $write("SGNJ");   
                1: $write("SGNJN");
                2: $write("SGNJX");
                3: $write("MIN");
                4: $write("MAX");
                5: $write("MVXW");
                6: $write("MVWX");
                endcase
            end 
            default:    $write("?");
        endcase
    end
    `EX_GPU: begin
        case (`GPU_BITS'(op_type))
            `GPU_TMC:   $write("TMC");
            `GPU_WSPAWN:$write("WSPAWN");
            `GPU_SPLIT: $write("SPLIT");
            `GPU_JOIN:  $write("JOIN");
            `GPU_BAR:   $write("BAR");
            default:    $write("?");
        endcase
    end    
    default: $write("?");
    endcase 
endtask

`endif
