`include "VX_define.vh"
`include "VX_print_instr.vh"

module VX_decode  #(
    parameter CORE_ID = 0
) (
    input  wire         clk,
    input  wire         reset,

    // inputs
    VX_ifetch_rsp_if    ifetch_rsp_if,

    // outputs      
    VX_decode_if        decode_if,
    VX_wstall_if        wstall_if,
    VX_join_if          join_if
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    wire [31:0] instr = ifetch_rsp_if.instr;

    reg [`ALU_BITS-1:0] alu_op;
    reg [`BR_BITS-1:0]  br_op;    
    reg [`LSU_BITS-1:0] lsu_op;
    reg [`CSR_BITS-1:0] csr_op;
    reg [`MUL_BITS-1:0] mul_op;
    reg [`FPU_BITS-1:0] fpu_op;
    reg [`GPU_BITS-1:0] gpu_op;

    reg [19:0] upper_imm;
    reg [31:0] jalx_offset;
    reg [31:0] src2_imm;

    wire [6:0] opcode = instr[6:0];  
    wire [2:0] func3  = instr[14:12];
    wire [6:0] func7  = instr[31:25];
    wire [11:0] u_12  = instr[31:20]; 

    wire [4:0] rd  = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];     
    wire [4:0] rs3 = instr[31:27];

    // opcode types
    wire is_rtype = (opcode == `INST_R);
    wire is_ltype = (opcode == `INST_L);
    wire is_itype = (opcode == `INST_I); 
    wire is_stype = (opcode == `INST_S);
    wire is_btype = (opcode == `INST_B);    
    wire is_jal   = (opcode == `INST_JAL);
    wire is_jalr  = (opcode == `INST_JALR);
    wire is_lui   = (opcode == `INST_LUI);
    wire is_auipc = (opcode == `INST_AUIPC);    
    wire is_jals  = (opcode == `INST_SYS) && (func3 == 0);
    wire is_csr   = (opcode == `INST_SYS) && (func3 != 0);
    wire is_gpu   = (opcode == `INST_GPU);  
        
    // upper immediate

    always @(*) begin
        case (opcode)
            `INST_LUI:   upper_imm = {func7, rs2, rs1, func3};
            `INST_AUIPC: upper_imm = {func7, rs2, rs1, func3};
            default:     upper_imm = 20'h0;
        endcase
    end  

    // I-type immediate

    wire alu_shift_i          = (func3 == 3'h1) || (func3 == 3'h5);
    wire [11:0] alu_shift_imm = {{7{1'b0}}, rs2};
    wire [11:0] alu_imm       = alu_shift_i ? alu_shift_imm : u_12;
    always @(*) begin
        case (opcode)
            `INST_I:  src2_imm = {{20{alu_imm[11]}}, alu_imm};
            `INST_S,
            `INST_FS: src2_imm = {{20{func7[6]}}, func7, rd};
            `INST_L, 
            `INST_FL: src2_imm = {{20{u_12[11]}}, u_12};
            `INST_B:  src2_imm = {{20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0};
            default: src2_imm = 'x;
        endcase
    end     

    // JAL

    wire [20:0] jal_imm     = {instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
    wire [31:0] jal_offset  = {{11{jal_imm[20]}}, jal_imm};
    wire [11:0] jalr_imm    = {func7, rs2};
    wire [31:0] jalr_offset = {{20{jalr_imm[11]}}, jalr_imm};
    always @(*) begin        
        case (opcode)
            `INST_JAL:  jalx_offset = jal_offset;
            `INST_JALR: jalx_offset = jalr_offset;
            default:    jalx_offset = 32'd4;
        endcase
    end 

    // BRANCH

    wire is_br = (is_btype || is_jal || is_jalr || is_jals);

    always @(*) begin
        br_op = `BR_OTHER;
        case (opcode)
            `INST_B: begin
                case (func3)
                    3'h0: br_op = `BR_EQ;
                    3'h1: br_op = `BR_NE;
                    3'h4: br_op = `BR_LT;
                    3'h5: br_op = `BR_GE;
                    3'h6: br_op = `BR_LTU;
                    3'h7: br_op = `BR_GEU;
                    default:; 
                endcase
            end
            `INST_JAL:  br_op = `BR_JAL;
            `INST_JALR: br_op = `BR_JALR;
            `INST_SYS: begin
                if (is_jals && u_12 == 12'h000) br_op = `BR_ECALL;
                if (is_jals && u_12 == 12'h001) br_op = `BR_EBREAK;             
                if (is_jals && u_12 == 12'h302) br_op = `BR_MRET;
                if (is_jals && u_12 == 12'h102) br_op = `BR_SRET;
                if (is_jals && u_12 == 12'h7B2) br_op = `BR_DRET;
            end
            default:;
        endcase
    end
    
    // ALU    

    always @(*) begin
        alu_op = `ALU_OTHER;
        if (is_lui) begin
            alu_op = `ALU_LUI;
        end else if (is_auipc) begin
            alu_op = `ALU_AUIPC;
        end else if (is_itype || is_rtype) begin
            case (func3)
                3'h0: alu_op = (is_rtype && func7 == 7'h20) ? `ALU_SUB : `ALU_ADD;
                3'h1: alu_op = `ALU_SLL;
                3'h2: alu_op = `ALU_SLT;
                3'h3: alu_op = `ALU_SLTU;
                3'h4: alu_op = `ALU_XOR;
                3'h5: alu_op = (func7 == 7'h0) ? `ALU_SRL : `ALU_SRA;
                3'h6: alu_op = `ALU_OR;
                3'h7: alu_op = `ALU_AND;
                default:;
            endcase
        end
    end

    // CSR  

    wire is_csr_imm = is_csr && (func3[2] == 1);

    always @(*) begin
        csr_op = `CSR_OTHER;
        case (func3[1:0])
            2'h1: csr_op = `CSR_RW;
            2'h2: csr_op = `CSR_RS;
            2'h3: csr_op = `CSR_RC;
            default:;
        endcase
    end

    // MUL   
`ifdef EXT_M_ENABLE
    wire is_mul = is_rtype && (func7 == 7'h1); 
    always @(*) begin
        mul_op = `MUL_MUL;
        case (func3)
            3'h0: mul_op = `MUL_MUL;
            3'h1: mul_op = `MUL_MULH;
            3'h2: mul_op = `MUL_MULHSU;
            3'h3: mul_op = `MUL_MULHU;
            3'h4: mul_op = `MUL_DIV;
            3'h5: mul_op = `MUL_DIVU;
            3'h6: mul_op = `MUL_REM;
            3'h7: mul_op = `MUL_REMU;
            default:; 
        endcase
    end
`else
    wire is_mul = 0;
    always @(*) begin
        mul_op = `MUL_MUL;
    end
`endif

    // FPU
`ifdef EXT_F_ENABLE
    wire is_fl     = (opcode == `INST_FL) && ((func3 == 2));
    wire is_fs     = (opcode == `INST_FS) && ((func3 == 2));
    wire is_fci    = (opcode == `INST_FCI);
    wire is_fmadd  = (opcode == `INST_FMADD);
    wire is_fmsub  = (opcode == `INST_FMSUB);
    wire is_fnmsub = (opcode == `INST_FNMSUB);
    wire is_fnmadd = (opcode == `INST_FNMADD); 

    wire is_fcmp   = is_fci && (func7 == 7'h50); // compare
    wire is_fcvti  = is_fci && (func7 == 7'h60); // convert to int
    wire is_fcvtf  = is_fci && (func7 == 7'h68); // convert to float
    wire is_fmvw_clss = is_fci && (func7 == 7'h70); // move to int + class
    wire is_fmvx   = is_fci && (func7 == 7'h78); // move to float
    wire is_fr4    = is_fmadd || is_fmsub || is_fnmsub || is_fnmadd;
    wire is_fpu    = (is_fl || is_fs || is_fci || is_fr4);   

    reg [2:0] frm;

    always @(*) begin    
        fpu_op = `FPU_MISC;    
        frm = func3;
        if (is_fr4) begin
            case ({is_fmadd, is_fmsub, is_fnmsub, is_fnmadd})
                4'b1000: fpu_op = `FPU_MADD;
                4'b0100: fpu_op = `FPU_MSUB;
                4'b0010: fpu_op = `FPU_NMSUB;
                4'b0001: fpu_op = `FPU_NMADD;
                default:;
            endcase
        end
        else begin
            case (func7)
                7'h00: fpu_op = `FPU_ADD;
                7'h04: fpu_op = `FPU_SUB;
                7'h08: fpu_op = `FPU_MUL;
                7'h0C: fpu_op = `FPU_DIV;
                7'h10: begin 
                    fpu_op = `FPU_MISC;
                       frm = (func3[1]) ? 2 : ((func3[0]) ? 1 : 0);
                end
                7'h14: begin
                    fpu_op = `FPU_MISC;
                       frm = (func3 == 3'h0) ? 3 : 4;
                end
                7'h2C: fpu_op = `FPU_SQRT;
                7'h50: fpu_op = `FPU_CMP;   // wb to intReg
                7'h60: fpu_op = (instr[20]) ? `FPU_CVTWUS : `FPU_CVTWS; // doesn't need rs2, and read rs1 from fpReg, WB to intReg
                7'h68: fpu_op = (instr[20]) ? `FPU_CVTSWU : `FPU_CVTSW; // doesn't need rs2, and read rs1 from intReg
                7'h70: begin 
                    fpu_op = (func3 == 3'h0) ? `FPU_MISC : `FPU_CLASS;
                       frm = (func3 == 3'h0) ? 5 : func3; 
                end 
                7'h78: begin fpu_op = `FPU_MISC; frm = 6; end             
                default:;
            endcase
        end
    end
`else
    wire is_fl        = 0;
    wire is_fs        = 0;
    wire is_fci       = 0;
    wire is_fcvti     = 0;
    wire is_fcvtf     = 0;
    wire is_fmvw_clss = 0;
    wire is_fmvx      = 0;
    wire is_fr4       = 0;
    wire is_fpu       = 0; 
    wire [2:0] frm    = 0;  

    always @(*) begin
        fpu_op = `FPU_MISC;
    end
`endif

    // LSU

    wire is_lsu = (is_ltype || is_stype || is_fl || is_fs);
    always @(*) begin
        lsu_op = {is_stype, func3};    
        if (is_fl) lsu_op = `LSU_LW;
        if (is_fs) lsu_op = `LSU_SW;
    end

    // GPU

    always @(*) begin
        gpu_op = `GPU_OTHER;
        case (func3)
            3'h0: gpu_op = `GPU_TMC;
            3'h1: gpu_op = `GPU_WSPAWN;
            3'h2: gpu_op = `GPU_SPLIT;
            3'h3: gpu_op = `GPU_JOIN;
            3'h4: gpu_op = `GPU_BAR;
            default:;
        endcase
    end

    ///////////////////////////////////////////////////////////////////////////

    wire use_rd = (is_fl || is_fci || is_fr4) 
               || ((rd != 0) && (is_itype || is_rtype || is_lui || is_auipc || is_csr || is_jal || is_jalr || is_jals || is_ltype));

    wire use_rs1 = is_fpu 
                || is_gpu
                || ((is_jalr || is_btype || is_ltype || is_stype || is_itype || is_rtype || !is_csr_imm || is_gpu) && (rs1 != 0));

    wire use_rs2 = (is_fpu && ~(is_fl || (fpu_op == `FPU_SQRT) || is_fcvti || is_fcvtf || is_fmvw_clss || is_fmvx))
                || (is_gpu && (gpu_op == `GPU_BAR || gpu_op == `GPU_WSPAWN))
                || ((is_btype || is_stype || is_rtype) && (rs2 != 0));

    wire use_rs3 = is_fr4;

    wire [4:0] rs1_qual = is_lui ? 5'h0 : rs1;

    ///////////////////////////////////////////////////////////////////////////

    assign decode_if.valid = ifetch_rsp_if.valid 
                          && (decode_if.ex_type != `EX_NOP); // skip noop

    assign decode_if.wid   = ifetch_rsp_if.wid;
    assign decode_if.tmask = ifetch_rsp_if.tmask;
    assign decode_if.PC    = ifetch_rsp_if.PC;

    assign decode_if.ex_type = is_lsu ? `EX_LSU :
                                    is_csr ? `EX_CSR :
                                        is_mul ? `EX_MUL :
                                            is_fpu ? `EX_FPU :
                                                is_gpu ? `EX_GPU :
                                                    is_br ? `EX_ALU :                                                        
                                                        (is_rtype || is_itype || is_lui || is_auipc) ? `EX_ALU :
                                                            `EX_NOP;

    assign decode_if.op_type = is_lsu ? `OP_BITS'(lsu_op) :
                                is_csr ? `OP_BITS'(csr_op) :
                                    is_mul ? `OP_BITS'(mul_op) :
                                        is_fpu ? `OP_BITS'(fpu_op) :
                                            is_gpu ? `OP_BITS'(gpu_op) :
                                                is_br ? `OP_BITS'(br_op) :
                                                    (is_rtype || is_itype || is_lui || is_auipc) ? `OP_BITS'(alu_op) :
                                                        0;

    assign decode_if.wb = use_rd;

    `ifdef EXT_F_ENABLE
        wire rd_is_fp  = is_fpu && ~(is_fcmp || is_fcvti || is_fmvw_clss);
        wire rs1_is_fp = is_fr4 || (is_fci && ~(is_fcvtf || is_fmvx));
        wire rs2_is_fp = is_fs || is_fr4 || is_fci;

        assign decode_if.rd  = {rd_is_fp,  rd};
        assign decode_if.rs1 = {rs1_is_fp, rs1_qual};
        assign decode_if.rs2 = {rs2_is_fp, rs2};
        assign decode_if.rs3 = {1'b1,      rs3};
    `else
        assign decode_if.rd  = rd;
        assign decode_if.rs1 = rs1_qual;
        assign decode_if.rs2 = rs2;
        assign decode_if.rs3 = rs3;
    `endif

    assign decode_if.used_regs = ((`NUM_REGS)'(use_rd)  << decode_if.rd) 
                               | ((`NUM_REGS)'(use_rs1) << decode_if.rs1) 
                               | ((`NUM_REGS)'(use_rs2) << decode_if.rs2)
                               | ((`NUM_REGS)'(use_rs3) << decode_if.rs3);

    assign decode_if.imm = (is_lui || is_auipc) ? {upper_imm, 12'(0)} : 
                                (is_jal || is_jalr || is_jals) ? jalx_offset :
                                    is_csr ? 32'(u_12) :
                                        src2_imm;

    assign decode_if.rs1_is_PC  = is_auipc || is_btype || is_jal || is_jals;
    assign decode_if.rs2_is_imm = is_itype || is_lui || is_auipc || is_csr_imm || is_br; 
    
    assign decode_if.op_mod = is_fpu ? frm : (is_br ? 1 : 0);

    ///////////////////////////////////////////////////////////////////////////

    wire decode_fire = decode_if.valid && decode_if.ready;

    assign join_if.valid = decode_fire && is_gpu && (gpu_op == `GPU_JOIN);
    assign join_if.wid = ifetch_rsp_if.wid;

    assign wstall_if.valid = decode_fire && (is_btype
                                          || is_jal 
                                          || is_jalr 
                                          || (is_gpu && (gpu_op == `GPU_TMC 
                                                      || gpu_op == `GPU_SPLIT 
                                                      || gpu_op == `GPU_BAR)));
    assign wstall_if.wid = ifetch_rsp_if.wid;

    ///////////////////////////////////////////////////////////////////////////

    assign ifetch_rsp_if.ready = decode_if.ready;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && decode_if.ready) begin
            $write("%t: core%0d-decode: wid=%0d, PC=%0h, ex=", $time, CORE_ID, decode_if.wid, decode_if.PC);
            print_ex_type(decode_if.ex_type);
            $write(", op=");
            print_ex_op(decode_if.ex_type, decode_if.op_type, decode_if.op_mod);
            $write(", mod=%0d, tmask=%b, wb=%b, rd=%0d, rs1=%0d, rs2=%0d, rs3=%0d, imm=%0h, use_pc=%b, use_imm=%b\n", decode_if.op_mod, decode_if.tmask, decode_if.wb, decode_if.rd, decode_if.rs1, decode_if.rs2, decode_if.rs3, decode_if.imm, decode_if.rs1_is_PC, decode_if.rs2_is_imm);                        
        end
    end
`endif

endmodule