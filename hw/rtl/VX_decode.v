
`include "VX_define.vh"

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
    wire        in_valid = (| ifetch_rsp_if.valid);
    wire [31:0] instr    = ifetch_rsp_if.instr;

    reg [`ALU_BITS-1:0] alu_op;
    reg [`BR_BITS-1:0]  br_op;
    reg [`MUL_BITS-1:0] mul_op;
    wire [`LSU_BITS-1:0] lsu_op;
    reg [`CSR_BITS-1:0] csr_op;
    reg [`GPU_BITS-1:0] gpu_op;

    reg [19:0] upper_imm;
    reg [31:0] jalx_offset;
    reg [31:0] src2_imm;

    wire [6:0] opcode = instr[6:0];  
    wire [2:0] func3  = instr[14:12];
    wire [6:0] func7  = instr[31:25];
    wire [11:0] u_12  = instr[31:20]; 

    wire [`NR_BITS-1:0] rd  = instr[11:7];
    wire [`NR_BITS-1:0] rs1 = instr[19:15];
    wire [`NR_BITS-1:0] rs2 = instr[24:20];     

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
    wire is_br    = (is_btype || is_jal || is_jalr || is_jals);
    wire is_mul   = is_rtype && (func7 == 7'h1);
    
    // upper immediate
    always @(*) begin
        case (opcode)
            `INST_LUI:   upper_imm = {func7, rs2, rs1, func3};
            `INST_AUIPC: upper_imm = {func7, rs2, rs1, func3};
            default:     upper_imm = 20'h0;
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

    // I-type immediate
    wire alu_shift_i          = (func3 == 3'h1) || (func3 == 3'h5);
    wire [11:0] alu_shift_imm = {{7{1'b0}}, rs2};
    wire [11:0] alu_imm       = alu_shift_i ? alu_shift_imm : u_12;
    always @(*) begin
        case (opcode)
            `INST_I: src2_imm = {{20{alu_imm[11]}}, alu_imm};
            `INST_S: src2_imm = {{20{func7[6]}}, func7, rd};
            `INST_L: src2_imm = {{20{u_12[11]}}, u_12};
            `INST_B: src2_imm = {{20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0};
            default: src2_imm = 32'hdeadbeef;
        endcase
    end   

    // BRANCH
    always @(*) begin
        br_op = `BR_EQ;
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

    // MUL    
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

    // LSU
    wire is_lsu   = (is_ltype || is_stype);
    assign lsu_op = {is_stype, func3};    

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

    VX_decode_if decode_tmp_if();

    assign decode_tmp_if.valid    = ifetch_rsp_if.valid;
    assign decode_tmp_if.warp_num = ifetch_rsp_if.warp_num;
    assign decode_tmp_if.curr_PC  = ifetch_rsp_if.curr_PC;
    assign decode_tmp_if.next_PC  = ifetch_rsp_if.curr_PC + 32'h4;

    assign decode_tmp_if.ex_type  = is_lsu ? `EX_LSU :
                                        is_csr ? `EX_CSR :
                                            is_mul ? `EX_MUL :
                                                is_gpu ? `EX_GPU :
                                                    is_br ? `EX_ALU :
                                                        (is_rtype || is_itype || is_lui || is_auipc) ? `EX_ALU :
                                                            `EX_NOP;

    assign decode_tmp_if.instr_op = is_lsu ? `OP_BITS'(lsu_op) :
                                        is_csr ? `OP_BITS'(csr_op) :
                                            is_mul ? `OP_BITS'(mul_op) :
                                                is_gpu ? `OP_BITS'(gpu_op) :
                                                    is_br ? `OP_BITS'({1'b1, br_op}) :
                                                        (is_rtype || is_itype || is_lui || is_auipc) ? `OP_BITS'(alu_op) :
                                                            0; 

    assign decode_tmp_if.rd  = rd;

    assign decode_tmp_if.rs1 = is_lui ? `NR_BITS'(0) : rs1;

    assign decode_tmp_if.rs2 = rs2;

    assign decode_tmp_if.imm = (is_lui || is_auipc) ? {upper_imm, 12'(0)} : 
                                    (is_jal || is_jalr || is_jals) ? jalx_offset :
                                        is_csr ? 32'(u_12) :
                                            src2_imm;

    assign decode_tmp_if.rs1_is_PC  = is_auipc;

    assign decode_tmp_if.rs2_is_imm = is_itype || is_lui || is_auipc || is_csr_imm;   

    assign decode_tmp_if.use_rs1 = (decode_tmp_if.rs1 != 0)
                                && (is_jalr || is_btype || is_ltype || is_stype || is_itype || is_rtype || ~is_csr_imm || is_gpu);

    assign decode_tmp_if.use_rs2 = (decode_tmp_if.rs2 != 0) 
                                && (is_btype || is_stype || is_rtype || (is_gpu && (gpu_op == `GPU_BAR || gpu_op == `GPU_WSPAWN)));

    assign decode_tmp_if.wb = (rd == 0) ? `WB_NO : // disable writeback to r0
                                (is_itype || is_rtype || is_lui || is_auipc || is_csr) ? `WB_ALU :
                                    (is_jal || is_jalr || is_jals) ? `WB_JAL :
                                        is_ltype ? `WB_MEM :                                        
                                            `WB_NO;    

    assign join_if.is_join  = is_gpu && (gpu_op == `GPU_JOIN) && in_valid;
    assign join_if.warp_num = ifetch_rsp_if.warp_num;

    assign wstall_if.wstall = (is_br || is_gpu) && in_valid;
    assign wstall_if.warp_num = ifetch_rsp_if.warp_num;

    wire stall = ~decode_if.ready && (| decode_if.valid);   
    
    VX_generic_register #(
        .N(`NUM_THREADS + `NW_BITS + 32 + 32 + `NR_BITS + `NR_BITS + `NR_BITS + 32 + 1 + 1 + 1 + 1 + `EX_BITS + `OP_BITS + `WB_BITS)
    ) decode_reg (
        .clk   (clk),
        .reset (reset),
        .stall (stall),
        .flush (0),
        .in    ({decode_tmp_if.valid, decode_tmp_if.warp_num, decode_tmp_if.curr_PC, decode_tmp_if.next_PC, decode_tmp_if.rd, decode_tmp_if.rs1, decode_tmp_if.rs2, decode_tmp_if.imm, decode_tmp_if.rs1_is_PC, decode_tmp_if.rs2_is_imm, decode_tmp_if.use_rs1, decode_tmp_if.use_rs2, decode_tmp_if.ex_type, decode_tmp_if.instr_op, decode_tmp_if.wb}),
        .out   ({decode_if.valid,     decode_if.warp_num,     decode_if.curr_PC,     decode_if.next_PC,     decode_if.rd,     decode_if.rs1,     decode_if.rs2,     decode_if.imm,     decode_if.rs1_is_PC,     decode_if.rs2_is_imm,     decode_if.use_rs1,     decode_if.use_rs2,     decode_if.ex_type,     decode_if.instr_op,     decode_if.wb})
    ); 

    assign ifetch_rsp_if.ready = ~stall;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if ((| decode_tmp_if.valid) && ~stall) begin
            $write("%t: Core%0d-Decode: warp=%0d, PC=%0h, ex=", $time, CORE_ID, decode_tmp_if.warp_num, decode_tmp_if.curr_PC);
            print_ex_type(decode_tmp_if.ex_type);
            $write(", op=");
            print_instr_op(decode_tmp_if.ex_type, decode_tmp_if.instr_op);
            $write(", wb=");
            print_wb(decode_tmp_if.wb);
            $write(", rd=%0d, rs1=%0d, rs2=%0d, imm=%0h, use_pc=%b, use_imm=%b, use_rs1=%b, use_rs2=%b\n", decode_tmp_if.rd, decode_tmp_if.rs1, decode_tmp_if.rs2, decode_tmp_if.imm, decode_tmp_if.rs1_is_PC, decode_tmp_if.rs2_is_imm, decode_tmp_if.use_rs1, decode_tmp_if.use_rs2);                        

            // trap unsupported instructions
            assert(~(~stall && (decode_tmp_if.ex_type == `EX_ALU) && `ALU_OP(decode_tmp_if.instr_op) == `ALU_OTHER));
            assert(~(~stall && (decode_tmp_if.ex_type == `EX_CSR) && `CSR_OP(decode_tmp_if.instr_op) == `CSR_OTHER));
            assert(~(~stall && (decode_tmp_if.ex_type == `EX_GPU) && `GPU_OP(decode_tmp_if.instr_op) == `GPU_OTHER));
        end
    end
`endif

endmodule