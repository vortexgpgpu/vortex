`include "VX_define.vh"
`include "VX_print_instr.vh"

`ifdef EXT_F_ENABLE
    `define USED_IREG(r)    \
        used_regs[{1'b0, r}] = 1

    `define USED_FREG(r)    \
        r``_r[5] = 1;       \
        used_regs[{1'b1, r}] = 1
`else
    `define USED_IREG(r) \
        used_regs[r] = 1
`endif

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
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    
    reg [`EX_BITS-1:0]  ex_type;    
    reg [`OP_BITS-1:0]  op_type; 
    reg [`MOD_BITS-1:0] op_mod;
    reg [`NR_BITS-1:0]  rd_r, rs1_r, rs2_r, rs3_r;
    reg [31:0]          imm;    
    reg use_rd, use_PC, use_imm;
    reg is_join, is_wstall;
    reg [`NUM_REGS-1:0] used_regs;

    wire [31:0] instr = ifetch_rsp_if.instr;
    wire [6:0] opcode = instr[6:0];  
    wire [2:0] func3  = instr[14:12];
    wire [6:0] func7  = instr[31:25];
    wire [11:0] u_12  = instr[31:20]; 

    wire [4:0] rd  = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];
    wire [4:0] rs3 = instr[31:27];

    wire [19:0] upper_imm = {func7, rs2, rs1, func3};
    wire [11:0] alu_imm   = (func3[0] && ~func3[1]) ? {{7{1'b0}}, rs2} : u_12;
    wire [11:0] s_imm     = {func7, rd};
    wire [12:0] b_imm     = {instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [20:0] jal_imm   = {instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};
    wire [11:0] jalr_imm  = {func7, rs2};
    
    always @(*) begin

        ex_type   = 0;
        op_type   = 'x;
        op_mod    = 0;
        rd_r      = `NR_BITS'(rd);
        rs1_r     = `NR_BITS'(rs1);
        rs2_r     = `NR_BITS'(rs2);
        rs3_r     = `NR_BITS'(rs3);
        imm       = 'x;
        use_imm   = 0;
        use_PC    = 0;
        use_rd    = 0;
        is_join   = 0;
        is_wstall = 0;
        used_regs = 0;

        case (opcode)            
            `INST_I: begin
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = `OP_BITS'(`ALU_ADD);
                    3'h1: op_type = `OP_BITS'(`ALU_SLL);
                    3'h2: op_type = `OP_BITS'(`ALU_SLT);
                    3'h3: op_type = `OP_BITS'(`ALU_SLTU);
                    3'h4: op_type = `OP_BITS'(`ALU_XOR);
                    3'h5: op_type = (func7[5]) ? `OP_BITS'(`ALU_SRA) : `OP_BITS'(`ALU_SRL);
                    3'h6: op_type = `OP_BITS'(`ALU_OR);
                    3'h7: op_type = `OP_BITS'(`ALU_AND);
                    default:;
                endcase
                use_rd  = 1;
                use_imm = 1;
                imm     = {{20{alu_imm[11]}}, alu_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_R: begin 
                ex_type = `EX_ALU;
            `ifdef EXT_F_ENABLE
                if (func7[0]) begin
                    case (func3)
                        3'h0: op_type = `OP_BITS'(`MUL_MUL);
                        3'h1: op_type = `OP_BITS'(`MUL_MULH);
                        3'h2: op_type = `OP_BITS'(`MUL_MULHSU);
                        3'h3: op_type = `OP_BITS'(`MUL_MULHU);
                        3'h4: op_type = `OP_BITS'(`MUL_DIV);
                        3'h5: op_type = `OP_BITS'(`MUL_DIVU);
                        3'h6: op_type = `OP_BITS'(`MUL_REM);
                        3'h7: op_type = `OP_BITS'(`MUL_REMU);
                        default:; 
                    endcase
                    op_mod = 2;
                end else 
            `endif
                begin
                    case (func3)
                        3'h0: op_type = (func7[5]) ? `OP_BITS'(`ALU_SUB) : `OP_BITS'(`ALU_ADD);
                        3'h1: op_type = `OP_BITS'(`ALU_SLL);
                        3'h2: op_type = `OP_BITS'(`ALU_SLT);
                        3'h3: op_type = `OP_BITS'(`ALU_SLTU);
                        3'h4: op_type = `OP_BITS'(`ALU_XOR);
                        3'h5: op_type = (func7[5]) ? `OP_BITS'(`ALU_SRA) : `OP_BITS'(`ALU_SRL);
                        3'h6: op_type = `OP_BITS'(`ALU_OR);
                        3'h7: op_type = `OP_BITS'(`ALU_AND);
                        default:;
                    endcase
                end          
                use_rd = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_LUI: begin 
                ex_type = `EX_ALU;
                op_type = `OP_BITS'(`ALU_LUI);
                use_rd  = 1;
                use_imm = 1;
                imm     = {upper_imm, 12'(0)};
                `USED_IREG (rd);
                rs1_r   = 0;
            end
            `INST_AUIPC: begin 
                ex_type = `EX_ALU;
                op_type = `OP_BITS'(`ALU_AUIPC);
                use_rd  = 1;
                use_imm = 1;
                use_PC  = 1;
                imm     = {upper_imm, 12'(0)};
                `USED_IREG (rd);
            end
            `INST_JAL: begin 
                ex_type = `EX_ALU;
                op_type = `OP_BITS'(`BR_JAL);
                op_mod  = 1;
                use_rd  = 1;
                use_imm = 1;
                use_PC  = 1;
                is_wstall = 1;
                imm     = {{11{jal_imm[20]}}, jal_imm};
                `USED_IREG (rd);
            end
            `INST_JALR: begin 
                ex_type = `EX_ALU;
                op_type = `OP_BITS'(`BR_JALR);
                op_mod  = 1;
                use_rd  = 1;
                use_imm = 1;
                is_wstall = 1;
                imm     = {{20{jalr_imm[11]}}, jalr_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_B: begin 
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = `OP_BITS'(`BR_EQ);
                    3'h1: op_type = `OP_BITS'(`BR_NE);
                    3'h4: op_type = `OP_BITS'(`BR_LT);
                    3'h5: op_type = `OP_BITS'(`BR_GE);
                    3'h6: op_type = `OP_BITS'(`BR_LTU);
                    3'h7: op_type = `OP_BITS'(`BR_GEU);
                    default:;
                endcase
                op_mod  = 1;
                use_imm = 1;
                use_PC  = 1;
                is_wstall = 1;
                imm     = {{19{b_imm[12]}}, b_imm};
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_F: begin
                ex_type = `EX_LSU;
                op_mod  = `MOD_BITS'(!func3[0]); // data fence
            end
            `INST_SYS : begin 
                if (func3[1:0] != 0) begin                    
                    ex_type = `EX_CSR;
                    op_type = `OP_BITS'(func3[1:0]);
                    use_rd  = 1;
                    use_imm = func3[2]; 
                    imm     = 32'(u_12); // addr
                    `USED_IREG (rd);
                    if (func3[2]) begin
                        rs1_r = `NR_BITS'(rs1); // imm
                    end else begin
                        `USED_IREG (rs1);
                    end                    
                end else begin
                    ex_type = `EX_ALU;
                    case (u_12)
                        12'h000: op_type = `OP_BITS'(`BR_ECALL);
                        12'h001: op_type = `OP_BITS'(`BR_EBREAK);             
                        12'h302: op_type = `OP_BITS'(`BR_MRET);
                        12'h102: op_type = `OP_BITS'(`BR_SRET);
                        12'h7B2: op_type = `OP_BITS'(`BR_DRET);
                        default:;
                    endcase
                    op_mod  = 1;
                    use_rd  = 1;
                    use_imm = 1;
                    use_PC  = 1;
                    imm     = 32'd4;
                    `USED_IREG (rd);
                end
            end
        `ifdef EXT_F_ENABLE
            `INST_FL, 
        `endif
            `INST_L: begin 
                ex_type = `EX_LSU;
                op_type = `OP_BITS'({1'b0, func3});
                use_rd  = 1;
                imm     = {{20{u_12[11]}}, u_12};
            `ifdef EXT_F_ENABLE
                if (opcode[2]) begin
                    `USED_FREG (rd);
                end else
            `endif
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
        `ifdef EXT_F_ENABLE
            `INST_FS, 
        `endif
            `INST_S: begin 
                ex_type = `EX_LSU;
                op_type = `OP_BITS'({1'b1, func3});
                imm     = {{20{s_imm[11]}}, s_imm};
                `USED_IREG (rs1);
            `ifdef EXT_F_ENABLE
                if (opcode[2]) begin
                    `USED_FREG (rs2);
                end else
            `endif
                `USED_IREG (rs2);
            end
        `ifdef EXT_F_ENABLE
            `INST_FMADD,
            `INST_FMSUB,
            `INST_FNMSUB,
            `INST_FNMADD: begin 
                ex_type = `EX_FPU;
                op_type = `OP_BITS'(opcode[3:0]);
                op_mod  = func3;
                use_rd  = 1;
                `USED_FREG (rd);              
                `USED_FREG (rs1);
                `USED_FREG (rs2);
                `USED_FREG (rs3);
            end
            `INST_FCI: begin 
                ex_type = `EX_FPU;
                op_mod  = func3;
                use_rd  = 1;                
                case (func7)
                    7'h00, // FADD
                    7'h04, // FSUB
                    7'h08, // FMUL
                    7'h0C: begin // FDIV
                        op_type = `OP_BITS'(func7[3:0]);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h2C: begin
                        op_type = `OP_BITS'(`FPU_SQRT);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end
                    7'h50: begin
                        op_type = `OP_BITS'(`FPU_CMP);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h60: begin
                        op_type = (instr[20]) ? `OP_BITS'(`FPU_CVTWUS) : `OP_BITS'(`FPU_CVTWS);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                    end
                    7'h68: begin
                        op_type = (instr[20]) ? `OP_BITS'(`FPU_CVTSWU) : `OP_BITS'(`FPU_CVTSW);
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                    7'h10: begin
                        // FSGNJ=0, FSGNJN=1, FSGNJX=2
                        op_type = `OP_BITS'(`FPU_MISC);
                        op_mod  = {1'b0, func3[1:0]};
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h14: begin
                        // FMIN=3, FMAX=4
                        op_type = `OP_BITS'(`FPU_MISC);
                        op_mod  = func3[0] ? 4 : 3;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h70: begin 
                        if (func3[0]) begin
                            // FCLASS
                            op_type = `OP_BITS'(`FPU_CLASS);                                     
                        end else begin
                            // FMV.X.W=5
                            op_type = `OP_BITS'(`FPU_MISC);
                            op_mod  = 5;
                        end
                        `USED_IREG (rd);
                        `USED_FREG (rs1);                                           
                    end 
                    7'h78: begin 
                        // FMV.W.X=6
                        op_type = `OP_BITS'(`FPU_MISC); 
                        op_mod  = 6;
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                default:;
                endcase
            end
        `endif
            `INST_GPU: begin 
                ex_type = `EX_GPU;
                case (func3)
                    3'h0: begin
                        op_type = `OP_BITS'(`GPU_TMC);
                        is_wstall = 1;
                        `USED_IREG (rs1);
                    end
                    3'h1: begin
                        op_type = `OP_BITS'(`GPU_WSPAWN);
                        `USED_IREG (rs1);
                        `USED_IREG (rs2);
                    end
                    3'h2: begin
                        op_type = `OP_BITS'(`GPU_SPLIT);
                        is_wstall = 1;
                        `USED_IREG (rs1);
                    end
                    3'h3: begin 
                        op_type = `OP_BITS'(`GPU_JOIN);
                        is_join = 1;
                    end
                    3'h4: begin 
                        op_type = `OP_BITS'(`GPU_BAR);
                        is_wstall = 1;
                        `USED_IREG (rs1);
                        `USED_IREG (rs2);
                    end
                    default:;
                endcase
            end
            default:;
        endcase
    end

    // disable write to integer register r0
    wire wb = use_rd && (| rd_r);

    assign decode_if.valid     = ifetch_rsp_if.valid;
    assign decode_if.wid       = ifetch_rsp_if.wid;
    assign decode_if.tmask     = ifetch_rsp_if.tmask;
    assign decode_if.PC        = ifetch_rsp_if.PC;
    assign decode_if.ex_type   = ex_type;
    assign decode_if.op_type   = op_type;
    assign decode_if.op_mod    = op_mod;
    assign decode_if.wb        = wb;
    assign decode_if.rd        = rd_r;
    assign decode_if.rs1       = rs1_r;
    assign decode_if.rs2       = rs2_r;
    assign decode_if.rs3       = rs3_r;
    assign decode_if.imm       = imm;
    assign decode_if.use_PC    = use_PC;
    assign decode_if.use_imm   = use_imm;
    assign decode_if.used_regs = used_regs;

    ///////////////////////////////////////////////////////////////////////////

    wire ifetch_rsp_fire = ifetch_rsp_if.valid && ifetch_rsp_if.ready;

    assign join_if.valid = ifetch_rsp_fire && is_join;
    assign join_if.wid = ifetch_rsp_if.wid;

    assign wstall_if.valid = ifetch_rsp_fire && is_wstall;
    assign wstall_if.wid = ifetch_rsp_if.wid;

    assign ifetch_rsp_if.ready = decode_if.ready;

`ifdef DBG_PRINT_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && decode_if.ready) begin
            $write("%t: core%0d-decode: wid=%0d, PC=%0h, ex=", $time, CORE_ID, decode_if.wid, decode_if.PC);
            print_ex_type(decode_if.ex_type);
            $write(", op=");
            print_ex_op(decode_if.ex_type, decode_if.op_type, decode_if.op_mod);
            $write(", mod=%0d, tmask=%b, wb=%b, rd=%0d, rs1=%0d, rs2=%0d, rs3=%0d, imm=%0h, use_pc=%b, use_imm=%b, use_regs=%b\n", decode_if.op_mod, decode_if.tmask, decode_if.wb, decode_if.rd, decode_if.rs1, decode_if.rs2, decode_if.rs3, decode_if.imm, decode_if.use_PC, decode_if.use_imm, decode_if.used_regs);                        
        end
    end
`endif

endmodule