`include "VX_define.vh"
`include "VX_config.vh"
`ifndef NDEBUG
`include "VX_trace_info.vh"
`endif

`ifdef EXT_F_ENABLE
    `define USED_IREG(r) \
        r``_r = {1'b0, ``r}

    `define USED_FREG(r) \
        r``_r = {1'b1, ``r}
`else
    `define USED_IREG(r) \
        r``_r = ``r
`endif

module VX_decode  #(
    parameter CORE_ID = 0
) (
    input  wire         clk,
    input  wire         reset,

    // inputs
    VX_ifetch_rsp_if.slave  ifetch_rsp_if,

    // outputs      
    VX_decode_if.master     decode_if,
    VX_wrelease_if.master   wrelease_if,
    VX_join_if.master       join_if
);
    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    
    reg [`EX_BITS-1:0]  ex_type;    
    reg [`INST_OP_BITS-1:0] op_type; 
    reg [`INST_MOD_BITS-1:0] op_mod;
    reg [`NR_BITS-1:0]  rd_r, rs1_r, rs2_r, rs3_r;
    reg [`XLEN-1:0]          imm;    
    reg use_rd, use_PC, use_imm;
    reg is_join, is_wstall;

    wire [31:0] instr = ifetch_rsp_if.data;
    wire [6:0] opcode = instr[6:0];  
    wire [1:0] func2  = instr[26:25];
    wire [2:0] func3  = instr[14:12];
    wire [6:0] func7  = instr[31:25];
    wire [11:0] u_12  = instr[31:20]; 

    wire [4:0] rd  = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];
    wire [4:0] rs3 = instr[31:27];

    wire [19:0] upper_imm = {func7, rs2, rs1, func3};
    wire [11:0] alu_imm   = (func3[0] && ~func3[1]) ? {7'b0, rs2} : u_12;
    wire [11:0] s_imm     = {func7, rd};
    wire [12:0] b_imm     = {instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [20:0] jal_imm   = {instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

    `UNUSED_VAR (rs3)

    always @(*) begin

        ex_type   = '0;
        op_type   = 'x;
        op_mod    = '0;
        rd_r      = '0;
        rs1_r     = '0;
        rs2_r     = '0;
        rs3_r     = '0;
        imm       = 'x;
        use_imm   = 0;
        use_PC    = 0;
        use_rd    = 0;
        is_join   = 0;
        is_wstall = 0;

        case (opcode)            
            `INST_I: begin
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = `INST_OP_BITS'(`INST_ALU_ADD);
                    3'h1: op_type = `INST_OP_BITS'(`INST_ALU_SLL);
                    3'h2: op_type = `INST_OP_BITS'(`INST_ALU_SLT);
                    3'h3: op_type = `INST_OP_BITS'(`INST_ALU_SLTU);
                    3'h4: op_type = `INST_OP_BITS'(`INST_ALU_XOR);
                    3'h5: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SRA) : `INST_OP_BITS'(`INST_ALU_SRL);
                    3'h6: op_type = `INST_OP_BITS'(`INST_ALU_OR);
                    3'h7: op_type = `INST_OP_BITS'(`INST_ALU_AND);
                endcase
                use_rd  = 1;
                use_imm = 1;
                imm     = {{(`XLEN-12){alu_imm[11]}}, alu_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_R: begin 
                ex_type = `EX_ALU;
            `ifdef EXT_M_ENABLE
                if (func7[0]) begin
                    case (func3)
                        3'h0: op_type = `INST_OP_BITS'(`INST_MUL_MUL);
                        3'h1: op_type = `INST_OP_BITS'(`INST_MUL_MULH);
                        3'h2: op_type = `INST_OP_BITS'(`INST_MUL_MULHSU);
                        3'h3: op_type = `INST_OP_BITS'(`INST_MUL_MULHU);
                        3'h4: op_type = `INST_OP_BITS'(`INST_MUL_DIV);
                        3'h5: op_type = `INST_OP_BITS'(`INST_MUL_DIVU);
                        3'h6: op_type = `INST_OP_BITS'(`INST_MUL_REM);
                        3'h7: op_type = `INST_OP_BITS'(`INST_MUL_REMU);
                        default:; 
                    endcase
                    op_mod = 2;
                end else 
            `endif
                begin
                    case (func3)
                        3'h0: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SUB) : `INST_OP_BITS'(`INST_ALU_ADD);
                        3'h1: op_type = `INST_OP_BITS'(`INST_ALU_SLL);
                        3'h2: op_type = `INST_OP_BITS'(`INST_ALU_SLT);
                        3'h3: op_type = `INST_OP_BITS'(`INST_ALU_SLTU);
                        3'h4: op_type = `INST_OP_BITS'(`INST_ALU_XOR);
                        3'h5: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SRA) : `INST_OP_BITS'(`INST_ALU_SRL);
                        3'h6: op_type = `INST_OP_BITS'(`INST_ALU_OR);
                        3'h7: op_type = `INST_OP_BITS'(`INST_ALU_AND);
                        default:;
                    endcase
                end          
                use_rd = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_I_W: begin
                // ADDIW, SLLIW, SRLIW, SRAIW
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = `INST_OP_BITS'(`INST_ALU_ADD_W);
                    3'h1: op_type = `INST_OP_BITS'(`INST_ALU_SLL_W);
                    3'h5: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SRA_W) : `INST_OP_BITS'(`INST_ALU_SRL_W);
                    default:;
                endcase
                use_rd  = 1;
                use_imm = 1;
                imm     = {{(`XLEN-12){alu_imm[11]}}, alu_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_R_W: begin
                // ADDW, SUBW, SLLW, SRLW, SRAW
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SUB_W) : `INST_OP_BITS'(`INST_ALU_ADD_W);
                    3'h1: op_type = `INST_OP_BITS'(`INST_ALU_SLL_W);
                    3'h5: op_type = (func7[5]) ? `INST_OP_BITS'(`INST_ALU_SRA_W) : `INST_OP_BITS'(`INST_ALU_SRL_W);
                    default:;
                endcase
                use_rd = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_LUI: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_ALU_LUI);
                use_rd  = 1;
                use_imm = 1;
                imm     = {{`XLEN-31{upper_imm[19]}}, upper_imm[18:0], 12'(0)};
                `USED_IREG (rd);
            end
            `INST_AUIPC: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_ALU_AUIPC);
                use_rd  = 1;
                use_imm = 1;
                use_PC  = 1;
                imm     = {{`XLEN-31{upper_imm[19]}}, upper_imm[18:0], 12'(0)};
                `USED_IREG (rd);
            end
            `INST_JAL: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_BR_JAL);
                op_mod  = 1;
                use_rd  = 1;
                use_imm = 1;
                use_PC  = 1;
                is_wstall = 1;
                imm     = {{(`XLEN-21){jal_imm[20]}}, jal_imm};
                `USED_IREG (rd);
            end
            `INST_JALR: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_BR_JALR);
                op_mod  = 1;
                use_rd  = 1;
                use_imm = 1;
                is_wstall = 1;
                imm     = {{(`XLEN-12){u_12[11]}}, u_12};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_B: begin 
                ex_type = `EX_ALU;
                case (func3)
                    3'h0: op_type = `INST_OP_BITS'(`INST_BR_EQ);
                    3'h1: op_type = `INST_OP_BITS'(`INST_BR_NE);
                    3'h4: op_type = `INST_OP_BITS'(`INST_BR_LT);
                    3'h5: op_type = `INST_OP_BITS'(`INST_BR_GE);
                    3'h6: op_type = `INST_OP_BITS'(`INST_BR_LTU);
                    3'h7: op_type = `INST_OP_BITS'(`INST_BR_GEU);
                    default:;
                endcase
                op_mod  = 1;
                use_imm = 1;
                use_PC  = 1;
                is_wstall = 1;
                imm     = {{(`XLEN-13){b_imm[12]}}, b_imm};
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_FENCE: begin
                ex_type = `EX_LSU;
                op_mod  = `INST_MOD_BITS'(1);
            end
            `INST_SYS : begin 
                if (func3[1:0] != 0) begin                    
                    ex_type = `EX_CSR;
                    op_type = `INST_OP_BITS'(func3[1:0]);
                    use_rd  = 1;
                    use_imm = func3[2]; 
                    imm[`CSR_ADDR_BITS-1:0] = u_12; // addr
                    `USED_IREG (rd);
                    if (func3[2]) begin
                        imm[`CSR_ADDR_BITS +: `NRI_BITS] = rs1; // imm
                    end else begin
                        `USED_IREG (rs1);
                    end                    
                end else begin
                    ex_type = `EX_ALU;
                    case (u_12)
                        12'h000: op_type = `INST_OP_BITS'(`INST_BR_ECALL);
                        12'h001: op_type = `INST_OP_BITS'(`INST_BR_EBREAK);             
                        12'h002: op_type = `INST_OP_BITS'(`INST_BR_URET);                        
                        12'h102: op_type = `INST_OP_BITS'(`INST_BR_SRET);                        
                        12'h302: op_type = `INST_OP_BITS'(`INST_BR_MRET);
                        default:;
                    endcase
                    op_mod  = 1;
                    use_rd  = 1;
                    use_imm = 1;
                    use_PC  = 1;
                    is_wstall = 1;
                    imm     = `XLEN'd4;
                    `USED_IREG (rd);
                end
            end
        `ifdef EXT_F_ENABLE
            `INST_FL, 
        `endif
            `INST_L: begin 
                ex_type = `EX_LSU;
                op_type = `INST_OP_BITS'({1'b0, func3});
                use_rd  = 1;
                imm     = {{(`XLEN-12){u_12[11]}}, u_12};
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
                op_type = `INST_OP_BITS'({1'b1, func3});
                imm     = {{(`XLEN-12){s_imm[11]}}, s_imm};
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
                op_type = `INST_OP_BITS'(opcode[3:0]);
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
                        op_type = `INST_OP_BITS'(func7[3:0]);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h2C: begin
                        op_type = `INST_OP_BITS'(`INST_FPU_SQRT);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end
                    7'h50: begin
                        op_type = `INST_OP_BITS'(`INST_FPU_CMP);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h60: begin
                        op_type = (instr[20]) ? `INST_OP_BITS'(`INST_FPU_CVTWUS) : `INST_OP_BITS'(`INST_FPU_CVTWS);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                    end
                    7'h68: begin
                        op_type = (instr[20]) ? `INST_OP_BITS'(`INST_FPU_CVTSWU) : `INST_OP_BITS'(`INST_FPU_CVTSW);
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                    7'h10: begin
                        // FSGNJ=0, FSGNJN=1, FSGNJX=2
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                        op_mod  = {1'b0, func3[1:0]};
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h14: begin
                        // FMIN=3, FMAX=4
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                        op_mod  = func3[0] ? 4 : 3;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    7'h70: begin 
                        if (func3[0]) begin
                            // FCLASS
                            op_type = `INST_OP_BITS'(`INST_FPU_CLASS);                                     
                        end else begin
                            // FMV.X.W=5
                            op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                            op_mod  = 5;
                        end
                        `USED_IREG (rd);
                        `USED_FREG (rs1);                                           
                    end 
                    7'h78: begin 
                        // FMV.W.X=6
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC); 
                        op_mod  = 6;
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                default:;
                endcase
            end
        `endif
            `INST_EXT1: begin 
                case (func7)
                    7'h00: begin
                        ex_type = `EX_GPU;
                        case (func3)
                            3'h0: begin // TMC, PRED
                                op_type = rs2[0] ? `INST_OP_BITS'(`INST_GPU_PRED) : `INST_OP_BITS'(`INST_GPU_TMC);
                                is_wstall = 1;
                                `USED_IREG (rs1);
                            end
                            3'h1: begin // WSPAWN
                                op_type = `INST_OP_BITS'(`INST_GPU_WSPAWN);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h2: begin // SPLIT
                                op_type = `INST_OP_BITS'(`INST_GPU_SPLIT);
                                is_wstall = 1;
                                `USED_IREG (rs1);
                            end
                            3'h3: begin // JOIN
                                op_type = `INST_OP_BITS'(`INST_GPU_JOIN);
                                is_join = 1;
                            end
                            3'h4: begin // BAR
                                op_type = `INST_OP_BITS'(`INST_GPU_BAR);
                                is_wstall = 1;
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            default:;
                        endcase
                    end
                    7'h01: begin
                        case (func3)
                        `ifdef EXT_RASTER_ENABLE
                            3'h0: begin // RASTER
                                ex_type   = `EX_GPU;
                                op_type   = `INST_OP_BITS'(`INST_GPU_RASTER);
                                use_rd    = 1;
                                `USED_IREG (rd);
                            end
                        `endif
                            default:;
                        endcase
                    end
                    default:;
                endcase
            end
            `INST_EXT2: begin                
                case (func3)
                `ifdef EXT_TEX_ENABLE
                    3'h0: begin // TEX
                        ex_type = `EX_GPU;
                        op_type = `INST_OP_BITS'(`INST_GPU_TEX);
                        op_mod  = `INST_MOD_BITS'(func2);
                        use_rd  = 1;
                        `USED_IREG (rd);       
                        `USED_IREG (rs1);
                        `USED_IREG (rs2);
                        `USED_IREG (rs3);
                    end
                `endif
                    3'h1: begin
                        case (func2)                       
                            2'h0: begin // CMOV
                                ex_type = `EX_GPU;
                                op_type = `INST_OP_BITS'(`INST_GPU_CMOV);
                                use_rd = 1;
                                `USED_IREG (rd);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                                `USED_IREG (rs3);
                            end
                        `ifdef EXT_ROP_ENABLE
                            2'h1: begin // ROP
                                ex_type = `EX_GPU;
                                op_type = `INST_OP_BITS'(`INST_GPU_ROP);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                                `USED_IREG (rs3);
                            end
                        `endif
                            default:;
                        endcase
                    end
                `ifdef EXT_IMADD_ENABLE
                    3'h2: begin // IMADD
                        ex_type = `EX_GPU;
                        op_type = `INST_OP_BITS'(`INST_GPU_IMADD);
                        op_mod  = `INST_MOD_BITS'(func2);
                        use_rd  = 1;
                        `USED_IREG (rd);
                        `USED_IREG (rs1);
                        `USED_IREG (rs2);
                        `USED_IREG (rs3);
                    end
                `endif
                    default:;
                endcase
            end
            default:;
        endcase
    end

    `UNUSED_VAR (func2)

    // disable write to integer register r0
    wire wb = use_rd && (| rd_r);

    assign decode_if.valid     = ifetch_rsp_if.valid;
    assign decode_if.uuid      = ifetch_rsp_if.uuid;
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

    ///////////////////////////////////////////////////////////////////////////

    wire ifetch_rsp_fire = ifetch_rsp_if.valid && ifetch_rsp_if.ready;

    assign join_if.valid = ifetch_rsp_fire && is_join;
    assign join_if.wid   = ifetch_rsp_if.wid;

    assign wrelease_if.valid = ifetch_rsp_fire && ~is_wstall;
    assign wrelease_if.wid   = ifetch_rsp_if.wid;

    assign ifetch_rsp_if.ibuf_pop = decode_if.ibuf_pop;
    assign ifetch_rsp_if.ready = decode_if.ready;

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && decode_if.ready) begin
            `TRACE(1, ("%d: core%0d-decode: wid=%0d, PC=0x%0h, ex=", $time, CORE_ID, decode_if.wid, decode_if.PC));
            `TRACE_EX_TYPE(1, decode_if.ex_type);
            `TRACE(1, (", op="));
            `TRACE_EX_OP(1, decode_if.ex_type, decode_if.op_type, decode_if.op_mod);
            `TRACE(1, (", mod=%0d, tmask=%b, wb=%b, rd=%0d, rs1=%0d, rs2=%0d, rs3=%0d, imm=0x%0h, use_pc=%b, use_imm=%b (#%0d)\n",
                decode_if.op_mod, decode_if.tmask, decode_if.wb, decode_if.rd, decode_if.rs1, decode_if.rs2, decode_if.rs3, decode_if.imm, decode_if.use_PC, decode_if.use_imm, decode_if.uuid));
        end
    end
`endif

endmodule
