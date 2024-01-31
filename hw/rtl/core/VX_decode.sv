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

`include "VX_define.vh"
`include "VX_trace.vh"

`ifdef EXT_F_ENABLE
    `define USED_IREG(x) \
        x``_r = {1'b0, ``x}; \
        use_``x = 1

    `define USED_FREG(x) \
        x``_r = {1'b1, ``x}; \
        use_``x = 1
`else
    `define USED_IREG(x) \
        x``_r = ``x; \
        use_``x = 1
`endif

module VX_decode  #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    // inputs
    VX_fetch_if.slave       fetch_if,

    // outputs      
    VX_decode_if.master     decode_if,
    VX_decode_sched_if.master decode_sched_if
);

    localparam DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + (`NR_BITS * 4) + `XLEN + 1 + 1;

    `UNUSED_PARAM (CORE_ID)
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)
    
    reg [`EX_BITS-1:0] ex_type;    
    reg [`INST_OP_BITS-1:0] op_type; 
    reg [`INST_MOD_BITS-1:0] op_mod;
    reg [`NR_BITS-1:0] rd_r, rs1_r, rs2_r, rs3_r;
    reg [`XLEN-1:0] imm;    
    reg use_rd, use_rs1, use_rs2, use_rs3, use_PC, use_imm;
    reg is_wstall;

    wire [31:0] instr = fetch_if.data.instr;
    wire [6:0] opcode = instr[6:0];  
    wire [1:0] func2  = instr[26:25];
    wire [2:0] func3  = instr[14:12];
    wire [4:0] func5  = instr[31:27];
    wire [6:0] func7  = instr[31:25];
    wire [11:0] u_12  = instr[31:20];

    wire [4:0] rd  = instr[11:7];
    wire [4:0] rs1 = instr[19:15];
    wire [4:0] rs2 = instr[24:20];
    wire [4:0] rs3 = instr[31:27];

    `UNUSED_VAR (func2)
    `UNUSED_VAR (func5)
    `UNUSED_VAR (rs3)
    `UNUSED_VAR (use_rd)
    `UNUSED_VAR (use_rs1)
    `UNUSED_VAR (use_rs2)
    `UNUSED_VAR (use_rs3)

    wire is_itype_sh = func3[0] && ~func3[1];

    wire [19:0] ui_imm  = instr[31:12];
`ifdef XLEN_64
    wire [11:0] i_imm   = is_itype_sh ? {6'b0, instr[25:20]} : u_12;
    wire [11:0] iw_imm  = is_itype_sh ? {7'b0, instr[24:20]} : u_12;
`else
    wire [11:0] i_imm   = is_itype_sh ? {7'b0, instr[24:20]} : u_12;
`endif    
    wire [11:0] s_imm   = {func7, rd};
    wire [12:0] b_imm   = {instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
    wire [20:0] jal_imm = {instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

    reg [`INST_ALU_BITS-1:0] r_type;
    always @(*) begin
        case (func3)
            3'h0: r_type = (opcode[5] && func7[5]) ? `INST_ALU_SUB : `INST_ALU_ADD;
            3'h1: r_type = `INST_ALU_SLL;
            3'h2: r_type = `INST_ALU_SLT;
            3'h3: r_type = `INST_ALU_SLTU;
            3'h4: r_type = `INST_ALU_XOR;
            3'h5: r_type = func7[5] ? `INST_ALU_SRA : `INST_ALU_SRL;
            3'h6: r_type = `INST_ALU_OR;
            3'h7: r_type = `INST_ALU_AND;
        endcase
    end

    reg [`INST_BR_BITS-1:0] b_type;
    always @(*) begin
        case (func3)
            3'h0: b_type = `INST_BR_EQ;
            3'h1: b_type = `INST_BR_NE;
            3'h4: b_type = `INST_BR_LT;
            3'h5: b_type = `INST_BR_GE;
            3'h6: b_type = `INST_BR_LTU;
            3'h7: b_type = `INST_BR_GEU;
            default: b_type = 'x;
        endcase
    end

    reg [`INST_BR_BITS-1:0] s_type;
    always @(*) begin
        case (u_12)
            12'h000: s_type = `INST_OP_BITS'(`INST_BR_ECALL);
            12'h001: s_type = `INST_OP_BITS'(`INST_BR_EBREAK);             
            12'h002: s_type = `INST_OP_BITS'(`INST_BR_URET);                        
            12'h102: s_type = `INST_OP_BITS'(`INST_BR_SRET);                        
            12'h302: s_type = `INST_OP_BITS'(`INST_BR_MRET);
            default: s_type = 'x;
        endcase
    end

`ifdef EXT_M_ENABLE
    reg [`INST_M_BITS-1:0] m_type;
    always @(*) begin
        case (func3)
            3'h0: m_type = `INST_M_MUL;
            3'h1: m_type = `INST_M_MULH;
            3'h2: m_type = `INST_M_MULHSU;
            3'h3: m_type = `INST_M_MULHU;
            3'h4: m_type = `INST_M_DIV;
            3'h5: m_type = `INST_M_DIVU;
            3'h6: m_type = `INST_M_REM;
            3'h7: m_type = `INST_M_REMU;
        endcase
    end
`endif

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
        use_rs1   = 0;
        use_rs2   = 0;
        use_rs3   = 0;
        is_wstall = 0;

        case (opcode)            
            `INST_I: begin
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(r_type);
                use_rd  = 1;
                use_imm = 1;
                imm     = {{(`XLEN-12){i_imm[11]}}, i_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_R: begin 
                ex_type = `EX_ALU;
            `ifdef EXT_M_ENABLE
                if (func7[0]) begin
                    op_type = `INST_OP_BITS'(m_type);
                    op_mod[1] = 1;
                end else 
            `endif
                begin
                    op_type = `INST_OP_BITS'(r_type);
                end          
                use_rd = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
        `ifdef XLEN_64
            `INST_I_W: begin
                // ADDIW, SLLIW, SRLIW, SRAIW
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(r_type);
                op_mod[2] = 1;
                use_rd  = 1;
                use_imm = 1;
                imm     = {{(`XLEN-12){iw_imm[11]}}, iw_imm};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_R_W: begin
                ex_type = `EX_ALU;
            `ifdef EXT_M_ENABLE                
                if (func7[0]) begin
                    // MULW, DIVW, DIVUW, REMW, REMUW
                    op_type = `INST_OP_BITS'(m_type);
                    op_mod[1] = 1;                    
                end else 
            `endif
                begin
                    // ADDW, SUBW, SLLW, SRLW, SRAW
                    op_type = `INST_OP_BITS'(r_type);
                end
                op_mod[2] = 1;
                use_rd = 1;
                `USED_IREG (rd);
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
        `endif
            `INST_LUI: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_ALU_LUI);
                use_rd  = 1;
                use_imm = 1;
                imm     = {{`XLEN-31{ui_imm[19]}}, ui_imm[18:0], 12'(0)};
                `USED_IREG (rd);
            end
            `INST_AUIPC: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_ALU_AUIPC);
                use_rd  = 1;
                use_imm = 1;
                use_PC  = 1;
                imm     = {{`XLEN-31{ui_imm[19]}}, ui_imm[18:0], 12'(0)};
                `USED_IREG (rd);
            end
            `INST_JAL: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(`INST_BR_JAL);
                op_mod[0] = 1;
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
                op_mod[0] = 1;
                use_rd  = 1;
                use_imm = 1;
                is_wstall = 1;
                imm     = {{(`XLEN-12){u_12[11]}}, u_12};
                `USED_IREG (rd);
                `USED_IREG (rs1);
            end
            `INST_B: begin 
                ex_type = `EX_ALU;
                op_type = `INST_OP_BITS'(b_type);
                op_mod[0] = 1;
                use_imm = 1;
                use_PC  = 1;
                is_wstall = 1;
                imm     = {{(`XLEN-13){b_imm[12]}}, b_imm};
                `USED_IREG (rs1);
                `USED_IREG (rs2);
            end
            `INST_FENCE: begin
                ex_type = `EX_LSU;
                op_type = `INST_LSU_FENCE;
            end
            `INST_SYS : begin 
                if (func3[1:0] != 0) begin                    
                    ex_type = `EX_SFU;
                    op_type = `INST_OP_BITS'(`INST_SFU_CSR(func3[1:0]));
                    use_rd  = 1;
                    is_wstall = 1;
                    use_imm = func3[2];
                    imm[`VX_CSR_ADDR_BITS-1:0] = u_12; // addr
                    `USED_IREG (rd);
                    if (func3[2]) begin
                        imm[`VX_CSR_ADDR_BITS +: `NRI_BITS] = rs1; // imm
                    end else begin
                        `USED_IREG (rs1);
                    end                    
                end else begin
                    ex_type = `EX_ALU;
                    op_type = `INST_OP_BITS'(s_type);
                    op_mod[0] = 1;
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
                use_imm = 1;
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
                use_imm = 1;
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
                op_type = `INST_OP_BITS'({2'b11, opcode[3:2]});
                op_mod  = `INST_MOD_BITS'(func3);
                imm[0]  = func2[0]; // destination is double?
                use_rd  = 1;
                `USED_FREG (rd);              
                `USED_FREG (rs1);
                `USED_FREG (rs2);
                `USED_FREG (rs3);
            end
            `INST_FCI: begin 
                ex_type = `EX_FPU;
                op_mod  = `INST_MOD_BITS'(func3);
            `ifdef FLEN_64
                imm[0]  = func2[0]; // destination is double?
            `endif
                use_rd  = 1;                
                case (func5)
                    5'b00000, // FADD
                    5'b00001, // FSUB
                    5'b00010, // FMUL
                    5'b00011: begin // FDIV
                        op_type = `INST_OP_BITS'(func5[1:0]);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b00100: begin
                        // NCP: FSGNJ=0, FSGNJN=1, FSGNJX=2
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                        op_mod  = `INST_MOD_BITS'(func3[1:0]);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end
                    5'b00101: begin
                        // NCP: FMIN=6, FMAX=7
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                        op_mod  = func3[0] ? 7 : 6;
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end 
                `ifdef FLEN_64
                    5'b01000: begin   
                        // CVT.S.D, CVT.D.S
                        op_type = `INST_OP_BITS'(`INST_FPU_F2F);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end
                `endif
                    5'b01011: begin                        
                        // SQRT
                        op_type = `INST_OP_BITS'(`INST_FPU_SQRT);
                        `USED_FREG (rd);
                        `USED_FREG (rs1);
                    end   
                    5'b10100: begin
                        // CMP
                        op_type = `INST_OP_BITS'(`INST_FPU_CMP);
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                        `USED_FREG (rs2);
                    end             
                    5'b11000: begin
                        // CVT.W.X, CVT.WU.X
                        op_type = (rs2[0]) ? `INST_OP_BITS'(`INST_FPU_F2U) : `INST_OP_BITS'(`INST_FPU_F2I);
                    `ifdef XLEN_64
                        imm[1] = rs2[1]; // is 64-bit integer
                    `endif
                        `USED_IREG (rd);
                        `USED_FREG (rs1);
                    end
                    5'b11010: begin
                        // CVT.X.W, CVT.X.WU
                        op_type = (rs2[0]) ? `INST_OP_BITS'(`INST_FPU_U2F) : `INST_OP_BITS'(`INST_FPU_I2F);
                    `ifdef XLEN_64
                        imm[1] = rs2[1]; // is 64-bit integer
                    `endif
                        `USED_FREG (rd);
                        `USED_IREG (rs1);
                    end
                    5'b11100: begin 
                        if (func3[0]) begin
                            // NCP: FCLASS=3
                            op_type = `INST_OP_BITS'(`INST_FPU_MISC);                                     
                            op_mod  = 3;
                        end else begin
                            // NCP: FMV.X.W=4
                            op_type = `INST_OP_BITS'(`INST_FPU_MISC);
                            op_mod  = 4;
                        end
                        `USED_IREG (rd);
                        `USED_FREG (rs1);                                           
                    end 
                    5'b11110: begin 
                        // NCP: FMV.W.X=5
                        op_type = `INST_OP_BITS'(`INST_FPU_MISC); 
                        op_mod  = 5;
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
                        ex_type = `EX_SFU;
                        is_wstall = 1;
                        case (func3)
                            3'h0: begin // TMC
                                op_type = `INST_OP_BITS'(`INST_SFU_TMC);
                                `USED_IREG (rs1);
                            end
                            3'h1: begin // WSPAWN
                                op_type = `INST_OP_BITS'(`INST_SFU_WSPAWN);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h2: begin // SPLIT
                                op_type = `INST_OP_BITS'(`INST_SFU_SPLIT);
                                use_rd    = 1;
                                `USED_IREG (rs1);                                
                                `USED_IREG (rd);                                
                            end
                            3'h3: begin // JOIN
                                op_type = `INST_OP_BITS'(`INST_SFU_JOIN);
                                `USED_IREG (rs1);
                            end
                            3'h4: begin // BAR
                                op_type = `INST_OP_BITS'(`INST_SFU_BAR);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            3'h5: begin // PRED
                                op_type = `INST_OP_BITS'(`INST_SFU_PRED);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                            end
                            default:;
                        endcase
                    end
                    default:;
                endcase
            end
            `INST_EXT2: begin                
                case (func3)
                    3'h1: begin
                        case (func2)                       
                            2'h0: begin // CMOV
                                ex_type = `EX_SFU;
                                op_type = `INST_OP_BITS'(`INST_SFU_CMOV);
                                use_rd = 1;
                                `USED_IREG (rd);
                                `USED_IREG (rs1);
                                `USED_IREG (rs2);
                                `USED_IREG (rs3);
                            end
                            default:;
                        endcase
                    end
                    default:;
                endcase
            end
            default:;
        endcase
    end

    // disable write to integer register r0
    wire wb = use_rd && (rd_r != 0);

    VX_elastic_buffer #(
        .DATAW (DATAW),
        .SIZE  (0)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (fetch_if.valid),
        .ready_in  (fetch_if.ready),
        .data_in   ({fetch_if.data.uuid, fetch_if.data.wid, fetch_if.data.tmask, fetch_if.data.PC, ex_type, op_type, op_mod, use_PC, imm, use_imm, wb, rd_r, rs1_r, rs2_r, rs3_r}),
        .data_out  ({decode_if.data.uuid, decode_if.data.wid, decode_if.data.tmask, decode_if.data.PC, decode_if.data.ex_type, decode_if.data.op_type, decode_if.data.op_mod, decode_if.data.use_PC, decode_if.data.imm, decode_if.data.use_imm, decode_if.data.wb, decode_if.data.rd, decode_if.data.rs1, decode_if.data.rs2, decode_if.data.rs3}),
        .valid_out (decode_if.valid),
        .ready_out (decode_if.ready)
    );

    ///////////////////////////////////////////////////////////////////////////

    wire fetch_fire = fetch_if.valid && fetch_if.ready;

    assign decode_sched_if.valid    = fetch_fire;
    assign decode_sched_if.wid      = fetch_if.data.wid;
    assign decode_sched_if.is_wstall = is_wstall;
`ifndef L1_ENABLE    
    assign fetch_if.ibuf_pop = decode_if.ibuf_pop;
`endif

`ifdef DBG_TRACE_CORE_PIPELINE
    always @(posedge clk) begin
        if (decode_if.valid && decode_if.ready) begin
            `TRACE(1, ("%d: core%0d-decode: wid=%0d, PC=0x%0h, instr=0x%0h, ex=", $time, CORE_ID, decode_if.data.wid, decode_if.data.PC, instr));
            trace_ex_type(1, decode_if.data.ex_type);
            `TRACE(1, (", op="));
            trace_ex_op(1, decode_if.data.ex_type, decode_if.data.op_type, decode_if.data.op_mod, decode_if.data.rd, decode_if.data.rs2, decode_if.data.use_imm, decode_if.data.imm);
            `TRACE(1, (", mod=%0d, tmask=%b, wb=%b, rd=%0d, rs1=%0d, rs2=%0d, rs3=%0d, imm=0x%0h, opds=%b%b%b%b, use_pc=%b, use_imm=%b (#%0d)\n",
                decode_if.data.op_mod, decode_if.data.tmask, decode_if.data.wb, decode_if.data.rd, decode_if.data.rs1, decode_if.data.rs2, decode_if.data.rs3, decode_if.data.imm, use_rd, use_rs1, use_rs2, use_rs3, decode_if.data.use_PC, decode_if.data.use_imm, decode_if.data.uuid));
        end
    end
`endif

endmodule
