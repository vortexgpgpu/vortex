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

`ifndef VX_GPU_PKG_VH
`define VX_GPU_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_gpu_pkg;

	localparam NC_BITS = `CLOG2(`NUM_CORES);
	localparam NW_BITS = `CLOG2(`NUM_WARPS);
	localparam NT_BITS = `CLOG2(`NUM_THREADS);
	localparam NB_BITS = `CLOG2(`NUM_BARRIERS);

	localparam NC_WIDTH = `UP(NC_BITS);
	localparam NW_WIDTH = `UP(NW_BITS);
	localparam NT_WIDTH = `UP(NT_BITS);
	localparam NB_WIDTH = `UP(NB_BITS);

    localparam XLENB    = `XLEN / 8;

	localparam RV_REGS = 32;
	localparam RV_REGS_BITS = 5;

    localparam REG_TYPE_I = 0;
    localparam REG_TYPE_F = 1;

`ifdef EXT_F_ENABLE
	localparam REG_TYPES = 2;
`else
	localparam REG_TYPES = 1;
`endif

	localparam NUM_REGS = (REG_TYPES * RV_REGS);

	localparam REG_TYPE_BITS = `LOG2UP(REG_TYPES);

	localparam NUM_REGS_BITS = `CLOG2(NUM_REGS);

	localparam DV_STACK_SIZE = `UP(`NUM_THREADS-1);
	localparam DV_STACK_SIZEW = `UP(`CLOG2(DV_STACK_SIZE));

	localparam PERF_CTR_BITS = 44;

    localparam SIMD_COUNT = `NUM_THREADS / `SIMD_WIDTH;
    localparam SIMD_IDX_BITS = `CLOG2(SIMD_COUNT);
    localparam SIMD_IDX_W = `UP(SIMD_IDX_BITS);

    localparam NUM_OPCS_BITS = `CLOG2(`NUM_OPCS);
    localparam NUM_OPCS_W = `UP(NUM_OPCS_BITS);

`ifndef NDEBUG
	localparam UUID_WIDTH = 44;
`else
`ifdef SCOPE
	localparam UUID_WIDTH = 44;
`else
	localparam UUID_WIDTH = 1;
`endif
`endif

`ifndef NDEBUG
	localparam PC_BITS = `XLEN;
    function automatic logic [`XLEN-1:0] to_fullPC(input logic[PC_BITS-1:0] pc);
        to_fullPC = pc;
    endfunction
    function automatic logic [PC_BITS-1:0] from_fullPC(input logic[`XLEN-1:0] pc);
        from_fullPC = pc;
    endfunction
`else
    localparam PC_BITS = (`XLEN-2);
    function automatic logic [`XLEN-1:0] to_fullPC(input logic[PC_BITS-1:0] pc);
        to_fullPC = {pc, 2'b0};
    endfunction
    function automatic logic [PC_BITS-1:0] from_fullPC(input logic[`XLEN-1:0] pc);
        from_fullPC = PC_BITS'(pc >> 2);
    endfunction
`endif

	localparam OFFSET_BITS = 12;

    localparam NUM_SRC_OPDS = 3;
    localparam SRC_OPD_BITS = `CLOG2(NUM_SRC_OPDS);
    localparam SRC_OPD_WIDTH = `UP(SRC_OPD_BITS);

	localparam NUM_SOCKETS = `UP(`NUM_CORES / `SOCKET_SIZE);

    localparam MEM_REQ_FLAG_FLUSH =  0;
    localparam MEM_REQ_FLAG_IO =     1;
    localparam MEM_REQ_FLAG_LOCAL =  2; // shoud be last since optional
    localparam MEM_FLAGS_WIDTH = (MEM_REQ_FLAG_LOCAL + `LMEM_ENABLED);

    localparam VX_DCR_ADDR_WIDTH = `VX_DCR_ADDR_BITS;
    localparam VX_DCR_DATA_WIDTH = 32;

    localparam STALL_TIMEOUT = (100000 * (1 ** (`L2_ENABLED + `L3_ENABLED)));

    ///////////////////////////////////////////////////////////////////////////

	localparam EX_ALU = 0;
	localparam EX_LSU = 1;
	localparam EX_SFU = 2;
	localparam EX_FPU = (EX_SFU + `EXT_F_ENABLED);
    localparam EX_TCU = (EX_FPU + `EXT_TCU_ENABLED);

	localparam NUM_EX_UNITS = EX_TCU + 1;
	localparam EX_BITS = `CLOG2(NUM_EX_UNITS);
	localparam EX_WIDTH = `UP(EX_BITS);

	localparam SFU_CSRS = 0;
	localparam SFU_WCTL = 1;

	localparam NUM_SFU_UNITS = (2);
	localparam SFU_BITS = `CLOG2(NUM_SFU_UNITS);
	localparam SFU_WIDTH = `UP(SFU_BITS);

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_LUI =        7'b0110111;
    localparam INST_AUIPC =      7'b0010111;
    localparam INST_JAL =        7'b1101111;
    localparam INST_JALR =       7'b1100111;
    localparam INST_B =          7'b1100011; // branch instructions
    localparam INST_L =          7'b0000011; // load instructions
    localparam INST_S =          7'b0100011; // store instructions
    localparam INST_I =          7'b0010011; // immediate instructions
    localparam INST_R =          7'b0110011; // register instructions
    localparam INST_V =          7'b1010111; // vector instructions
    localparam INST_FENCE =      7'b0001111; // Fence instructions
    localparam INST_SYS =        7'b1110011; // system instructions

    // RV64I instruction specific opcodes (for any W instruction)
    localparam INST_I_W =        7'b0011011; // W type immediate instructions
    localparam INST_R_W =        7'b0111011; // W type register instructions

    localparam INST_FL =         7'b0000111; // float load instruction
    localparam INST_FS =         7'b0100111; // float store  instruction
    localparam INST_FMADD =      7'b1000011;
    localparam INST_FMSUB =      7'b1000111;
    localparam INST_FNMSUB =     7'b1001011;
    localparam INST_FNMADD =     7'b1001111;
    localparam INST_FCI =        7'b1010011; // float common instructions

    // Custom extension opcodes
    localparam INST_EXT1 =       7'b0001011; // 0x0B
    localparam INST_EXT2 =       7'b0101011; // 0x2B
    localparam INST_EXT3 =       7'b1011011; // 0x5B
    localparam INST_EXT4 =       7'b1111011; // 0x7B

    // Opcode extensions
    localparam INST_R_F7_MUL =   7'b0000001;
    localparam INST_R_F7_ZICOND= 7'b0000111;

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_FRM_RNE =    3'b000;  // round to nearest even
    localparam INST_FRM_RTZ =    3'b001;  // round to zero
    localparam INST_FRM_RDN =    3'b010;  // round to -inf
    localparam INST_FRM_RUP =    3'b011;  // round to +inf
    localparam INST_FRM_RMM =    3'b100;  // round to nearest max magnitude
    localparam INST_FRM_DYN =    3'b111;  // dynamic mode
    localparam INST_FRM_BITS =   3;

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_OP_BITS =    4;
    localparam INST_FMT_BITS =   2;

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_ALU_ADD =    4'b0000;
    //localparam INST_ALU_UNUSED=4'b0001;
    localparam INST_ALU_LUI =    4'b0010;
    localparam INST_ALU_AUIPC =  4'b0011;
    localparam INST_ALU_SLTU =   4'b0100;
    localparam INST_ALU_SLT =    4'b0101;
    //localparam INST_ALU_UNUSED=4'b0110;
    localparam INST_ALU_SUB =    4'b0111;
    localparam INST_ALU_SRL =    4'b1000;
    localparam INST_ALU_SRA =    4'b1001;
    localparam INST_ALU_CZEQ =   4'b1010;
    localparam INST_ALU_CZNE =   4'b1011;
    localparam INST_ALU_AND =    4'b1100;
    localparam INST_ALU_OR =     4'b1101;
    localparam INST_ALU_XOR =    4'b1110;
    localparam INST_ALU_SLL =    4'b1111;
    localparam INST_ALU_BITS =   4;

    localparam ALU_TYPE_BITS =   2;
    localparam ALU_TYPE_ARITH =  0;
    localparam ALU_TYPE_BRANCH = 1;
    localparam ALU_TYPE_MULDIV = 2;
    localparam ALU_TYPE_OTHER =  3;

    function automatic logic [1:0] inst_alu_class(input logic [INST_ALU_BITS-1:0] op);
        return op[3:2];
    endfunction

    function automatic logic inst_alu_signed(input logic [INST_ALU_BITS-1:0] op);
        return op[0];
    endfunction

    function automatic logic inst_alu_is_sub(input logic [INST_ALU_BITS-1:0] op);
        return op[1];
    endfunction

    function automatic logic inst_alu_is_czero(input logic [INST_ALU_BITS-1:0] op);
        return (op[3:1] == 3'b101);
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_BR_BEQ =     4'b0000;
    localparam INST_BR_BNE =     4'b0010;
    localparam INST_BR_BLTU =    4'b0100;
    localparam INST_BR_BGEU =    4'b0110;
    localparam INST_BR_BLT =     4'b0101;
    localparam INST_BR_BGE =     4'b0111;
    localparam INST_BR_JAL =     4'b1000;
    localparam INST_BR_JALR =    4'b1001;
    localparam INST_BR_ECALL =   4'b1010;
    localparam INST_BR_EBREAK =  4'b1011;
    localparam INST_BR_URET =    4'b1100;
    localparam INST_BR_SRET =    4'b1101;
    localparam INST_BR_MRET =    4'b1110;
    localparam INST_BR_OTHER =   4'b1111;
    localparam INST_BR_BITS =    4;

    function automatic logic [1:0] inst_br_class(input logic [INST_BR_BITS-1:0] op);
        return {1'b0, ~op[3]};
    endfunction

    function automatic logic inst_br_is_neg(input logic [INST_BR_BITS-1:0] op);
        return op[1];
    endfunction

    function automatic logic inst_br_is_less(input logic [INST_BR_BITS-1:0] op);
        return op[2];
    endfunction

    function automatic logic inst_br_is_static(input logic [INST_BR_BITS-1:0] op);
        return op[3];
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    // Shuffle & Vote Extension

    localparam INST_VOTE_ALL =   2'b00;
    localparam INST_VOTE_ANY =   2'b01;
    localparam INST_VOTE_UNI =   2'b10;
    localparam INST_VOTE_BAL =   2'b11;

    localparam INST_SHFL_UP =    2'b00;
    localparam INST_SHFL_DOWN =  2'b01;
    localparam INST_SHFL_BFLY =  2'b10;
    localparam INST_SHFL_IDX =   2'b11;

    localparam INST_VOTE_BITS =  2;
    localparam INST_SHFL_BITS =  2;

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_M_MUL =      3'b000;
    localparam INST_M_MULHU =    3'b001;
    localparam INST_M_MULH =     3'b010;
    localparam INST_M_MULHSU =   3'b011;
    localparam INST_M_DIV =      3'b100;
    localparam INST_M_DIVU =     3'b101;
    localparam INST_M_REM =      3'b110;
    localparam INST_M_REMU =     3'b111;
    localparam INST_M_BITS =     3;

    function automatic logic inst_m_signed(input logic [INST_M_BITS-1:0] op);
        return (~op[0]);
    endfunction

    function automatic logic inst_m_is_mulx(input logic [INST_M_BITS-1:0] op);
        return (~op[2]);
    endfunction

    function automatic logic inst_m_is_mulh(input logic [INST_M_BITS-1:0] op);
        return (op[1:0] != 0);
    endfunction

    function automatic logic inst_m_signed_a(input logic [INST_M_BITS-1:0] op);
        return (op[1:0] != 1);
    endfunction

    function automatic logic inst_m_is_rem(input logic [INST_M_BITS-1:0] op);
        return op[1];
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    localparam LSU_FMT_B =       3'b000;
    localparam LSU_FMT_H =       3'b001;
    localparam LSU_FMT_W =       3'b010;
    localparam LSU_FMT_D =       3'b011;
    localparam LSU_FMT_BU =      3'b100;
    localparam LSU_FMT_HU =      3'b101;
    localparam LSU_FMT_WU =      3'b110;

    localparam INST_LSU_LB =     4'b0000;
    localparam INST_LSU_LH =     4'b0001;
    localparam INST_LSU_LW =     4'b0010;
    localparam INST_LSU_LD =     4'b0011; // new for RV64I LD
    localparam INST_LSU_LBU =    4'b0100;
    localparam INST_LSU_LHU =    4'b0101;
    localparam INST_LSU_LWU =    4'b0110; // new for RV64I LWU
    localparam INST_LSU_SB =     4'b1000;
    localparam INST_LSU_SH =     4'b1001;
    localparam INST_LSU_SW =     4'b1010;
    localparam INST_LSU_SD =     4'b1011; // new for RV64I SD
    localparam INST_LSU_FENCE =  4'b1111;
    localparam INST_LSU_BITS =   4;

    localparam INST_FENCE_BITS = 1;
    localparam INST_FENCE_D =    1'h0;
    localparam INST_FENCE_I =    1'h1;

    function automatic logic [2:0] inst_lsu_fmt(input logic [INST_LSU_BITS-1:0] op);
        return op[2:0];
    endfunction

    function automatic logic [1:0] inst_lsu_wsize(input logic [INST_LSU_BITS-1:0] op);
        return op[1:0];
    endfunction

    function automatic logic inst_lsu_is_fence(input logic [INST_LSU_BITS-1:0] op);
        return (op[3:2] == 3);
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_FPU_ADD =    4'b0000; // SUB=fmt[1]
    localparam INST_FPU_MUL =    4'b0001;
    localparam INST_FPU_MADD =   4'b0010; // SUB=fmt[1]
    localparam INST_FPU_NMADD =  4'b0011; // SUB=fmt[1]
    localparam INST_FPU_DIV =    4'b0100;
    localparam INST_FPU_SQRT =   4'b0101;
    localparam INST_FPU_F2I =    4'b1000; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_F2U =    4'b1001; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_I2F =    4'b1010; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_U2F =    4'b1011; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_CMP =    4'b1100; // frm: LE=0, LT=1, EQ=2
    localparam INST_FPU_F2F =    4'b1101; // fmt[0]: F32=0, F64=1
    localparam INST_FPU_MISC =   4'b1110; // frm: SGNJ=0, SGNJN=1, SGNJX=2, CLASS=3, MVXW=4, MVWX=5, FMIN=6, FMAX=7
    localparam INST_FPU_BITS =   4;

    function automatic logic inst_fpu_is_class(input logic [INST_FPU_BITS-1:0] op, input logic [INST_FRM_BITS-1:0] frm);
        return (op == INST_FPU_MISC && frm == 3);
    endfunction

    function automatic logic inst_fpu_is_mvxw(input logic [INST_FPU_BITS-1:0] op, input logic [INST_FRM_BITS-1:0] frm);
        return (op == INST_FPU_MISC && frm == 4);
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_SFU_TMC =    4'h0;
    localparam INST_SFU_WSPAWN = 4'h1;
    localparam INST_SFU_SPLIT =  4'h2;
    localparam INST_SFU_JOIN =   4'h3;
    localparam INST_SFU_BAR =    4'h4;
    localparam INST_SFU_PRED =   4'h5;
    localparam INST_SFU_CSRRW =  4'h6;
    localparam INST_SFU_CSRRS =  4'h7;
    localparam INST_SFU_CSRRC =  4'h8;
    localparam INST_SFU_BITS =   4;

    function automatic logic [3:0] inst_sfu_csr(input logic [2:0] funct3);
        return (4'h6 + 4'(funct3[1:0]) - 4'h1);
    endfunction

    function automatic logic inst_sfu_is_wctl(input logic [INST_SFU_BITS-1:0] op);
        return (op <= 5);
    endfunction

    function automatic logic inst_sfu_is_csr(input logic [INST_SFU_BITS-1:0] op);
        return (op >= 6 && op <= 8);
    endfunction

    /////////////////////////////// Issue parameters //////////////////////////

    localparam ISSUE_ISW_BITS = `CLOG2(`ISSUE_WIDTH);
    localparam ISSUE_ISW_W = `UP(ISSUE_ISW_BITS);
    localparam PER_ISSUE_WARPS = `NUM_WARPS / `ISSUE_WIDTH;
    localparam ISSUE_WIS_BITS = `CLOG2(PER_ISSUE_WARPS);
    localparam ISSUE_WIS_W = `UP(ISSUE_WIS_BITS);

    function automatic logic [NW_WIDTH-1:0] wis_to_wid(
        input logic [ISSUE_WIS_W-1:0] wis,
        input logic [ISSUE_ISW_W-1:0] isw
    );
        if (ISSUE_WIS_BITS == 0) begin
            wis_to_wid = NW_WIDTH'(isw);
        end else if (ISSUE_ISW_BITS == 0) begin
            wis_to_wid = NW_WIDTH'(wis);
        end else begin
            wis_to_wid = NW_WIDTH'({wis, isw});
        end
    endfunction

    function automatic logic [ISSUE_ISW_W-1:0] wid_to_isw(
        input logic [NW_WIDTH-1:0] wid
    );
        if (ISSUE_ISW_BITS != 0) begin
            wid_to_isw = wid[ISSUE_ISW_W-1:0];
        end else begin
            wid_to_isw = 0;
        end
    endfunction

    function automatic logic [ISSUE_WIS_W-1:0] wid_to_wis(
        input logic [NW_WIDTH-1:0] wid
    );
        if (ISSUE_WIS_BITS != 0) begin
            wid_to_wis = ISSUE_WIS_W'(wid >> ISSUE_ISW_BITS);
        end else begin
            wid_to_wis = 0;
        end
    endfunction

    /////////////////////////////// TENSOR UNIT ///////////////////////////////

`ifdef EXT_TCU_ENABLE

    localparam INST_TCU_WMMA = 4'h0;
    localparam INST_TCU_BITS = 4;

`endif

    ///////////////////////////////////////////////////////////////////////////

    typedef struct packed {
        logic                    valid;
        logic [`NUM_THREADS-1:0] tmask;
    } tmc_t;

    typedef struct packed {
        logic                   valid;
        logic [`NUM_WARPS-1:0]  wmask;
        logic [PC_BITS-1:0]     pc;
    } wspawn_t;

    typedef struct packed {
        logic                   valid;
        logic                   is_dvg;
        logic [`NUM_THREADS-1:0] then_tmask;
        logic [`NUM_THREADS-1:0] else_tmask;
        logic [PC_BITS-1:0]     next_pc;
    } split_t;

    typedef struct packed {
        logic                   valid;
        logic [DV_STACK_SIZEW-1:0] stack_ptr;
    } join_t;

    typedef struct packed {
        logic                   valid;
        logic [NB_WIDTH-1:0]    id;
        logic                   is_global;
    `ifdef GBAR_ENABLE
        logic [`MAX(NW_WIDTH, NC_WIDTH)-1:0] size_m1;
    `else
        logic [NW_WIDTH-1:0]    size_m1;
    `endif
        logic                   is_noop;
    } barrier_t;

    typedef struct packed {
        logic [`XLEN-1:0]   startup_addr;
        logic [`XLEN-1:0]   startup_arg;
        logic [7:0]         mpm_class;
    } base_dcrs_t;

    //////////////////////// instruction arguments ////////////////////////////

    localparam INST_ARGS_BITS = ALU_TYPE_BITS + `XLEN + 3;

    typedef struct packed {
        logic use_PC;
        logic use_imm;
        logic is_w;
        logic [ALU_TYPE_BITS-1:0] xtype;
        logic [`XLEN-1:0] imm;
    } alu_args_t;
    `PACKAGE_ASSERT($bits(alu_args_t) == INST_ARGS_BITS)

    typedef struct packed {
        logic [(INST_ARGS_BITS-INST_FRM_BITS-INST_FMT_BITS)-1:0] __padding;
        logic [INST_FRM_BITS-1:0] frm;
        logic [INST_FMT_BITS-1:0] fmt;
    } fpu_args_t;
    `PACKAGE_ASSERT($bits(fpu_args_t) == INST_ARGS_BITS)

    typedef struct packed {
        logic [(INST_ARGS_BITS-1-1-OFFSET_BITS)-1:0] __padding;
        logic is_store;
        logic is_float;
        logic [OFFSET_BITS-1:0] offset;
    } lsu_args_t;
    `PACKAGE_ASSERT($bits(lsu_args_t) == INST_ARGS_BITS)

    typedef struct packed {
        logic [(INST_ARGS_BITS-1-`VX_CSR_ADDR_BITS-5)-1:0] __padding;
        logic use_imm;
        logic [`VX_CSR_ADDR_BITS-1:0] addr;
        logic [4:0] imm;
    } csr_args_t;
    `PACKAGE_ASSERT($bits(csr_args_t) == INST_ARGS_BITS)

    typedef struct packed {
        logic [(INST_ARGS_BITS-1)-1:0] __padding;
        logic is_neg;
    } wctl_args_t;
    `PACKAGE_ASSERT($bits(wctl_args_t) == INST_ARGS_BITS)

`ifdef EXT_TCU_ENABLE
    typedef struct packed {
        logic [(INST_ARGS_BITS-16)-1:0] __padding;
        logic [3:0] fmt_d;
        logic [3:0] fmt_s;
        logic [3:0] step_n;
        logic [3:0] step_m;
    } tcu_args_t;
    `PACKAGE_ASSERT($bits(tcu_args_t) == INST_ARGS_BITS)
`endif

    typedef union packed {
        alu_args_t  alu;
        fpu_args_t  fpu;
        lsu_args_t  lsu;
        csr_args_t  csr;
        wctl_args_t wctl;
    `ifdef EXT_TCU_ENABLE
        tcu_args_t  tcu;
    `endif
    } op_args_t;
    `PACKAGE_ASSERT($bits(op_args_t) == INST_ARGS_BITS)

    //////////////////////////// Pipeline Data Types //////////////////////////

    typedef struct packed {
        logic [UUID_WIDTH-1:0]  uuid;
        logic [NW_WIDTH-1:0]    wid;
        logic [`NUM_THREADS-1:0] tmask;
        logic [PC_BITS-1:0]     PC;
        logic [31:0]            instr;
    } fetch_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]      uuid;
        logic [NW_WIDTH-1:0]        wid;
        logic [`NUM_THREADS-1:0]    tmask;
        logic [PC_BITS-1:0]         PC;
        logic [EX_BITS-1:0]         ex_type;
        logic [INST_OP_BITS-1:0]    op_type;
        op_args_t                   op_args;
        logic                       wb;
        logic [NUM_SRC_OPDS-1:0]    used_rs;
        logic [NUM_REGS_BITS-1:0]   rd;
        logic [NUM_REGS_BITS-1:0]   rs1;
        logic [NUM_REGS_BITS-1:0]   rs2;
        logic [NUM_REGS_BITS-1:0]   rs3;
    } decode_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]      uuid;
        logic [`NUM_THREADS-1:0]    tmask;
        logic [PC_BITS-1:0]         PC;
        logic [EX_BITS-1:0]         ex_type;
        logic [INST_OP_BITS-1:0]    op_type;
        op_args_t                   op_args;
        logic                       wb;
        logic [NUM_SRC_OPDS-1:0]    used_rs;
        logic [NUM_REGS_BITS-1:0]   rd;
        logic [NUM_REGS_BITS-1:0]   rs1;
        logic [NUM_REGS_BITS-1:0]   rs2;
        logic [NUM_REGS_BITS-1:0]   rs3;
    } ibuffer_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]      uuid;
        logic [ISSUE_WIS_W-1:0]     wis;
        logic [`NUM_THREADS-1:0]    tmask;
        logic [PC_BITS-1:0]         PC;
        logic [EX_BITS-1:0]         ex_type;
        logic [INST_OP_BITS-1:0]    op_type;
        op_args_t                   op_args;
        logic                       wb;
        logic [NUM_SRC_OPDS-1:0]    used_rs;
        logic [NUM_REGS_BITS-1:0]   rd;
        logic [NUM_REGS_BITS-1:0]   rs1;
        logic [NUM_REGS_BITS-1:0]   rs2;
        logic [NUM_REGS_BITS-1:0]   rs3;
    } scoreboard_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]              uuid;
        logic [ISSUE_WIS_W-1:0]             wis;
        logic [SIMD_IDX_W-1:0]              sid;
        logic [`SIMD_WIDTH-1:0]             tmask;
        logic [PC_BITS-1:0]                 PC;
        logic [EX_BITS-1:0]                 ex_type;
        logic [INST_OP_BITS-1:0]            op_type;
        op_args_t                           op_args;
        logic                               wb;
        logic [NUM_REGS_BITS-1:0]           rd;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs1_data;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs2_data;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs3_data;
        logic                               sop;
        logic                               eop;
    } operands_t;

    // warning: this layout should not be modified without updating VX_dispatch_unit!!!
    typedef struct packed {
        logic [UUID_WIDTH-1:0]              uuid;
        logic [ISSUE_WIS_W-1:0]             wis;
        logic [SIMD_IDX_W-1:0]              sid;
        logic [`SIMD_WIDTH-1:0]             tmask;
        logic [PC_BITS-1:0]                 PC;
        logic [INST_ALU_BITS-1:0]           op_type;
        op_args_t                           op_args;
        logic                               wb;
        logic [NUM_REGS_BITS-1:0]           rd;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs1_data;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs2_data;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  rs3_data;
        logic                               sop;
        logic                               eop;
    } dispatch_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]              uuid;
        logic [NW_WIDTH-1:0]                wid;
        logic [SIMD_IDX_W-1:0]              sid;
        logic [`SIMD_WIDTH-1:0]             tmask;
        logic [PC_BITS-1:0]                 PC;
        logic                               wb;
        logic [NUM_REGS_BITS-1:0]           rd;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  data;
        logic                               sop;
        logic                               eop;
    } commit_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]              uuid;
        logic [ISSUE_WIS_W-1:0]             wis;
        logic [SIMD_IDX_W-1:0]              sid;
        logic [`SIMD_WIDTH-1:0]             tmask;
        logic [PC_BITS-1:0]                 PC;
        logic [NUM_REGS_BITS-1:0]           rd;
        logic [`SIMD_WIDTH-1:0][`XLEN-1:0]  data;
        logic                               sop;
        logic                               eop;
    } writeback_t;

    typedef struct packed {
        logic [UUID_WIDTH-1:0]              uuid;
        logic [NW_WIDTH-1:0]                wid;
        logic [`NUM_THREADS-1:0]            tmask;
        logic [PC_BITS-1:0]                 PC;
    } schedule_t;

    `DECL_EXECUTE_T (alu_exe_t, `NUM_ALU_LANES);
    `DECL_RESULT_T  (alu_res_t, `NUM_ALU_LANES);

    `DECL_EXECUTE_T (lsu_exe_t, `NUM_LSU_LANES);
    `DECL_RESULT_T (lsu_res_t, `NUM_LSU_LANES);

    `DECL_EXECUTE_T (sfu_exe_t, `NUM_SFU_LANES);
    `DECL_RESULT_T (sfu_res_t, `NUM_SFU_LANES);

    //////////////////////////// Perf counter types ///////////////////////////

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] reads;
        logic [PERF_CTR_BITS-1:0] writes;
        logic [PERF_CTR_BITS-1:0] read_misses;
        logic [PERF_CTR_BITS-1:0] write_misses;
        logic [PERF_CTR_BITS-1:0] bank_stalls;
        logic [PERF_CTR_BITS-1:0] mshr_stalls;
        logic [PERF_CTR_BITS-1:0] mem_stalls;
        logic [PERF_CTR_BITS-1:0] crsp_stalls;
    } cache_perf_t;

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] reads;
        logic [PERF_CTR_BITS-1:0] writes;
        logic [PERF_CTR_BITS-1:0] bank_stalls;
        logic [PERF_CTR_BITS-1:0] crsp_stalls;
    } lmem_perf_t;

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] misses;
    } coalescer_perf_t;

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] reads;
        logic [PERF_CTR_BITS-1:0] writes;
        logic [PERF_CTR_BITS-1:0] latency;
    } mem_perf_t;

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] idles;
        logic [PERF_CTR_BITS-1:0] stalls;
    } sched_perf_t;

    typedef struct packed {
        logic [PERF_CTR_BITS-1:0] ibf_stalls;
        logic [PERF_CTR_BITS-1:0] scb_stalls;
        logic [PERF_CTR_BITS-1:0] opd_stalls;
        logic [NUM_EX_UNITS-1:0][PERF_CTR_BITS-1:0] units_uses;
        logic [NUM_SFU_UNITS-1:0][PERF_CTR_BITS-1:0] sfu_uses;
    } issue_perf_t;

    typedef struct packed {
        cache_perf_t icache;
        cache_perf_t dcache;
        cache_perf_t l2cache;
        cache_perf_t l3cache;
        lmem_perf_t  lmem;
        coalescer_perf_t coalescer;
        mem_perf_t   mem;
    } sysmem_perf_t;

    typedef struct packed {
        sched_perf_t              sched;
        issue_perf_t              issue;
        logic [PERF_CTR_BITS-1:0] ifetches;
        logic [PERF_CTR_BITS-1:0] loads;
        logic [PERF_CTR_BITS-1:0] stores;
        logic [PERF_CTR_BITS-1:0] ifetch_latency;
        logic [PERF_CTR_BITS-1:0] load_latency;
   } pipeline_perf_t;

    ///////////////////////// LSU memory Parameters ///////////////////////////

    localparam LSU_WORD_SIZE        = XLENB;
    localparam LSU_ADDR_WIDTH	    = (`MEM_ADDR_WIDTH - `CLOG2(LSU_WORD_SIZE));
    localparam LSU_MEM_BATCHES      = 1;
    localparam LSU_TAG_ID_BITS      = (`CLOG2(`LSUQ_IN_SIZE) + `CLOG2(LSU_MEM_BATCHES));
    localparam LSU_TAG_WIDTH        = (UUID_WIDTH + LSU_TAG_ID_BITS);
    localparam LSU_NUM_REQS	        = `NUM_LSU_BLOCKS * `NUM_LSU_LANES;
    localparam LMEM_TAG_WIDTH       = LSU_TAG_WIDTH + `CLOG2(`NUM_LSU_BLOCKS);

    ////////////////////////// Icache Parameters //////////////////////////////

    // Word size in bytes
    localparam ICACHE_WORD_SIZE	    = 4;
    localparam ICACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(ICACHE_WORD_SIZE));

    // Block size in bytes
    localparam ICACHE_LINE_SIZE	    = `L1_LINE_SIZE;

    // Core request tag Id bits
    localparam ICACHE_TAG_ID_BITS	= NW_WIDTH;

    // Core request tag bits
    localparam ICACHE_TAG_WIDTH	    = (UUID_WIDTH + ICACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam ICACHE_MEM_DATA_WIDTH = (ICACHE_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef ICACHE_ENABLE
    localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`ICACHE_MSHR_SIZE, 1, 1, `NUM_ICACHES, UUID_WIDTH);
`else
    localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_MEM_TAG_WIDTH(1, 1, ICACHE_LINE_SIZE, ICACHE_WORD_SIZE, ICACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_ICACHES);
`endif

    ////////////////////////// Dcache Parameters //////////////////////////////

    // Word size in bytes
    localparam DCACHE_WORD_SIZE	    = `LSU_LINE_SIZE;
    localparam DCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(DCACHE_WORD_SIZE));

    // Block size in bytes
    localparam DCACHE_LINE_SIZE 	= `L1_LINE_SIZE;

    // Input request size (using coalesced memory blocks)
    localparam DCACHE_CHANNELS	    = `UP((`NUM_LSU_LANES * LSU_WORD_SIZE) / DCACHE_WORD_SIZE);
    localparam DCACHE_NUM_REQS	    = `NUM_LSU_BLOCKS * DCACHE_CHANNELS;

    // Core request tag Id bits
    localparam DCACHE_MERGED_REQS   = (`NUM_LSU_LANES * LSU_WORD_SIZE) / DCACHE_WORD_SIZE;
    localparam DCACHE_MEM_BATCHES   = `CDIV(DCACHE_MERGED_REQS, DCACHE_CHANNELS);
    localparam DCACHE_TAG_ID_BITS   = (`CLOG2(`LSUQ_OUT_SIZE) + `CLOG2(DCACHE_MEM_BATCHES));

    // Core request tag bits
    localparam DCACHE_TAG_WIDTH	    = (UUID_WIDTH + DCACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam DCACHE_MEM_DATA_WIDTH = (DCACHE_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef DCACHE_ENABLE
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`DCACHE_MSHR_SIZE, `DCACHE_NUM_BANKS, DCACHE_NUM_REQS, `L1_MEM_PORTS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES, UUID_WIDTH);
`else
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_MEM_TAG_WIDTH(DCACHE_NUM_REQS, `L1_MEM_PORTS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES);
`endif

    /////////////////////////////// L1 Parameters /////////////////////////////

    // arbitrate between icache and dcache
    localparam L1_MEM_TAG_WIDTH     = `MAX(ICACHE_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH);
    localparam L1_MEM_ARB_TAG_WIDTH = (L1_MEM_TAG_WIDTH + `CLOG2(2));

    /////////////////////////////// L2 Parameters /////////////////////////////

    localparam ICACHE_MEM_ARB_IDX   = 0;
    localparam DCACHE_MEM_ARB_IDX   = ICACHE_MEM_ARB_IDX + 1;

    // Word size in bytes
    localparam L2_WORD_SIZE	        = `L1_LINE_SIZE;

    // Input request size
    localparam L2_NUM_REQS	        = NUM_SOCKETS * `L1_MEM_PORTS;

    // Core request tag bits
    localparam L2_TAG_WIDTH	        = L1_MEM_ARB_TAG_WIDTH;

    // Memory request data bits
    localparam L2_MEM_DATA_WIDTH	= (`L2_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef L2_ENABLE
    localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L2_MSHR_SIZE, `L2_NUM_BANKS, L2_NUM_REQS, `L2_MEM_PORTS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH, UUID_WIDTH);
`else
    localparam L2_MEM_TAG_WIDTH     = `CACHE_BYPASS_TAG_WIDTH(L2_NUM_REQS, `L2_MEM_PORTS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
`endif

    /////////////////////////////// L3 Parameters /////////////////////////////

    // Word size in bytes
    localparam L3_WORD_SIZE	        = `L2_LINE_SIZE;

    // Input request size
    localparam L3_NUM_REQS	        = `NUM_CLUSTERS * `L2_MEM_PORTS;

    // Core request tag bits
    localparam L3_TAG_WIDTH	        = L2_MEM_TAG_WIDTH;

    // Memory request data bits
    localparam L3_MEM_DATA_WIDTH	= (`L3_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef L3_ENABLE
    localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L3_MSHR_SIZE, `L3_NUM_BANKS, L3_NUM_REQS, `L3_MEM_PORTS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH, UUID_WIDTH);
`else
    localparam L3_MEM_TAG_WIDTH     = `CACHE_BYPASS_TAG_WIDTH(L3_NUM_REQS, `L3_MEM_PORTS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
`endif

    ///////////////////////////////////////////////////////////////////////////

    localparam VX_MEM_PORTS =           `L3_MEM_PORTS;
    localparam VX_MEM_BYTEEN_WIDTH =    `L3_LINE_SIZE;
    localparam VX_MEM_ADDR_WIDTH =      (`MEM_ADDR_WIDTH - `CLOG2(`L3_LINE_SIZE));
    localparam VX_MEM_DATA_WIDTH =      (`L3_LINE_SIZE * 8);
    localparam VX_MEM_TAG_WIDTH =       L3_MEM_TAG_WIDTH;

    ///////////////////////// Miscaellaneous functions ////////////////////////

    function automatic logic [SFU_WIDTH-1:0] op_to_sfu_type(
        input logic [INST_OP_BITS-1:0] op_type
    );
        case (op_type)
            INST_SFU_CSRRW,
            INST_SFU_CSRRS,
            INST_SFU_CSRRC: op_to_sfu_type = SFU_CSRS;
            default: op_to_sfu_type = SFU_WCTL;
        endcase
    endfunction

    function automatic logic [NUM_REGS_BITS-1:0] make_reg_num(input logic [REG_TYPE_BITS-1:0] rtype, logic [RV_REGS_BITS-1:0] idx);
        return (NUM_REGS_BITS'(rtype) << RV_REGS_BITS) | NUM_REGS_BITS'(idx);
    endfunction

    function automatic logic [REG_TYPE_BITS-1:0] get_reg_type(input logic [NUM_REGS_BITS-1:0] reg_num);
        return REG_TYPE_BITS'(reg_num >> RV_REGS_BITS);
    endfunction

    function automatic logic [RV_REGS_BITS-1:0] get_reg_idx(input logic [NUM_REGS_BITS-1:0] reg_num);
        return reg_num[RV_REGS_BITS-1:0];
    endfunction

endpackage

`IGNORE_UNUSED_END

`endif // VX_GPU_PKG_VH
