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
	localparam RV_REGS_BITS = `CLOG2(RV_REGS);

`ifdef EXT_F_ENABLE
	localparam REG_TYPES = 2;
`else
	localparam REG_TYPES = 1;
`endif

    localparam NUM_V_REGS = 1 * RV_REGS;

	localparam NUM_REGS = (REG_TYPES * RV_REGS);

	localparam REG_TYPE_BITS = `LOG2UP(REG_TYPES);

	localparam NUM_REGS_BITS = `CLOG2(NUM_REGS);

	localparam REG_EXT_BITS = 2;

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

    localparam IO_MPM_SIZE = (8 * 32 * `NUM_CORES * `NUM_CLUSTERS);

    localparam STACK_SIZE = (1 << `STACK_LOG2_SIZE);

	localparam PC_BITS = (`XLEN-1);

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

	localparam NUM_EX_UNITS = (3 + `EXT_F_ENABLED);
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

    localparam INST_ALU_ADD =       4'b0000;
    //localparam INST_ALU_UNUSED =  4'b0001;
    localparam INST_ALU_LUI =       4'b0010;
    localparam INST_ALU_AUIPC =     4'b0011;
    localparam INST_ALU_SLTU =      4'b0100;
    localparam INST_ALU_SLT =       4'b0101;
    //localparam INST_ALU_UNUSED =  4'b0110;
    localparam INST_ALU_SUB =       4'b0111;
    localparam INST_ALU_SRL =       4'b1000;
    localparam INST_ALU_SRA =       4'b1001;
    localparam INST_ALU_CZEQ =      4'b1010;
    localparam INST_ALU_CZNE =      4'b1011;
    localparam INST_ALU_AND =       4'b1100;
    localparam INST_ALU_OR =        4'b1101;
    localparam INST_ALU_XOR =       4'b1110;
    localparam INST_ALU_SLL =       4'b1111;
    localparam INST_ALU_BITS =      4;

    localparam ALU_TYPE_BITS =      2;
    localparam ALU_TYPE_ARITH =     0;
    localparam ALU_TYPE_BRANCH =    1;
    localparam ALU_TYPE_MULDIV =    2;
    localparam ALU_TYPE_OTHER =     3;

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

    localparam INST_BR_EQ =         4'b0000;
    localparam INST_BR_NE =         4'b0010;
    localparam INST_BR_LTU =        4'b0100;
    localparam INST_BR_GEU =        4'b0110;
    localparam INST_BR_LT =         4'b0101;
    localparam INST_BR_GE =         4'b0111;
    localparam INST_BR_JAL =        4'b1000;
    localparam INST_BR_JALR =       4'b1001;
    localparam INST_BR_ECALL =      4'b1010;
    localparam INST_BR_EBREAK =     4'b1011;
    localparam INST_BR_URET =       4'b1100;
    localparam INST_BR_SRET =       4'b1101;
    localparam INST_BR_MRET =       4'b1110;
    localparam INST_BR_OTHER =      4'b1111;
    localparam INST_BR_BITS =       4;

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

    localparam INST_M_MUL =         3'b000;
    localparam INST_M_MULHU =       3'b001;
    localparam INST_M_MULH =        3'b010;
    localparam INST_M_MULHSU =      3'b011;
    localparam INST_M_DIV =         3'b100;
    localparam INST_M_DIVU =        3'b101;
    localparam INST_M_REM =         3'b110;
    localparam INST_M_REMU =        3'b111;
    localparam INST_M_BITS =        3;

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

    localparam LSU_FMT_B =          3'b000;
    localparam LSU_FMT_H =          3'b001;
    localparam LSU_FMT_W =          3'b010;
    localparam LSU_FMT_D =          3'b011;
    localparam LSU_FMT_BU =         3'b100;
    localparam LSU_FMT_HU =         3'b101;
    localparam LSU_FMT_WU =         3'b110;

    localparam INST_LSU_LB =        4'b0000;
    localparam INST_LSU_LH =        4'b0001;
    localparam INST_LSU_LW =        4'b0010;
    localparam INST_LSU_LD =        4'b0011; // new for RV64I LD
    localparam INST_LSU_LBU =       4'b0100;
    localparam INST_LSU_LHU =       4'b0101;
    localparam INST_LSU_LWU =       4'b0110; // new for RV64I LWU
    localparam INST_LSU_SB =        4'b1000;
    localparam INST_LSU_SH =        4'b1001;
    localparam INST_LSU_SW =        4'b1010;
    localparam INST_LSU_SD =        4'b1011; // new for RV64I SD
    localparam INST_LSU_FENCE =     4'b1111;
    localparam INST_LSU_BITS =      4;

    localparam INST_FENCE_BITS =    1;
    localparam INST_FENCE_D =       1'h0;
    localparam INST_FENCE_I =       1'h1;

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

    localparam INST_FPU_ADD =       4'b0000; // SUB=fmt[1]
    localparam INST_FPU_MUL =       4'b0001;
    localparam INST_FPU_MADD =      4'b0010; // SUB=fmt[1]
    localparam INST_FPU_NMADD =     4'b0011; // SUB=fmt[1]
    localparam INST_FPU_DIV =       4'b0100;
    localparam INST_FPU_SQRT =      4'b0101;
    localparam INST_FPU_F2I =       4'b1000; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_F2U =       4'b1001; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_I2F =       4'b1010; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_U2F =       4'b1011; // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
    localparam INST_FPU_CMP =       4'b1100; // frm: LE=0, LT=1, EQ=2
    localparam INST_FPU_F2F =       4'b1101; // fmt[0]: F32=0, F64=1
    localparam INST_FPU_MISC =      4'b1110; // frm: SGNJ=0, SGNJN=1, SGNJX=2, CLASS=3, MVXW=4, MVWX=5, FMIN=6, FMAX=7
    localparam INST_FPU_BITS =      4;

    function automatic logic inst_fpu_is_class(input logic [INST_FPU_BITS-1:0] op, input logic [INST_FRM_BITS-1:0] frm);
        return (op == INST_FPU_MISC && frm == 3);
    endfunction

    function automatic logic inst_fpu_is_mvxw(input logic [INST_FPU_BITS-1:0] op, input logic [INST_FRM_BITS-1:0] frm);
        return (op == INST_FPU_MISC && frm == 4);
    endfunction

    ///////////////////////////////////////////////////////////////////////////

    localparam INST_SFU_TMC =       4'h0;
    localparam INST_SFU_WSPAWN =    4'h1;
    localparam INST_SFU_SPLIT =     4'h2;
    localparam INST_SFU_JOIN =      4'h3;
    localparam INST_SFU_BAR =       4'h4;
    localparam INST_SFU_PRED =      4'h5;
    localparam INST_SFU_CSRRW =     4'h6;
    localparam INST_SFU_CSRRS =     4'h7;
    localparam INST_SFU_CSRRC =     4'h8;
    localparam INST_SFU_BITS =      4;

    function automatic logic [3:0] inst_sfu_csr(input logic [2:0] funct3);
        return (4'h6 + 4'(funct3[1:0]) - 4'h1);
    endfunction

    function automatic logic inst_sfu_is_wctl(input logic [INST_SFU_BITS-1:0] op);
        return (op <= 5);
    endfunction

    function automatic logic inst_sfu_is_csr(input logic [INST_SFU_BITS-1:0] op);
        return (op >= 6 && op <= 8);
    endfunction

///////////////////////////////////////////////////////////////////////////////

    typedef struct packed {
        logic [REG_EXT_BITS-1:0]  ext;
        logic [REG_TYPE_BITS-1:0] rtype;
        logic [RV_REGS_BITS-1:0]  id;
    } reg_idx_t;

	localparam REG_IDX_BITS = $bits(reg_idx_t);

    typedef struct packed {
        logic                   valid;
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

    typedef struct packed {
        logic use_PC;
        logic use_imm;
        logic is_w;
        logic [ALU_TYPE_BITS-1:0] xtype;
        logic [`XLEN-1:0] imm;
    } alu_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-INST_FRM_BITS-INST_FMT_BITS)-1:0] __padding;
        logic [INST_FRM_BITS-1:0] frm;
        logic [INST_FMT_BITS-1:0] fmt;
    } fpu_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-1-1-OFFSET_BITS)-1:0] __padding;
        logic is_store;
        logic is_float;
        logic [OFFSET_BITS-1:0] offset;
    } lsu_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-1-`VX_CSR_ADDR_BITS-5)-1:0] __padding;
        logic use_imm;
        logic [`VX_CSR_ADDR_BITS-1:0] addr;
        logic [4:0] imm;
    } csr_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-1)-1:0] __padding;
        logic is_neg;
    } wctl_args_t;

    typedef union packed {
        alu_args_t  alu;
        fpu_args_t  fpu;
        lsu_args_t  lsu;
        csr_args_t  csr;
        wctl_args_t wctl;
    } op_args_t;

    localparam INST_ARGS_BITS = $bits(op_args_t);

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
        reg_idx_t                   rd;
        reg_idx_t                   rs1;
        reg_idx_t                   rs2;
        reg_idx_t                   rs3;
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
        reg_idx_t                   rd;
        reg_idx_t                   rs1;
        reg_idx_t                   rs2;
        reg_idx_t                   rs3;
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
        reg_idx_t                   rd;
        reg_idx_t                   rs1;
        reg_idx_t                   rs2;
        reg_idx_t                   rs3;
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

    localparam LSU_WORD_SIZE        = `XLEN / 8;
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

    function automatic logic [NUM_REGS_BITS-1:0] to_reg_number(input reg_idx_t reg_idx);
    `ifdef EXT_F_ENABLE
        return {reg_idx.rtype[0], reg_idx.id};
    `else
        return reg_idx.id;
    `endif
    endfunction

    function automatic logic [RV_REGS-1:0] to_reg_mask(input reg_idx_t reg_idx);
        return ((1 << (1 << reg_idx.ext))-1) << reg_idx.id;
    endfunction

    ////////////////////////////////// Tracing ////////////////////////////////

`ifdef SIMULATION

`ifdef SV_DPI
    import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
`endif

    task trace_reg_idx(input int level, input reg_idx_t reg_id);
        automatic  logic [NUM_REGS_BITS-1:0] reg_base = to_reg_number(reg_id);
        if (reg_id.ext != 0) begin
            automatic logic [NUM_REGS_BITS-1:0] reg_ext = reg_base + (1 << reg_id.ext) - 1;
            `TRACE(level, ("%0d..%0d", reg_base, reg_ext));
        end else begin
            `TRACE(level, ("%0d", reg_base));
        end
    endtask

    task trace_ex_type(input int level, input [EX_BITS-1:0] ex_type);
        case (ex_type)
            EX_ALU: `TRACE(level, ("ALU"))
            EX_LSU: `TRACE(level, ("LSU"))
            EX_SFU: `TRACE(level, ("SFU"))
        `ifdef EXT_F_ENABLE
            EX_FPU: `TRACE(level, ("FPU"))
        `endif
            default: `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [EX_BITS-1:0] ex_type,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        case (ex_type)
        EX_ALU: begin
            case (op_args.alu.xtype)
                ALU_TYPE_ARITH: begin
                    if (op_args.alu.is_w) begin
                        if (op_args.alu.use_imm) begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD: `TRACE(level, ("ADDIW"))
                                INST_ALU_SLL: `TRACE(level, ("SLLIW"))
                                INST_ALU_SRL: `TRACE(level, ("SRLIW"))
                                INST_ALU_SRA: `TRACE(level, ("SRAIW"))
                                default:      `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD: `TRACE(level, ("ADDW"))
                                INST_ALU_SUB: `TRACE(level, ("SUBW"))
                                INST_ALU_SLL: `TRACE(level, ("SLLW"))
                                INST_ALU_SRL: `TRACE(level, ("SRLW"))
                                INST_ALU_SRA: `TRACE(level, ("SRAW"))
                                default:      `TRACE(level, ("?"))
                            endcase
                        end
                    end else begin
                        if (op_args.alu.use_imm) begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD:   `TRACE(level, ("ADDI"))
                                INST_ALU_SLL:   `TRACE(level, ("SLLI"))
                                INST_ALU_SRL:   `TRACE(level, ("SRLI"))
                                INST_ALU_SRA:   `TRACE(level, ("SRAI"))
                                INST_ALU_SLT:   `TRACE(level, ("SLTI"))
                                INST_ALU_SLTU:  `TRACE(level, ("SLTIU"))
                                INST_ALU_XOR:   `TRACE(level, ("XORI"))
                                INST_ALU_OR:    `TRACE(level, ("ORI"))
                                INST_ALU_AND:   `TRACE(level, ("ANDI"))
                                INST_ALU_LUI:   `TRACE(level, ("LUI"))
                                INST_ALU_AUIPC: `TRACE(level, ("AUIPC"))
                                default:        `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (INST_ALU_BITS'(op_type))
                                INST_ALU_ADD:   `TRACE(level, ("ADD"))
                                INST_ALU_SUB:   `TRACE(level, ("SUB"))
                                INST_ALU_SLL:   `TRACE(level, ("SLL"))
                                INST_ALU_SRL:   `TRACE(level, ("SRL"))
                                INST_ALU_SRA:   `TRACE(level, ("SRA"))
                                INST_ALU_SLT:   `TRACE(level, ("SLT"))
                                INST_ALU_SLTU:  `TRACE(level, ("SLTU"))
                                INST_ALU_XOR:   `TRACE(level, ("XOR"))
                                INST_ALU_OR:    `TRACE(level, ("OR"))
                                INST_ALU_AND:   `TRACE(level, ("AND"))
                                INST_ALU_CZEQ:  `TRACE(level, ("CZERO.EQZ"))
                                INST_ALU_CZNE:  `TRACE(level, ("CZERO.NEZ"))
                                default:        `TRACE(level, ("?"))
                            endcase
                        end
                    end
                end
                ALU_TYPE_BRANCH: begin
                    case (INST_BR_BITS'(op_type))
                        INST_BR_EQ:    `TRACE(level, ("BEQ"))
                        INST_BR_NE:    `TRACE(level, ("BNE"))
                        INST_BR_LT:    `TRACE(level, ("BLT"))
                        INST_BR_GE:    `TRACE(level, ("BGE"))
                        INST_BR_LTU:   `TRACE(level, ("BLTU"))
                        INST_BR_GEU:   `TRACE(level, ("BGEU"))
                        INST_BR_JAL:   `TRACE(level, ("JAL"))
                        INST_BR_JALR:  `TRACE(level, ("JALR"))
                        INST_BR_ECALL: `TRACE(level, ("ECALL"))
                        INST_BR_EBREAK:`TRACE(level, ("EBREAK"))
                        INST_BR_URET:  `TRACE(level, ("URET"))
                        INST_BR_SRET:  `TRACE(level, ("SRET"))
                        INST_BR_MRET:  `TRACE(level, ("MRET"))
                        default:       `TRACE(level, ("?"))
                    endcase
                end
                ALU_TYPE_MULDIV: begin
                    if (op_args.alu.is_w) begin
                        case (INST_M_BITS'(op_type))
                            INST_M_MUL:  `TRACE(level, ("MULW"))
                            INST_M_DIV:  `TRACE(level, ("DIVW"))
                            INST_M_DIVU: `TRACE(level, ("DIVUW"))
                            INST_M_REM:  `TRACE(level, ("REMW"))
                            INST_M_REMU: `TRACE(level, ("REMUW"))
                            default:      `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (INST_M_BITS'(op_type))
                            INST_M_MUL:   `TRACE(level, ("MUL"))
                            INST_M_MULH:  `TRACE(level, ("MULH"))
                            INST_M_MULHSU:`TRACE(level, ("MULHSU"))
                            INST_M_MULHU: `TRACE(level, ("MULHU"))
                            INST_M_DIV:   `TRACE(level, ("DIV"))
                            INST_M_DIVU:  `TRACE(level, ("DIVU"))
                            INST_M_REM:   `TRACE(level, ("REM"))
                            INST_M_REMU:  `TRACE(level, ("REMU"))
                            default:      `TRACE(level, ("?"))
                        endcase
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
        EX_LSU: begin
            if (op_args.lsu.is_float) begin
                case (INST_LSU_BITS'(op_type))
                    INST_LSU_LW: `TRACE(level, ("FLW"))
                    INST_LSU_LD: `TRACE(level, ("FLD"))
                    INST_LSU_SW: `TRACE(level, ("FSW"))
                    INST_LSU_SD: `TRACE(level, ("FSD"))
                    default:     `TRACE(level, ("?"))
                endcase
            end else begin
                case (INST_LSU_BITS'(op_type))
                    INST_LSU_LB: `TRACE(level, ("LB"))
                    INST_LSU_LH: `TRACE(level, ("LH"))
                    INST_LSU_LW: `TRACE(level, ("LW"))
                    INST_LSU_LD: `TRACE(level, ("LD"))
                    INST_LSU_LBU:`TRACE(level, ("LBU"))
                    INST_LSU_LHU:`TRACE(level, ("LHU"))
                    INST_LSU_LWU:`TRACE(level, ("LWU"))
                    INST_LSU_SB: `TRACE(level, ("SB"))
                    INST_LSU_SH: `TRACE(level, ("SH"))
                    INST_LSU_SW: `TRACE(level, ("SW"))
                    INST_LSU_SD: `TRACE(level, ("SD"))
                    INST_LSU_FENCE:`TRACE(level,("FENCE"))
                    default:     `TRACE(level, ("?"))
                endcase
            end
        end
        EX_SFU: begin
            case (INST_SFU_BITS'(op_type))
                INST_SFU_TMC:   `TRACE(level, ("TMC"))
                INST_SFU_WSPAWN:`TRACE(level, ("WSPAWN"))
                INST_SFU_SPLIT: begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("SPLIT.N"))
                    end else begin
                        `TRACE(level, ("SPLIT"))
                    end
                end
                INST_SFU_JOIN:  `TRACE(level, ("JOIN"))
                INST_SFU_BAR:   `TRACE(level, ("BAR"))
                INST_SFU_PRED:  begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("PRED.N"))
                    end else begin
                        `TRACE(level, ("PRED"))
                    end
                end
                INST_SFU_CSRRW: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRWI"))
                    end else begin
                        `TRACE(level, ("CSRRW"))
                    end
                end
                INST_SFU_CSRRS: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRSI"))
                    end else begin
                        `TRACE(level, ("CSRRS"))
                    end
                end
                INST_SFU_CSRRC: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRCI"))
                    end else begin
                        `TRACE(level, ("CSRRC"))
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
    `ifdef EXT_F_ENABLE
        EX_FPU: begin
            case (INST_FPU_BITS'(op_type))
                INST_FPU_ADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FSUB.D"))
                        end else begin
                            `TRACE(level, ("FSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FADD.D"))
                        end else begin
                            `TRACE(level, ("FADD.S"))
                        end
                    end
                end
                INST_FPU_MADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FMSUB.D"))
                        end else begin
                            `TRACE(level, ("FMSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FMADD.D"))
                        end else begin
                            `TRACE(level, ("FMADD.S"))
                        end
                    end
                end
                INST_FPU_NMADD: begin
                    if (op_args.fpu.fmt[1]) begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FNMSUB.D"))
                        end else begin
                            `TRACE(level, ("FNMSUB.S"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[0]) begin
                            `TRACE(level, ("FNMADD.D"))
                        end else begin
                            `TRACE(level, ("FNMADD.S"))
                        end
                    end
                end
                INST_FPU_MUL: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FMUL.D"))
                    end else begin
                        `TRACE(level, ("FMUL.S"))
                        end
                end
                INST_FPU_DIV: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FDIV.D"))
                    end else begin
                        `TRACE(level, ("FDIV.S"))
                        end
                end
                INST_FPU_SQRT: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FSQRT.D"))
                    end else begin
                        `TRACE(level, ("FSQRT.S"))
                    end
                end
                INST_FPU_CMP: begin
                    if (op_args.fpu.fmt[0]) begin
                        case (op_args.fpu.frm[1:0])
                        0:       `TRACE(level, ("FLE.D"))
                        1:       `TRACE(level, ("FLT.D"))
                        2:       `TRACE(level, ("FEQ.D"))
                        default: `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (op_args.fpu.frm[1:0])
                        0:       `TRACE(level, ("FLE.S"))
                        1:       `TRACE(level, ("FLT.S"))
                        2:       `TRACE(level, ("FEQ.S"))
                        default: `TRACE(level, ("?"))
                        endcase
                    end
                end
                INST_FPU_F2F: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FCVT.D.S"))
                    end else begin
                        `TRACE(level, ("FCVT.S.D"))
                    end
                end
                INST_FPU_F2I: begin
                    if (op_args.fpu.fmt[0]) begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.L.D"))
                        end else begin
                            `TRACE(level, ("FCVT.W.D"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.L.S"))
                        end else begin
                            `TRACE(level, ("FCVT.W.S"))
                        end
                    end
                end
                INST_FPU_F2U: begin
                    if (op_args.fpu.fmt[0]) begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.LU.D"))
                        end else begin
                            `TRACE(level, ("FCVT.WU.D"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.LU.S"))
                        end else begin
                            `TRACE(level, ("FCVT.WU.S"))
                        end
                    end
                end
                INST_FPU_I2F: begin
                    if (op_args.fpu.fmt[0]) begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.D.L"))
                        end else begin
                            `TRACE(level, ("FCVT.D.W"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.S.L"))
                        end else begin
                            `TRACE(level, ("FCVT.S.W"))
                        end
                    end
                end
                INST_FPU_U2F: begin
                    if (op_args.fpu.fmt[0]) begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.D.LU"))
                        end else begin
                            `TRACE(level, ("FCVT.D.WU"))
                        end
                    end else begin
                        if (op_args.fpu.fmt[1]) begin
                            `TRACE(level, ("FCVT.S.LU"))
                        end else begin
                            `TRACE(level, ("FCVT.S.WU"))
                        end
                    end
                end
                INST_FPU_MISC: begin
                    if (op_args.fpu.fmt[0]) begin
                        case (op_args.fpu.frm)
                            0: `TRACE(level, ("FSGNJ.D"))
                            1: `TRACE(level, ("FSGNJN.D"))
                            2: `TRACE(level, ("FSGNJX.D"))
                            3: `TRACE(level, ("FCLASS.D"))
                            4: `TRACE(level, ("FMV.X.D"))
                            5: `TRACE(level, ("FMV.D.X"))
                            6: `TRACE(level, ("FMIN.D"))
                            7: `TRACE(level, ("FMAX.D"))
                        endcase
                    end else begin
                        case (op_args.fpu.frm)
                            0: `TRACE(level, ("FSGNJ.S"))
                            1: `TRACE(level, ("FSGNJN.S"))
                            2: `TRACE(level, ("FSGNJX.S"))
                            3: `TRACE(level, ("FCLASS.S"))
                            4: `TRACE(level, ("FMV.X.S"))
                            5: `TRACE(level, ("FMV.S.X"))
                            6: `TRACE(level, ("FMIN.S"))
                            7: `TRACE(level, ("FMAX.S"))
                        endcase
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
    `endif
        default: `TRACE(level, ("?"))
        endcase
    endtask

    task trace_op_args(input int level,
                       input [EX_BITS-1:0] ex_type,
                       input [INST_OP_BITS-1:0] op_type,
                       input op_args_t op_args
    );
        case (ex_type)
        EX_ALU: begin
            `TRACE(level, ("use_PC=%b, use_imm=%b, imm=0x%0h", op_args.alu.use_PC, op_args.alu.use_imm, op_args.alu.imm))
        end
        EX_LSU: begin
            `TRACE(level, ("offset=0x%0h", op_args.lsu.offset))
        end
        EX_SFU: begin
            if (inst_sfu_is_csr(op_type)) begin
                `TRACE(level, ("addr=0x%0h, use_imm=%b, imm=0x%0h", op_args.csr.addr, op_args.csr.use_imm, op_args.csr.imm))
            end
        end
    `ifdef EXT_F_ENABLE
        EX_FPU: begin
            `TRACE(level, ("fmt=0x%0h, frm=0x%0h", op_args.fpu.fmt, op_args.fpu.frm))
        end
    `endif
        default:;
        endcase
    endtask

    task trace_base_dcr(input int level, input [VX_DCR_ADDR_WIDTH-1:0] addr);
        case (addr)
            `VX_DCR_BASE_STARTUP_ADDR0: `TRACE(level, ("STARTUP_ADDR0"))
            `VX_DCR_BASE_STARTUP_ADDR1: `TRACE(level, ("STARTUP_ADDR1"))
            `VX_DCR_BASE_STARTUP_ARG0:  `TRACE(level, ("STARTUP_ARG0"))
            `VX_DCR_BASE_STARTUP_ARG1:  `TRACE(level, ("STARTUP_ARG1"))
            `VX_DCR_BASE_MPM_CLASS:     `TRACE(level, ("MPM_CLASS"))
            default:                    `TRACE(level, ("?"))
        endcase
    endtask

`endif

endpackage

`IGNORE_UNUSED_END

`endif // VX_GPU_PKG_VH
