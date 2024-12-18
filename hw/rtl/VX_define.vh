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

`ifndef VX_DEFINE_VH
`define VX_DEFINE_VH

`include "VX_platform.vh"
`include "VX_config.vh"
`include "VX_types.vh"

///////////////////////////////////////////////////////////////////////////////

`define NW_BITS         `CLOG2(`NUM_WARPS)
`define NC_WIDTH        `UP(`NC_BITS)

`define NT_BITS         `CLOG2(`NUM_THREADS)
`define NW_WIDTH        `UP(`NW_BITS)

`define NC_BITS         `CLOG2(`NUM_CORES)
`define NT_WIDTH        `UP(`NT_BITS)

`define NB_BITS         `CLOG2(`NUM_BARRIERS)
`define NB_WIDTH        `UP(`NB_BITS)

`define NUM_IREGS       32

`define NRI_BITS        `CLOG2(`NUM_IREGS)

`ifdef EXT_F_ENABLE
`define NUM_REGS        (2 * `NUM_IREGS)
`else
`define NUM_REGS        `NUM_IREGS
`endif

`define NR_BITS         `CLOG2(`NUM_REGS)

`define DV_STACK_SIZE   `UP(`NUM_THREADS-1)
`define DV_STACK_SIZEW  `UP(`CLOG2(`DV_STACK_SIZE))

`define PERF_CTR_BITS   44

`ifndef NDEBUG
`define UUID_ENABLE
`define UUID_WIDTH      44
`else
`ifdef SCOPE
`define UUID_ENABLE
`define UUID_WIDTH      44
`else
`define UUID_WIDTH      1
`endif
`endif

`define PC_BITS         (`XLEN-1)
`define OFFSET_BITS     12
`define IMM_BITS        `XLEN

`define NUM_SOCKETS     `UP(`NUM_CORES / `SOCKET_SIZE)

///////////////////////////////////////////////////////////////////////////////

`define EX_ALU          0
`define EX_LSU          1
`define EX_SFU          2
`define EX_FPU          (`EX_SFU + `EXT_F_ENABLED)

`define NUM_EX_UNITS    (3 + `EXT_F_ENABLED)
`define EX_BITS         `CLOG2(`NUM_EX_UNITS)
`define EX_WIDTH        `UP(`EX_BITS)

`define SFU_CSRS        0
`define SFU_WCTL        1

`define NUM_SFU_UNITS   (2)
`define SFU_BITS        `CLOG2(`NUM_SFU_UNITS)
`define SFU_WIDTH       `UP(`SFU_BITS)

///////////////////////////////////////////////////////////////////////////////

`define INST_LUI        7'b0110111
`define INST_AUIPC      7'b0010111
`define INST_JAL        7'b1101111
`define INST_JALR       7'b1100111
`define INST_B          7'b1100011 // branch instructions
`define INST_L          7'b0000011 // load instructions
`define INST_S          7'b0100011 // store instructions
`define INST_I          7'b0010011 // immediate instructions
`define INST_R          7'b0110011 // register instructions
`define INST_FENCE      7'b0001111 // Fence instructions
`define INST_SYS        7'b1110011 // system instructions

// RV64I instruction specific opcodes (for any W instruction)
`define INST_I_W        7'b0011011 // W type immediate instructions
`define INST_R_W        7'b0111011 // W type register instructions

`define INST_FL         7'b0000111 // float load instruction
`define INST_FS         7'b0100111 // float store  instruction
`define INST_FMADD      7'b1000011
`define INST_FMSUB      7'b1000111
`define INST_FNMSUB     7'b1001011
`define INST_FNMADD     7'b1001111
`define INST_FCI        7'b1010011 // float common instructions

// Custom extension opcodes
`define INST_EXT1       7'b0001011 // 0x0B
`define INST_EXT2       7'b0101011 // 0x2B
`define INST_EXT3       7'b1011011 // 0x5B
`define INST_EXT4       7'b1111011 // 0x7B

// Opcode extensions
`define INST_R_F7_MUL   7'b0000001
`define INST_R_F7_ZICOND 7'b0000111

///////////////////////////////////////////////////////////////////////////////

`define INST_FRM_RNE    3'b000  // round to nearest even
`define INST_FRM_RTZ    3'b001  // round to zero
`define INST_FRM_RDN    3'b010  // round to -inf
`define INST_FRM_RUP    3'b011  // round to +inf
`define INST_FRM_RMM    3'b100  // round to nearest max magnitude
`define INST_FRM_DYN    3'b111  // dynamic mode
`define INST_FRM_BITS   3

///////////////////////////////////////////////////////////////////////////////

`define INST_OP_BITS    4
`define INST_ARGS_BITS   $bits(op_args_t)
`define INST_FMT_BITS   2

///////////////////////////////////////////////////////////////////////////////

`define INST_ALU_ADD         4'b0000
//`define INST_ALU_UNUSED    4'b0001
`define INST_ALU_LUI         4'b0010
`define INST_ALU_AUIPC       4'b0011
`define INST_ALU_SLTU        4'b0100
`define INST_ALU_SLT         4'b0101
//`define INST_ALU_UNUSED    4'b0110
`define INST_ALU_SUB         4'b0111
`define INST_ALU_SRL         4'b1000
`define INST_ALU_SRA         4'b1001
`define INST_ALU_CZEQ        4'b1010
`define INST_ALU_CZNE        4'b1011
`define INST_ALU_AND         4'b1100
`define INST_ALU_OR          4'b1101
`define INST_ALU_XOR         4'b1110
`define INST_ALU_SLL         4'b1111


`define ALU_TYPE_BITS        2
`define ALU_TYPE_ARITH       0
`define ALU_TYPE_BRANCH      1
`define ALU_TYPE_MULDIV      2
`define ALU_TYPE_OTHER       3

`define INST_ALU_BITS        4
`define INST_ALU_CLASS(op)   op[3:2]
`define INST_ALU_SIGNED(op)  op[0]
`define INST_ALU_IS_SUB(op)  op[1]
`define INST_ALU_IS_CZERO(op) (op[3:1] == 3'b101)

`define INST_BR_EQ           4'b0000
`define INST_BR_NE           4'b0010
`define INST_BR_LTU          4'b0100
`define INST_BR_GEU          4'b0110
`define INST_BR_LT           4'b0101
`define INST_BR_GE           4'b0111
`define INST_BR_JAL          4'b1000
`define INST_BR_JALR         4'b1001
`define INST_BR_ECALL        4'b1010
`define INST_BR_EBREAK       4'b1011
`define INST_BR_URET         4'b1100
`define INST_BR_SRET         4'b1101
`define INST_BR_MRET         4'b1110
`define INST_BR_OTHER        4'b1111
`define INST_BR_BITS         4
`define INST_BR_CLASS(op)    {1'b0, ~op[3]}
`define INST_BR_IS_NEG(op)   op[1]
`define INST_BR_IS_LESS(op)  op[2]
`define INST_BR_IS_STATIC(op) op[3]

`define INST_M_MUL           3'b000
`define INST_M_MULHU         3'b001
`define INST_M_MULH          3'b010
`define INST_M_MULHSU        3'b011
`define INST_M_DIV           3'b100
`define INST_M_DIVU          3'b101
`define INST_M_REM           3'b110
`define INST_M_REMU          3'b111
`define INST_M_BITS          3
`define INST_M_SIGNED(op)    (~op[0])
`define INST_M_IS_MULX(op)   (~op[2])
`define INST_M_IS_MULH(op)   (op[1:0] != 0)
`define INST_M_SIGNED_A(op)  (op[1:0] != 1)
`define INST_M_IS_REM(op)    op[1]

`define INST_FMT_B           3'b000
`define INST_FMT_H           3'b001
`define INST_FMT_W           3'b010
`define INST_FMT_D           3'b011
`define INST_FMT_BU          3'b100
`define INST_FMT_HU          3'b101
`define INST_FMT_WU          3'b110

`define INST_LSU_LB          4'b0000
`define INST_LSU_LH          4'b0001
`define INST_LSU_LW          4'b0010
`define INST_LSU_LD          4'b0011 // new for RV64I LD
`define INST_LSU_LBU         4'b0100
`define INST_LSU_LHU         4'b0101
`define INST_LSU_LWU         4'b0110 // new for RV64I LWU
`define INST_LSU_SB          4'b1000
`define INST_LSU_SH          4'b1001
`define INST_LSU_SW          4'b1010
`define INST_LSU_SD          4'b1011 // new for RV64I SD
`define INST_LSU_FENCE       4'b1111
`define INST_LSU_BITS        4
`define INST_LSU_FMT(op)     op[2:0]
`define INST_LSU_WSIZE(op)   op[1:0]
`define INST_LSU_IS_FENCE(op) (op[3:2] == 3)

`define INST_FENCE_BITS      1
`define INST_FENCE_D         1'h0
`define INST_FENCE_I         1'h1

`define INST_FPU_ADD         4'b0000 // SUB=fmt[1]
`define INST_FPU_MUL         4'b0001
`define INST_FPU_MADD        4'b0010 // SUB=fmt[1]
`define INST_FPU_NMADD       4'b0011 // SUB=fmt[1]
`define INST_FPU_DIV         4'b0100
`define INST_FPU_SQRT        4'b0101
`define INST_FPU_F2I         4'b1000 // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
`define INST_FPU_F2U         4'b1001 // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
`define INST_FPU_I2F         4'b1010 // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
`define INST_FPU_U2F         4'b1011 // fmt[0]: F32=0, F64=1, fmt[1]: I32=0, I64=1
`define INST_FPU_CMP         4'b1100 // frm: LE=0, LT=1, EQ=2
`define INST_FPU_F2F         4'b1101 // fmt[0]: F32=0, F64=1
`define INST_FPU_MISC        4'b1110 // frm: SGNJ=0, SGNJN=1, SGNJX=2, CLASS=3, MVXW=4, MVWX=5, FMIN=6, FMAX=7
`define INST_FPU_BITS        4
`define INST_FPU_IS_CLASS(op, frm) (op == `INST_FPU_MISC && frm == 3)
`define INST_FPU_IS_MVXW(op, frm) (op == `INST_FPU_MISC && frm == 4)

`define INST_SFU_TMC         4'h0
`define INST_SFU_WSPAWN      4'h1
`define INST_SFU_SPLIT       4'h2
`define INST_SFU_JOIN        4'h3
`define INST_SFU_BAR         4'h4
`define INST_SFU_PRED        4'h5
`define INST_SFU_CSRRW       4'h6
`define INST_SFU_CSRRS       4'h7
`define INST_SFU_CSRRC       4'h8
`define INST_SFU_BITS        4
`define INST_SFU_CSR(f3)     (4'h6 + 4'(f3) - 4'h1)
`define INST_SFU_IS_WCTL(op) (op <= 5)
`define INST_SFU_IS_CSR(op)  (op >= 6 && op <= 8)

///////////////////////////////////////////////////////////////////////////////

`define ARB_SEL_BITS(I, O)  ((I > O) ? `CLOG2(`CDIV(I, O)) : 0)

///////////////////////////////////////////////////////////////////////////////

`define CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width) \
        (uuid_width + `CLOG2(mshr_size) + `CLOG2(`CDIV(num_banks, mem_ports)))

`define CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width) \
        (`CLOG2(`CDIV(num_reqs, mem_ports)) + `CLOG2(line_size / word_size) + tag_width)

`define CACHE_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, tag_width, uuid_width) \
        (`MAX(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width), `CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width)) + 1)

///////////////////////////////////////////////////////////////////////////////

`define CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches) \
        (tag_width + `ARB_SEL_BITS(num_inputs, `UP(num_caches)))

`define CACHE_CLUSTER_MEM_ARB_TAG(tag_width, num_caches) \
        (tag_width + `ARB_SEL_BITS(`UP(num_caches), 1))

`define CACHE_CLUSTER_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, num_caches, uuid_width) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks, mem_ports, uuid_width), num_caches)

`define CACHE_CLUSTER_BYPASS_MEM_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, tag_width, num_inputs, num_caches) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_BYPASS_TAG_WIDTH(num_reqs, mem_ports, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches)), num_caches)

`define CACHE_CLUSTER_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, tag_width, num_inputs, num_caches, uuid_width) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, mem_ports, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches), uuid_width), num_caches)

///////////////////////////////////////////////////////////////////////////////

`ifdef ICACHE_ENABLE
`define L1_ENABLE
`endif

`ifdef DCACHE_ENABLE
`define L1_ENABLE
`endif

`define MEM_REQ_FLAG_FLUSH      0
`define MEM_REQ_FLAG_IO         1
`define MEM_REQ_FLAG_LOCAL      2 // shoud be last since optional
`define MEM_REQ_FLAGS_WIDTH     (`MEM_REQ_FLAG_LOCAL + `LMEM_ENABLED)

`define VX_MEM_PORTS            `L3_MEM_PORTS
`define VX_MEM_BYTEEN_WIDTH     `L3_LINE_SIZE
`define VX_MEM_ADDR_WIDTH       (`MEM_ADDR_WIDTH - `CLOG2(`L3_LINE_SIZE))
`define VX_MEM_DATA_WIDTH       (`L3_LINE_SIZE * 8)
`define VX_MEM_TAG_WIDTH        L3_MEM_TAG_WIDTH

`define VX_DCR_ADDR_WIDTH       `VX_DCR_ADDR_BITS
`define VX_DCR_DATA_WIDTH       32

`define TO_FULL_ADDR(x)         {x, (`MEM_ADDR_WIDTH-$bits(x))'(0)}

///////////////////////////////////////////////////////////////////////////////

`define NEG_EDGE(dst, src) \
    wire dst; \
    VX_edge_trigger #( \
        .POS  (0), \
        .INIT (0) \
    ) __``dst``__ ( \
        .clk      (clk), \
        .reset    (1'b0), \
        .data_in  (src), \
        .data_out (dst) \
    )

`define BUFFER_EX(dst, src, ena, RSTW, latency) \
    VX_pipe_register #( \
        .DATAW  ($bits(dst)), \
        .RESETW (RSTW), \
        .DEPTH  (latency) \
    ) __``dst``__ ( \
        .clk      (clk), \
        .reset    (reset), \
        .enable   (ena), \
        .data_in  (src), \
        .data_out (dst) \
    )

`define BUFFER(dst, src) `BUFFER_EX(dst, src, 1'b1, 0, 1)

`define POP_COUNT_EX(out, in, model) \
    VX_popcount #( \
        .N ($bits(in)), \
        .MODEL (model) \
    ) __``out``__ ( \
        .data_in  (in), \
        .data_out (out) \
    )

`define POP_COUNT(out, in) `POP_COUNT_EX(out, in, 1)

`define ASSIGN_VX_IF(dst, src) \
    assign dst.valid = src.valid; \
    assign dst.data  = src.data; \
    assign src.ready = dst.ready

`define ASSIGN_VX_MEM_BUS_IF(dst, src) \
    assign dst.req_valid  = src.req_valid; \
    assign dst.req_data   = src.req_data; \
    assign src.req_ready  = dst.req_ready; \
    assign src.rsp_valid  = dst.rsp_valid; \
    assign src.rsp_data   = dst.rsp_data; \
    assign dst.rsp_ready  = src.rsp_ready

`define ASSIGN_VX_MEM_BUS_RO_IF(dst, src) \
    assign dst.req_valid = src.req_valid; \
    assign dst.req_data.rw = 0; \
    assign dst.req_data.addr = src.req_data.addr; \
    assign dst.req_data.data = '0; \
    assign dst.req_data.byteen = '1; \
    assign dst.req_data.flags = src.req_data.flags; \
    assign dst.req_data.tag = src.req_data.tag; \
    assign src.req_ready = dst.req_ready; \
    assign src.rsp_valid = dst.rsp_valid; \
    assign src.rsp_data.data = dst.rsp_data.data; \
    assign src.rsp_data.tag = dst.rsp_data.tag; \
    assign dst.rsp_ready = src.rsp_ready

`define ASSIGN_VX_MEM_BUS_IF_EX(dst, src, TD, TS, UUID) \
    assign dst.req_valid = src.req_valid; \
    assign dst.req_data.rw = src.req_data.rw; \
    assign dst.req_data.addr = src.req_data.addr; \
    assign dst.req_data.data = src.req_data.data; \
    assign dst.req_data.byteen = src.req_data.byteen; \
    assign dst.req_data.flags = src.req_data.flags; \
    /* verilator lint_off GENUNNAMED */ \
    if (TD != TS) begin \
        if (UUID != 0) begin \
            if (TD > TS) begin \
                assign dst.req_data.tag = {src.req_data.tag.uuid, {(TD-TS){1'b0}}, src.req_data.tag.value}; \
            end else begin \
                assign dst.req_data.tag = {src.req_data.tag.uuid, src.req_data.tag.value[TD-UUID-1:0]}; \
            end \
        end else begin \
            if (TD > TS) begin \
                assign dst.req_data.tag = {{(TD-TS){1'b0}}, src.req_data.tag}; \
            end else begin \
                assign dst.req_data.tag = src.req_data.tag[TD-1:0]; \
            end \
        end \
    end else begin \
        assign dst.req_data.tag = src.req_data.tag; \
    end \
    /* verilator lint_on GENUNNAMED */ \
    assign src.req_ready = dst.req_ready; \
    assign src.rsp_valid = dst.rsp_valid; \
    assign src.rsp_data.data = dst.rsp_data.data; \
    /* verilator lint_off GENUNNAMED */ \
    if (TD != TS) begin \
        if (UUID != 0) begin \
            if (TD > TS) begin \
                assign src.rsp_data.tag = {dst.rsp_data.tag.uuid, dst.rsp_data.tag.value[TS-UUID-1:0]}; \
            end else begin \
                assign src.rsp_data.tag = {dst.rsp_data.tag.uuid, {(TS-TD){1'b0}}, dst.rsp_data.tag.value}; \
            end \
        end else begin \
            if (TD > TS) begin \
                assign src.rsp_data.tag = dst.rsp_data.tag[TS-1:0]; \
            end else begin \
                assign src.rsp_data.tag = {{(TS-TD){1'b0}}, dst.rsp_data.tag}; \
            end \
        end \
    end else begin \
        assign src.rsp_data.tag = dst.rsp_data.tag; \
    end \
    /* verilator lint_on GENUNNAMED */ \
    assign dst.rsp_ready = src.rsp_ready

`define INIT_VX_MEM_BUS_IF(itf) \
    assign itf.req_valid = 0; \
    assign itf.req_data = '0; \
    `UNUSED_VAR (itf.req_ready) \
    `UNUSED_VAR (itf.rsp_valid) \
    `UNUSED_VAR (itf.rsp_data) \
    assign itf.rsp_ready = 0;

`define UNUSED_VX_MEM_BUS_IF(itf) \
    `UNUSED_VAR (itf.req_valid) \
    `UNUSED_VAR (itf.req_data) \
    assign itf.req_ready = 0; \
    assign itf.rsp_valid = 0; \
    assign itf.rsp_data  = '0; \
    `UNUSED_VAR (itf.rsp_ready)


`define BUFFER_DCR_BUS_IF(dst, src, ena, latency) \
    /* verilator lint_off GENUNNAMED */ \
    if (latency != 0) begin \
        VX_pipe_register #( \
            .DATAW  (1 + `VX_DCR_ADDR_WIDTH + `VX_DCR_DATA_WIDTH), \
            .DEPTH  (latency) \
        ) pipe_reg ( \
            .clk      (clk), \
            .reset    (1'b0), \
            .enable   (1'b1), \
            .data_in  ({src.write_valid && ena, src.write_addr, src.write_data}), \
            .data_out ({dst.write_valid, dst.write_addr, dst.write_data}) \
        ); \
    end else begin \
        assign {dst.write_valid, dst.write_addr, dst.write_data} = {src.write_valid && ena, src.write_addr, src.write_data}; \
    end \
    /* verilator lint_on GENUNNAMED */

`define PERF_COUNTER_ADD(dst, src, field, width, count, reg_enable) \
    /* verilator lint_off GENUNNAMED */ \
    if (count > 1) begin \
        wire [count-1:0][width-1:0] __reduce_add_i_field; \
        wire [width-1:0] __reduce_add_o_field; \
        for (genvar __i = 0; __i < count; ++__i) begin \
            assign __reduce_add_i_field[__i] = src[__i].``field; \
        end \
        VX_reduce #(.DATAW_IN(width), .N(count), .OP("+")) __reduce_add_field ( \
            __reduce_add_i_field, \
            __reduce_add_o_field \
        ); \
        if (reg_enable) begin \
            reg [width-1:0] __reduce_add_r_field; \
            always @(posedge clk) begin \
                if (reset) begin \
                    __reduce_add_r_field <= '0; \
                end else begin \
                    __reduce_add_r_field <= __reduce_add_o_field; \
                end \
            end \
            assign dst.``field = __reduce_add_r_field; \
        end else begin \
            assign dst.``field = __reduce_add_o_field; \
        end \
    end else begin \
        assign dst.``field = src[0].``field; \
    end \
    /* verilator lint_on GENUNNAMED */

`define ASSIGN_BLOCKED_WID(dst, src, block_idx, block_size) \
    /* verilator lint_off GENUNNAMED */ \
    if (block_size != 1) begin \
        if (block_size != `NUM_WARPS) begin \
            assign dst = {src[`NW_WIDTH-1:`CLOG2(block_size)], `CLOG2(block_size)'(block_idx)}; \
        end else begin \
            assign dst = `NW_WIDTH'(block_idx); \
        end \
    end else begin \
        assign dst = src; \
    end \
    /* verilator lint_on GENUNNAMED */

`endif // VX_DEFINE_VH
