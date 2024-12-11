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

package VX_gpu_pkg;

    typedef struct packed {
        logic                    valid;
        logic [`NUM_THREADS-1:0] tmask;
    } tmc_t;

    typedef struct packed {
        logic                   valid;
        logic [`NUM_WARPS-1:0]  wmask;
        logic [`PC_BITS-1:0]    pc;
    } wspawn_t;

    typedef struct packed {
        logic                    valid;
        logic                    is_dvg;
        logic [`NUM_THREADS-1:0] then_tmask;
        logic [`NUM_THREADS-1:0] else_tmask;
        logic [`PC_BITS-1:0]     next_pc;
    } split_t;

    typedef struct packed {
        logic valid;
        logic [`DV_STACK_SIZEW-1:0] stack_ptr;
    } join_t;

    typedef struct packed {
        logic                   valid;
        logic [`NB_WIDTH-1:0]   id;
        logic                   is_global;
    `ifdef GBAR_ENABLE
        logic [`MAX(`NW_WIDTH, `NC_WIDTH)-1:0] size_m1;
    `else
        logic [`NW_WIDTH-1:0]   size_m1;
    `endif
        logic                   is_noop;
    } barrier_t;

    typedef struct packed {
        logic [`XLEN-1:0]       startup_addr;
        logic [`XLEN-1:0]       startup_arg;
        logic [7:0]             mpm_class;
    } base_dcrs_t;

    //////////////////////////// Perf counter types ///////////////////////////

    typedef struct packed {
        logic [`PERF_CTR_BITS-1:0] reads;
        logic [`PERF_CTR_BITS-1:0] writes;
        logic [`PERF_CTR_BITS-1:0] read_misses;
        logic [`PERF_CTR_BITS-1:0] write_misses;
        logic [`PERF_CTR_BITS-1:0] bank_stalls;
        logic [`PERF_CTR_BITS-1:0] mshr_stalls;
        logic [`PERF_CTR_BITS-1:0] mem_stalls;
        logic [`PERF_CTR_BITS-1:0] crsp_stalls;
    } cache_perf_t;

    typedef struct packed {
        logic [`PERF_CTR_BITS-1:0] reads;
        logic [`PERF_CTR_BITS-1:0] writes;
        logic [`PERF_CTR_BITS-1:0] latency;
    } mem_perf_t;

    typedef struct packed {
        logic [`PERF_CTR_BITS-1:0] idles;
        logic [`PERF_CTR_BITS-1:0] stalls;
    } sched_perf_t;

    typedef struct packed {
        logic [`PERF_CTR_BITS-1:0] ibf_stalls;
        logic [`PERF_CTR_BITS-1:0] scb_stalls;
        logic [`PERF_CTR_BITS-1:0] opd_stalls;
        logic [`NUM_EX_UNITS-1:0][`PERF_CTR_BITS-1:0] units_uses;
        logic [`NUM_SFU_UNITS-1:0][`PERF_CTR_BITS-1:0] sfu_uses;
    } issue_perf_t;

    //////////////////////// instruction arguments ////////////////////////////

    typedef struct packed {
        logic use_PC;
        logic use_imm;
        logic is_w;
        logic [`ALU_TYPE_BITS-1:0] xtype;
        logic [`IMM_BITS-1:0] imm;
    } alu_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-`INST_FRM_BITS-`INST_FMT_BITS)-1:0] __padding;
        logic [`INST_FRM_BITS-1:0] frm;
        logic [`INST_FMT_BITS-1:0] fmt;
    } fpu_args_t;

    typedef struct packed {
        logic [($bits(alu_args_t)-1-1-`OFFSET_BITS)-1:0] __padding;
        logic is_store;
        logic is_float;
        logic [`OFFSET_BITS-1:0] offset;
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

`IGNORE_UNUSED_BEGIN

    ///////////////////////// LSU memory Parameters ///////////////////////////

    localparam LSU_WORD_SIZE        = `XLEN / 8;
    localparam LSU_ADDR_WIDTH	    = (`MEM_ADDR_WIDTH - `CLOG2(LSU_WORD_SIZE));
    localparam LSU_MEM_BATCHES      = 1;
    localparam LSU_TAG_ID_BITS      = (`CLOG2(`LSUQ_IN_SIZE) + `CLOG2(LSU_MEM_BATCHES));
    localparam LSU_TAG_WIDTH        = (`UUID_WIDTH + LSU_TAG_ID_BITS);
    localparam LSU_NUM_REQS	        = `NUM_LSU_BLOCKS * `NUM_LSU_LANES;

    ////////////////////////// Icache Parameters //////////////////////////////

    // Word size in bytes
    localparam ICACHE_WORD_SIZE	    = 4;
    localparam ICACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(ICACHE_WORD_SIZE));

    // Block size in bytes
    localparam ICACHE_LINE_SIZE	    = `L1_LINE_SIZE;

    // Core request tag Id bits
    localparam ICACHE_TAG_ID_BITS	= `NW_WIDTH;

    // Core request tag bits
    localparam ICACHE_TAG_WIDTH	    = (`UUID_WIDTH + ICACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam ICACHE_MEM_DATA_WIDTH = (ICACHE_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef ICACHE_ENABLE
    localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`ICACHE_MSHR_SIZE, 1, 1, `NUM_ICACHES, `UUID_WIDTH);
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
    localparam DCACHE_TAG_WIDTH	    = (`UUID_WIDTH + DCACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam DCACHE_MEM_DATA_WIDTH = (DCACHE_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef DCACHE_ENABLE
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`DCACHE_MSHR_SIZE, `DCACHE_NUM_BANKS, DCACHE_NUM_REQS, `L1_MEM_PORTS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES, `UUID_WIDTH);
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
    localparam L2_NUM_REQS	        = `NUM_SOCKETS * `L1_MEM_PORTS;

    // Core request tag bits
    localparam L2_TAG_WIDTH	        = L1_MEM_ARB_TAG_WIDTH;

    // Memory request data bits
    localparam L2_MEM_DATA_WIDTH	= (`L2_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef L2_ENABLE
    localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L2_MSHR_SIZE, `L2_NUM_BANKS, L2_NUM_REQS, `L2_MEM_PORTS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH, `UUID_WIDTH);
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
    localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L3_MSHR_SIZE, `L3_NUM_BANKS, L3_NUM_REQS, `L3_MEM_PORTS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH, `UUID_WIDTH);
`else
    localparam L3_MEM_TAG_WIDTH     = `CACHE_BYPASS_TAG_WIDTH(L3_NUM_REQS, `L3_MEM_PORTS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
`endif

    /////////////////////////////// Issue parameters //////////////////////////

    localparam ISSUE_ISW   = `CLOG2(`ISSUE_WIDTH);
    localparam ISSUE_ISW_W = `UP(ISSUE_ISW);
    localparam PER_ISSUE_WARPS = `NUM_WARPS / `ISSUE_WIDTH;
    localparam ISSUE_WIS   = `CLOG2(PER_ISSUE_WARPS);
    localparam ISSUE_WIS_W = `UP(ISSUE_WIS);

    function logic [`NW_WIDTH-1:0] wis_to_wid(
        input logic [ISSUE_WIS_W-1:0] wis,
        input logic [ISSUE_ISW_W-1:0] isw
    );
        if (ISSUE_WIS == 0) begin
            wis_to_wid = `NW_WIDTH'(isw);
        end else if (ISSUE_ISW == 0) begin
            wis_to_wid = `NW_WIDTH'(wis);
        end else begin
            wis_to_wid = `NW_WIDTH'({wis, isw});
        end
    endfunction

    function logic [ISSUE_ISW_W-1:0] wid_to_isw(
        input logic [`NW_WIDTH-1:0] wid
    );
        if (ISSUE_ISW != 0) begin
            wid_to_isw = wid[ISSUE_ISW_W-1:0];
        end else begin
            wid_to_isw = 0;
        end
    endfunction

    function logic [ISSUE_WIS_W-1:0] wid_to_wis(
        input logic [`NW_WIDTH-1:0] wid
    );
        if (ISSUE_WIS != 0) begin
            wid_to_wis = ISSUE_WIS_W'(wid >> ISSUE_ISW);
        end else begin
            wid_to_wis = 0;
        end
    endfunction

    ///////////////////////// Miscaellaneous functions ////////////////////////

    function logic [`SFU_WIDTH-1:0] op_to_sfu_type(
        input logic [`INST_OP_BITS-1:0] op_type
    );
        case (op_type)
        `INST_SFU_CSRRW,
        `INST_SFU_CSRRS,
        `INST_SFU_CSRRC: op_to_sfu_type = `SFU_CSRS;
        default: op_to_sfu_type = `SFU_WCTL;
        endcase
    endfunction

`IGNORE_UNUSED_END

////////////////////////////////// Tracing ////////////////////////////////////

`ifdef SIMULATION

`ifdef SV_DPI
    import "DPI-C" function void dpi_trace(input int level, input string format /*verilator sformat*/);
`endif

    task trace_ex_type(input int level, input [`EX_BITS-1:0] ex_type);
        case (ex_type)
            `EX_ALU: `TRACE(level, ("ALU"))
            `EX_LSU: `TRACE(level, ("LSU"))
            `EX_SFU: `TRACE(level, ("SFU"))
        `ifdef EXT_F_ENABLE
            `EX_FPU: `TRACE(level, ("FPU"))
        `endif
            default: `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [`EX_BITS-1:0] ex_type,
                     input [`INST_OP_BITS-1:0] op_type,
                     input VX_gpu_pkg::op_args_t op_args
    );
        case (ex_type)
        `EX_ALU: begin
            case (op_args.alu.xtype)
                `ALU_TYPE_ARITH: begin
                    if (op_args.alu.is_w) begin
                        if (op_args.alu.use_imm) begin
                            case (`INST_ALU_BITS'(op_type))
                                `INST_ALU_ADD: `TRACE(level, ("ADDIW"))
                                `INST_ALU_SLL: `TRACE(level, ("SLLIW"))
                                `INST_ALU_SRL: `TRACE(level, ("SRLIW"))
                                `INST_ALU_SRA: `TRACE(level, ("SRAIW"))
                                default:       `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (`INST_ALU_BITS'(op_type))
                                `INST_ALU_ADD: `TRACE(level, ("ADDW"))
                                `INST_ALU_SUB: `TRACE(level, ("SUBW"))
                                `INST_ALU_SLL: `TRACE(level, ("SLLW"))
                                `INST_ALU_SRL: `TRACE(level, ("SRLW"))
                                `INST_ALU_SRA: `TRACE(level, ("SRAW"))
                                default:       `TRACE(level, ("?"))
                            endcase
                        end
                    end else begin
                        if (op_args.alu.use_imm) begin
                            case (`INST_ALU_BITS'(op_type))
                                `INST_ALU_ADD:   `TRACE(level, ("ADDI"))
                                `INST_ALU_SLL:   `TRACE(level, ("SLLI"))
                                `INST_ALU_SRL:   `TRACE(level, ("SRLI"))
                                `INST_ALU_SRA:   `TRACE(level, ("SRAI"))
                                `INST_ALU_SLT:   `TRACE(level, ("SLTI"))
                                `INST_ALU_SLTU:  `TRACE(level, ("SLTIU"))
                                `INST_ALU_XOR:   `TRACE(level, ("XORI"))
                                `INST_ALU_OR:    `TRACE(level, ("ORI"))
                                `INST_ALU_AND:   `TRACE(level, ("ANDI"))
                                `INST_ALU_LUI:   `TRACE(level, ("LUI"))
                                `INST_ALU_AUIPC: `TRACE(level, ("AUIPC"))
                                default:         `TRACE(level, ("?"))
                            endcase
                        end else begin
                            case (`INST_ALU_BITS'(op_type))
                                `INST_ALU_ADD:   `TRACE(level, ("ADD"))
                                `INST_ALU_SUB:   `TRACE(level, ("SUB"))
                                `INST_ALU_SLL:   `TRACE(level, ("SLL"))
                                `INST_ALU_SRL:   `TRACE(level, ("SRL"))
                                `INST_ALU_SRA:   `TRACE(level, ("SRA"))
                                `INST_ALU_SLT:   `TRACE(level, ("SLT"))
                                `INST_ALU_SLTU:  `TRACE(level, ("SLTU"))
                                `INST_ALU_XOR:   `TRACE(level, ("XOR"))
                                `INST_ALU_OR:    `TRACE(level, ("OR"))
                                `INST_ALU_AND:   `TRACE(level, ("AND"))
                                `INST_ALU_CZEQ:  `TRACE(level, ("CZERO.EQZ"))
                                `INST_ALU_CZNE:  `TRACE(level, ("CZERO.NEZ"))
                                default:         `TRACE(level, ("?"))
                            endcase
                        end
                    end
                end
                `ALU_TYPE_BRANCH: begin
                    case (`INST_BR_BITS'(op_type))
                        `INST_BR_EQ:    `TRACE(level, ("BEQ"))
                        `INST_BR_NE:    `TRACE(level, ("BNE"))
                        `INST_BR_LT:    `TRACE(level, ("BLT"))
                        `INST_BR_GE:    `TRACE(level, ("BGE"))
                        `INST_BR_LTU:   `TRACE(level, ("BLTU"))
                        `INST_BR_GEU:   `TRACE(level, ("BGEU"))
                        `INST_BR_JAL:   `TRACE(level, ("JAL"))
                        `INST_BR_JALR:  `TRACE(level, ("JALR"))
                        `INST_BR_ECALL: `TRACE(level, ("ECALL"))
                        `INST_BR_EBREAK:`TRACE(level, ("EBREAK"))
                        `INST_BR_URET:  `TRACE(level, ("URET"))
                        `INST_BR_SRET:  `TRACE(level, ("SRET"))
                        `INST_BR_MRET:  `TRACE(level, ("MRET"))
                        default:        `TRACE(level, ("?"))
                    endcase
                end
                `ALU_TYPE_MULDIV: begin
                    if (op_args.alu.is_w) begin
                        case (`INST_M_BITS'(op_type))
                            `INST_M_MUL:  `TRACE(level, ("MULW"))
                            `INST_M_DIV:  `TRACE(level, ("DIVW"))
                            `INST_M_DIVU: `TRACE(level, ("DIVUW"))
                            `INST_M_REM:  `TRACE(level, ("REMW"))
                            `INST_M_REMU: `TRACE(level, ("REMUW"))
                            default:      `TRACE(level, ("?"))
                        endcase
                    end else begin
                        case (`INST_M_BITS'(op_type))
                            `INST_M_MUL:   `TRACE(level, ("MUL"))
                            `INST_M_MULH:  `TRACE(level, ("MULH"))
                            `INST_M_MULHSU:`TRACE(level, ("MULHSU"))
                            `INST_M_MULHU: `TRACE(level, ("MULHU"))
                            `INST_M_DIV:   `TRACE(level, ("DIV"))
                            `INST_M_DIVU:  `TRACE(level, ("DIVU"))
                            `INST_M_REM:   `TRACE(level, ("REM"))
                            `INST_M_REMU:  `TRACE(level, ("REMU"))
                            default:       `TRACE(level, ("?"))
                        endcase
                    end
                end
                default: `TRACE(level, ("?"))
            endcase
        end
        `EX_LSU: begin
            if (op_args.lsu.is_float) begin
                case (`INST_LSU_BITS'(op_type))
                    `INST_LSU_LW: `TRACE(level, ("FLW"))
                    `INST_LSU_LD: `TRACE(level, ("FLD"))
                    `INST_LSU_SW: `TRACE(level, ("FSW"))
                    `INST_LSU_SD: `TRACE(level, ("FSD"))
                    default:      `TRACE(level, ("?"))
                endcase
            end else begin
                case (`INST_LSU_BITS'(op_type))
                    `INST_LSU_LB: `TRACE(level, ("LB"))
                    `INST_LSU_LH: `TRACE(level, ("LH"))
                    `INST_LSU_LW: `TRACE(level, ("LW"))
                    `INST_LSU_LD: `TRACE(level, ("LD"))
                    `INST_LSU_LBU:`TRACE(level, ("LBU"))
                    `INST_LSU_LHU:`TRACE(level, ("LHU"))
                    `INST_LSU_LWU:`TRACE(level, ("LWU"))
                    `INST_LSU_SB: `TRACE(level, ("SB"))
                    `INST_LSU_SH: `TRACE(level, ("SH"))
                    `INST_LSU_SW: `TRACE(level, ("SW"))
                    `INST_LSU_SD: `TRACE(level, ("SD"))
                    `INST_LSU_FENCE:`TRACE(level,("FENCE"))
                    default:      `TRACE(level, ("?"))
                endcase
            end
        end
        `EX_SFU: begin
            case (`INST_SFU_BITS'(op_type))
                `INST_SFU_TMC:   `TRACE(level, ("TMC"))
                `INST_SFU_WSPAWN:`TRACE(level, ("WSPAWN"))
                `INST_SFU_SPLIT: begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("SPLIT.N"))
                    end else begin
                        `TRACE(level, ("SPLIT"))
                    end
                end
                `INST_SFU_JOIN:  `TRACE(level, ("JOIN"))
                `INST_SFU_BAR:   `TRACE(level, ("BAR"))
                `INST_SFU_PRED:  begin
                    if (op_args.wctl.is_neg) begin
                        `TRACE(level, ("PRED.N"))
                    end else begin
                        `TRACE(level, ("PRED"))
                    end
                end
                `INST_SFU_CSRRW: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRWI"))
                    end else begin
                        `TRACE(level, ("CSRRW"))
                    end
                end
                `INST_SFU_CSRRS: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRSI"))
                    end else begin
                        `TRACE(level, ("CSRRS"))
                    end
                end
                `INST_SFU_CSRRC: begin
                    if (op_args.csr.use_imm) begin
                        `TRACE(level, ("CSRRCI"))
                    end else begin
                        `TRACE(level, ("CSRRC"))
                    end
                end
                default:         `TRACE(level, ("?"))
            endcase
        end
    `ifdef EXT_F_ENABLE
        `EX_FPU: begin
            case (`INST_FPU_BITS'(op_type))
                `INST_FPU_ADD: begin
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
                `INST_FPU_MADD: begin
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
                `INST_FPU_NMADD: begin
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
                `INST_FPU_MUL: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FMUL.D"))
                    end else begin
                        `TRACE(level, ("FMUL.S"))
                        end
                end
                `INST_FPU_DIV: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FDIV.D"))
                    end else begin
                        `TRACE(level, ("FDIV.S"))
                        end
                end
                `INST_FPU_SQRT: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FSQRT.D"))
                    end else begin
                        `TRACE(level, ("FSQRT.S"))
                    end
                end
                `INST_FPU_CMP: begin
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
                `INST_FPU_F2F: begin
                    if (op_args.fpu.fmt[0]) begin
                        `TRACE(level, ("FCVT.D.S"))
                    end else begin
                        `TRACE(level, ("FCVT.S.D"))
                    end
                end
                `INST_FPU_F2I: begin
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
                `INST_FPU_F2U: begin
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
                `INST_FPU_I2F: begin
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
                `INST_FPU_U2F: begin
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
                `INST_FPU_MISC: begin
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
                       input [`EX_BITS-1:0] ex_type,
                       input [`INST_OP_BITS-1:0] op_type,
                       input VX_gpu_pkg::op_args_t op_args
    );
        case (ex_type)
        `EX_ALU: begin
            `TRACE(level, (", use_PC=%b, use_imm=%b, imm=0x%0h", op_args.alu.use_PC, op_args.alu.use_imm, op_args.alu.imm))
        end
        `EX_LSU: begin
            `TRACE(level, (", offset=0x%0h", op_args.lsu.offset))
        end
        `EX_SFU: begin
            if (`INST_SFU_IS_CSR(op_type)) begin
                `TRACE(level, (", addr=0x%0h, use_imm=%b, imm=0x%0h", op_args.csr.addr, op_args.csr.use_imm, op_args.csr.imm))
            end
        end
    `ifdef EXT_F_ENABLE
        `EX_FPU: begin
            `TRACE(level, (", fmt=0x%0h, frm=0x%0h", op_args.fpu.fmt, op_args.fpu.frm))
        end
    `endif
        default:;
        endcase
    endtask

    task trace_base_dcr(input int level, input [`VX_DCR_ADDR_WIDTH-1:0] addr);
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

`endif // VX_GPU_PKG_VH
