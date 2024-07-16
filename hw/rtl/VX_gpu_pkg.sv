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

    typedef struct packed {
        logic [($bits(alu_args_t)-`VX_TEX_STAGE_BITS)-1:0] __padding;
        logic [`VX_TEX_STAGE_BITS-1:0] stage;
    } tex_args_t;

    typedef union packed {
        alu_args_t  alu;
        fpu_args_t  fpu;
        lsu_args_t  lsu;
        csr_args_t  csr;
        wctl_args_t wctl;
        tex_args_t  tex;
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
    localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`ICACHE_MSHR_SIZE, 1, `NUM_ICACHES);
    `else
    localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_TAG_WIDTH(1, ICACHE_LINE_SIZE, ICACHE_WORD_SIZE, ICACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_ICACHES);
    `endif

    ////////////////////////// Dcache Parameters //////////////////////////////

    // Word size in bytes
    localparam DCACHE_WORD_SIZE	    = `LSU_LINE_SIZE;
    localparam DCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(DCACHE_WORD_SIZE));

    // Block size in bytes
    localparam DCACHE_LINE_SIZE 	= `L1_LINE_SIZE;

    // Input request size
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
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`DCACHE_MSHR_SIZE, `DCACHE_NUM_BANKS, DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES);
`else
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_MEM_TAG_WIDTH(DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES);
`endif

    /////////////////////////////// L1 Parameters /////////////////////////////

    localparam L1_MEM_TAG_WIDTH     = `MAX(ICACHE_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH);
    localparam L1_MEM_ARB_TAG_WIDTH = (L1_MEM_TAG_WIDTH + `CLOG2(2));

    ////////////////////////// Tcache Parameters //////////////////////////////

    // Word size in bytes
    localparam TCACHE_WORD_SIZE	    = 4;
    localparam TCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(TCACHE_WORD_SIZE));

    // Block size in bytes
    localparam TCACHE_LINE_SIZE	    = `L1_LINE_SIZE;

    // Input request size
    localparam TCACHE_NUM_REQS	    = `TCACHE_NUM_BANKS;

    // Memory request size
    localparam TEX_MEM_REQS	        = (4 * `NUM_SFU_LANES);

    // Batch select bits
    localparam TCACHE_BATCH_SEL_BITS =`ARB_SEL_BITS(TEX_MEM_REQS, TCACHE_NUM_REQS);

    // Core request tag Id bits
    localparam TCACHE_TAG_ID_BITS	= (`CLOG2(`TEX_MEM_QUEUE_SIZE) + TCACHE_BATCH_SEL_BITS);

    // Core request tag bits
    localparam TCACHE_TAG_WIDTH	    = (`UUID_WIDTH + TCACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam TCACHE_MEM_DATA_WIDTH = (TCACHE_LINE_SIZE * 8);

    // Memory request tag bits
    `ifdef TCACHE_ENABLE
    localparam TCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`TCACHE_MSHR_SIZE, `TCACHE_NUM_BANKS, `NUM_TCACHES);
    `else
    localparam TCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_TAG_WIDTH(TCACHE_NUM_REQS, TCACHE_LINE_SIZE, TCACHE_WORD_SIZE, TCACHE_TAG_WIDTH, `NUM_TEX_UNITS, `NUM_TCACHES);
    `endif

    ////////////////////////// Rcache Parameters //////////////////////////////

    // Word size in bytes
    localparam RCACHE_WORD_SIZE	    = 4;
    localparam RCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(RCACHE_WORD_SIZE));

    // Block size in bytes
    localparam RCACHE_LINE_SIZE	    = `L1_LINE_SIZE;

    // Input request size
    localparam RCACHE_NUM_REQS	    = `RCACHE_NUM_BANKS;

    // Raster memory request size
    localparam RASTER_MEM_REQS	    = 9;

    // Batch select bits
    localparam RCACHE_BATCH_SEL_BITS = `ARB_SEL_BITS(RASTER_MEM_REQS, RCACHE_NUM_REQS);

    // Core request tag Id bits
    localparam RCACHE_TAG_ID_BITS	= (`CLOG2(`RASTER_MEM_QUEUE_SIZE) + RCACHE_BATCH_SEL_BITS);

    // Core request tag bits
    localparam RCACHE_TAG_WIDTH	    = RCACHE_TAG_ID_BITS;

    // Memory request data bits
    localparam RCACHE_MEM_DATA_WIDTH= (RCACHE_LINE_SIZE * 8);

    // Memory request tag bits
    `ifdef RCACHE_ENABLE
    localparam RCACHE_MEM_TAG_WIDTH	= `CACHE_CLUSTER_MEM_TAG_WIDTH(`RCACHE_MSHR_SIZE, `RCACHE_NUM_BANKS, `NUM_RCACHES);
    `else
    localparam RCACHE_MEM_TAG_WIDTH	= `CACHE_CLUSTER_BYPASS_TAG_WIDTH(RCACHE_NUM_REQS, RCACHE_LINE_SIZE, RCACHE_WORD_SIZE, RCACHE_TAG_WIDTH, `NUM_RASTER_UNITS, `NUM_RCACHES);
    `endif

    ////////////////////////// Ocache Parameters //////////////////////////////

    // Word size in bytes
    localparam OCACHE_WORD_SIZE	    = 4;
    localparam OCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(OCACHE_WORD_SIZE));

    // Block size in bytes
    localparam OCACHE_LINE_SIZE	    = `L1_LINE_SIZE;

    // Input request size
    localparam OCACHE_NUM_REQS	    = `OCACHE_NUM_BANKS;

    // OM memory request size
    localparam OM_MEM_REQS	        = (2 * `NUM_SFU_LANES);

    // Batch select bits
    localparam OCACHE_BATCH_SEL_BITS = `ARB_SEL_BITS(OM_MEM_REQS, OCACHE_NUM_REQS);

    // Core request tag Id bits
    localparam OCACHE_TAG_ID_BITS	= (`CLOG2(`OM_MEM_QUEUE_SIZE) + OCACHE_BATCH_SEL_BITS);

    // Core request tag bits
    localparam OCACHE_TAG_WIDTH	    = (`UUID_WIDTH + OCACHE_TAG_ID_BITS);

    // Memory request data bits
    localparam OCACHE_MEM_DATA_WIDTH = (OCACHE_LINE_SIZE * 8);

    // Memory request tag bits
    `ifdef OCACHE_ENABLE
    localparam OCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`OCACHE_MSHR_SIZE, `OCACHE_NUM_BANKS, `NUM_OCACHES);
    `else
    localparam OCACHE_MEM_TAG_WIDTH	= `CACHE_CLUSTER_BYPASS_TAG_WIDTH(OCACHE_NUM_REQS, OCACHE_LINE_SIZE, OCACHE_WORD_SIZE, OCACHE_TAG_WIDTH, `NUM_OM_UNITS, `NUM_OCACHES);
    `endif

    /////////////////////////////// L2 Parameters /////////////////////////////

    localparam L1_MEM_L2_IDX        = 0;
    localparam TCACHE_MEM_L2_IDX    = L1_MEM_L2_IDX + `NUM_SOCKETS;
    localparam RCACHE_MEM_L2_IDX    = TCACHE_MEM_L2_IDX + `EXT_TEX_ENABLED;
    localparam OCACHE_MEM_L2_IDX    = RCACHE_MEM_L2_IDX + `EXT_RASTER_ENABLED;

    // Word size in bytes
    localparam L2_WORD_SIZE	        = `L1_LINE_SIZE;

    // Input request size
    localparam L2_NUM_REQS	        = (`NUM_SOCKETS + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_OM_ENABLED);

    // Core request tag bits
    localparam L2_TAG_WIDTH	        = `MAX(`MAX(`MAX(L1_MEM_ARB_TAG_WIDTH,
                                      (`EXT_TEX_ENABLED ? TCACHE_MEM_TAG_WIDTH : 0)),
                                      (`EXT_RASTER_ENABLED ? RCACHE_MEM_TAG_WIDTH : 0)),
                                      (`EXT_OM_ENABLED ? OCACHE_MEM_TAG_WIDTH : 0));

    // Memory request data bits
    localparam L2_MEM_DATA_WIDTH	= (`L2_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef L2_ENABLE
    localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L2_MSHR_SIZE, `L2_NUM_BANKS, L2_NUM_REQS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
`else
    localparam L2_MEM_TAG_WIDTH     = `CACHE_BYPASS_TAG_WIDTH(L2_NUM_REQS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
`endif

    /////////////////////////////// L3 Parameters /////////////////////////////

    // Word size in bytes
    localparam L3_WORD_SIZE	        = `L2_LINE_SIZE;

    // Input request size
    localparam L3_NUM_REQS	        = `NUM_CLUSTERS;

    // Core request tag bits
    localparam L3_TAG_WIDTH	        = L2_MEM_TAG_WIDTH;

    // Memory request data bits
    localparam L3_MEM_DATA_WIDTH	= (`L3_LINE_SIZE * 8);

    // Memory request tag bits
`ifdef L3_ENABLE
    localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L3_MSHR_SIZE, `L3_NUM_BANKS, L3_NUM_REQS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
`else
    localparam L3_MEM_TAG_WIDTH     = `CACHE_BYPASS_TAG_WIDTH(L3_NUM_REQS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
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
        `ifdef EXT_TEX_ENABLE
            `INST_SFU_TEX: op_to_sfu_type = `SFU_TEX;
        `endif
        `ifdef EXT_OM_ENABLE
            `INST_SFU_OM: op_to_sfu_type = `SFU_OM;
        `endif
        `ifdef EXT_RASTER_ENABLE
            `INST_SFU_RASTER: op_to_sfu_type = `SFU_RASTER;
        `endif
            default: op_to_sfu_type = `SFU_WCTL;
        endcase
    endfunction

`IGNORE_UNUSED_END

endpackage

`endif // VX_GPU_PKG_VH
