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
        logic [`XLEN-1:0]       pc;
    } wspawn_t;

    typedef struct packed {
        logic                    valid;
        logic                    is_dvg;
        logic [`NUM_THREADS-1:0] then_tmask;
        logic [`NUM_THREADS-1:0] else_tmask;
        logic [`XLEN-1:0]        next_pc;
    } split_t;

    typedef struct packed {
        logic valid;
        logic is_dvg;
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
    } barrier_t;

    typedef struct packed {
        logic [`XLEN-1:0]   startup_addr;
        logic [7:0]         mpm_class;
    } base_dcrs_t;

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

    /* verilator lint_off UNUSED */

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
    localparam DCACHE_WORD_SIZE	    = (`XLEN / 8);
    localparam DCACHE_ADDR_WIDTH	= (`MEM_ADDR_WIDTH - `CLOG2(DCACHE_WORD_SIZE));

    // Block size in bytes
    localparam DCACHE_LINE_SIZE 	= `L1_LINE_SIZE;

    // Input request size
    localparam DCACHE_NUM_REQS	    = `MAX(`DCACHE_NUM_BANKS, `SMEM_NUM_BANKS);

    // Memory request size
    localparam LSU_MEM_REQS	        = `NUM_LSU_LANES;

    // Batch select bits
    localparam DCACHE_NUM_BATCHES	= ((LSU_MEM_REQS + DCACHE_NUM_REQS - 1) / DCACHE_NUM_REQS);
    localparam DCACHE_BATCH_SEL_BITS = `CLOG2(DCACHE_NUM_BATCHES);

    // Core request tag Id bits
    localparam LSUQ_TAG_BITS	    = (`CLOG2(`LSUQ_SIZE) + DCACHE_BATCH_SEL_BITS);
    localparam DCACHE_TAG_ID_BITS	= (LSUQ_TAG_BITS + `CACHE_ADDR_TYPE_BITS);

    // Core request tag bits
    localparam DCACHE_TAG_WIDTH	    = (`UUID_WIDTH + DCACHE_TAG_ID_BITS);
    localparam DCACHE_NOSM_TAG_WIDTH = (DCACHE_TAG_WIDTH - `SM_ENABLED);
    
    // Memory request data bits
    localparam DCACHE_MEM_DATA_WIDTH = (DCACHE_LINE_SIZE * 8);

    // Memory request tag bits
    `ifdef DCACHE_ENABLE
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`DCACHE_MSHR_SIZE, `DCACHE_NUM_BANKS, DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_NOSM_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES);
    `else
    localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_BYPASS_TAG_WIDTH(DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_NOSM_TAG_WIDTH, `SOCKET_SIZE, `NUM_DCACHES);
    `endif

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

    /////////////////////////////// L1 Parameters /////////////////////////////

    localparam NUM_L1_OUTPUTS           = (2 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_OM_ENABLED);
    localparam ICACHE_MEM_ARB_TAG_WIDTH = (ICACHE_MEM_TAG_WIDTH + `CLOG2(`NUM_SOCKETS));
    localparam DCACHE_MEM_ARB_TAG_WIDTH = (DCACHE_MEM_TAG_WIDTH + `CLOG2(`NUM_SOCKETS));
    localparam L1_MEM_TAG_WIDTH         = `MAX(`MAX(`MAX(`MAX(ICACHE_MEM_ARB_TAG_WIDTH, DCACHE_MEM_ARB_TAG_WIDTH),
                                          (`EXT_TEX_ENABLED ? TCACHE_MEM_TAG_WIDTH : 0)),
                                          (`EXT_RASTER_ENABLED ? RCACHE_MEM_TAG_WIDTH : 0)),
                                          (`EXT_OM_ENABLED ? OCACHE_MEM_TAG_WIDTH : 0));

    /////////////////////////////// L2 Parameters /////////////////////////////

    localparam ICACHE_MEM_ARB_IDX = 0;
    localparam DCACHE_MEM_ARB_IDX = ICACHE_MEM_ARB_IDX + 1;
    localparam TCACHE_MEM_ARB_IDX = DCACHE_MEM_ARB_IDX + 1;
    localparam RCACHE_MEM_ARB_IDX = TCACHE_MEM_ARB_IDX + `EXT_TEX_ENABLED;
    localparam OCACHE_MEM_ARB_IDX = RCACHE_MEM_ARB_IDX + `EXT_RASTER_ENABLED;

    // Word size in bytes
    localparam L2_WORD_SIZE	        = `L1_LINE_SIZE;

    // Input request size
    localparam L2_NUM_REQS	        = (2 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_OM_ENABLED);

    // Core request tag bits
    localparam L2_TAG_WIDTH	        = L1_MEM_TAG_WIDTH;

    // Memory request data bits
    localparam L2_MEM_DATA_WIDTH	= (`L2_LINE_SIZE * 8);

    // Memory request tag bits
    `ifdef L2_ENABLE
    localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L2_MSHR_SIZE, `L2_NUM_BANKS, L2_NUM_REQS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
    `else
    localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_BYPASS_TAG_WIDTH(L2_NUM_REQS, `L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
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
    localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_BYPASS_TAG_WIDTH(L3_NUM_REQS, `L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
    `endif

    /* verilator lint_on UNUSED */

    /////////////////////////////// Issue parameters //////////////////////////

    localparam ISSUE_ISW   = `CLOG2(`ISSUE_WIDTH);
    localparam ISSUE_ISW_W = `UP(ISSUE_ISW);   
    localparam ISSUE_RATIO = `NUM_WARPS / `ISSUE_WIDTH;
    localparam ISSUE_WIS   = `CLOG2(ISSUE_RATIO);
    localparam ISSUE_WIS_W = `UP(ISSUE_WIS);
    
`IGNORE_UNUSED_BEGIN
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
`IGNORE_UNUSED_END

endpackage

`endif // VX_GPU_PKG_VH
