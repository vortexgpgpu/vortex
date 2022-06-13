`ifndef VX_CACHE_TYPES_VH
`define VX_CACHE_TYPES_VH

`include "VX_cache_define.vh"

package VX_cache_types;

////////////////////////// Icache Definitions /////////////////////////////////

// Word size in bytes
localparam ICACHE_WORD_SIZE	    = 4;
localparam ICACHE_ADDR_WIDTH	= (32 - `CLOG2(ICACHE_WORD_SIZE));

// Block size in bytes
localparam ICACHE_LINE_SIZE	    = `L1_BLOCK_SIZE;

// Core request tag Id bits       
localparam ICACHE_TAG_ID_BITS	= `UP(`NW_BITS);

// Core request tag bits
localparam ICACHE_TAG_WIDTH	    = (`UP(`UUID_BITS) + ICACHE_TAG_ID_BITS);

// Input request size
localparam ICACHE_NUM_REQS	    = 1;

// Number of banks
localparam ICACHE_NUM_BANKS	    = 1;

// Memory request data bits
localparam ICACHE_MEM_DATA_WIDTH= (ICACHE_LINE_SIZE * 8);

// Memory request tag bits
`ifdef ICACHE_ENABLE
localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`ICACHE_MSHR_SIZE, ICACHE_NUM_BANKS, `NUM_ICACHES);
`else
localparam ICACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_TAG_WIDTH(ICACHE_NUM_REQS, ICACHE_LINE_SIZE, ICACHE_WORD_SIZE, ICACHE_ARB_TAG_WIDTH, `NUM_CORES, `NUM_ICACHES);
`endif

////////////////////////// Dcache Definitions /////////////////////////////////

// Word size in bytes
localparam DCACHE_WORD_SIZE	    = 4;
localparam DCACHE_ADDR_WIDTH	= (32 - `CLOG2(DCACHE_WORD_SIZE));

// Block size in bytes
localparam DCACHE_LINE_SIZE 	= `L1_BLOCK_SIZE;

// Input request size
localparam DCACHE_NUM_REQS	    = `UP(`NUM_THREADS / 2);

// Memory request size
localparam LSU_MEM_REQS	        = `NUM_THREADS;

// Batch select bits
localparam DCACHE_NUM_BATCHES	= ((LSU_MEM_REQS + DCACHE_NUM_REQS - 1) / DCACHE_NUM_REQS);
localparam DCACHE_BATCH_SEL_BITS= `CLOG2(DCACHE_NUM_BATCHES);

// Core request tag Id bits
localparam LSUQ_TAG_BITS	    = (`CLOG2(`LSUQ_SIZE) + DCACHE_BATCH_SEL_BITS);
localparam DCACHE_TAG_ID_BITS	= (LSUQ_TAG_BITS + `CACHE_ADDR_TYPE_BITS);

// Core request tag bits
localparam DCACHE_TAG_WIDTH	    = (`UP(`UUID_BITS) + DCACHE_TAG_ID_BITS);
 
// Memory request data bits
localparam DCACHE_MEM_DATA_WIDTH= (DCACHE_LINE_SIZE * 8);

// Memory request tag bits
localparam DCACHE_NOSM_TAG_WIDTH= (DCACHE_TAG_WIDTH - `SM_ENABLED);
`ifdef DCACHE_ENABLE
localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_MEM_TAG_WIDTH(`DCACHE_MSHR_SIZE, `DCACHE_NUM_BANKS, DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_NOSM_TAG_WIDTH, `NUM_CORES, `NUM_DCACHES);
`else
localparam DCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_NC_BYPASS_TAG_WIDTH(DCACHE_NUM_REQS, DCACHE_LINE_SIZE, DCACHE_WORD_SIZE, DCACHE_NOSM_TAG_WIDTH, `NUM_CORES, `NUM_DCACHES);
`endif

////////////////////////// Tcache Definitions /////////////////////////////////

// Word size in bytes
localparam TCACHE_WORD_SIZE	    = 4;
localparam TCACHE_ADDR_WIDTH	= (32 - `CLOG2(TCACHE_WORD_SIZE));

// Block size in bytes
localparam TCACHE_LINE_SIZE	    = `L1_BLOCK_SIZE;

// Input request size
localparam TCACHE_NUM_REQS	    = `UP(`NUM_THREADS / 2);

// Memory request size
localparam TEX_MEM_REQS	        = (4 * `NUM_THREADS);

// Batch select bits
localparam TCACHE_BATCH_SEL_BITS=`ARB_SEL_BITS(TEX_MEM_REQS, TCACHE_NUM_REQS);

// Core request tag Id bits       
localparam TCACHE_TAG_ID_BITS	= (`CLOG2(`TEX_MEM_QUEUE_SIZE) + TCACHE_BATCH_SEL_BITS);

// Core request tag bits
localparam TCACHE_TAG_WIDTH	    = TCACHE_TAG_ID_BITS;

// Memory request data bits
localparam TCACHE_MEM_DATA_WIDTH= (TCACHE_LINE_SIZE * 8);

// Memory request tag bits
`ifdef TCACHE_ENABLE
localparam TCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_MEM_TAG_WIDTH(`TCACHE_MSHR_SIZE, `TCACHE_NUM_BANKS, `NUM_TCACHES);
`else
localparam TCACHE_MEM_TAG_WIDTH = `CACHE_CLUSTER_BYPASS_TAG_WIDTH(TCACHE_NUM_REQS, TCACHE_LINE_SIZE, TCACHE_WORD_SIZE, TCACHE_TAG_WIDTH, `NUM_TEX_UNITS, `NUM_TCACHES);
`endif

////////////////////////// Rcache Definitions /////////////////////////////////

// Word size in bytes
localparam RCACHE_WORD_SIZE	    = 4;
localparam RCACHE_ADDR_WIDTH	= (32 - `CLOG2(RCACHE_WORD_SIZE));

// Block size in bytes
localparam RCACHE_LINE_SIZE	    = `L1_BLOCK_SIZE;

// Input request size
localparam RCACHE_NUM_REQS	    = `RCACHE_NUM_BANKS;
 
// Raster memory request size
localparam RASTER_MEM_REQS	    = 9;

// Batch select bits
localparam RCACHE_BATCH_SEL_BITS= `ARB_SEL_BITS(RASTER_MEM_REQS, RCACHE_NUM_REQS);

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

////////////////////////// Ocache Definitions /////////////////////////////////

// Word size in bytes
localparam OCACHE_WORD_SIZE	    = 4;
localparam OCACHE_ADDR_WIDTH	= (32 - `CLOG2(OCACHE_WORD_SIZE));

// Block size in bytes
localparam OCACHE_LINE_SIZE	    = `L1_BLOCK_SIZE;

// Input request size
localparam OCACHE_NUM_REQS	    = `OCACHE_NUM_BANKS;

// ROP memory request size
localparam ROP_MEM_REQS	        = (2 * `NUM_THREADS);

// Batch select bits
localparam OCACHE_BATCH_SEL_BITS= `ARB_SEL_BITS(ROP_MEM_REQS, OCACHE_NUM_REQS);

// Core request tag Id bits       
localparam OCACHE_TAG_ID_BITS	= (`CLOG2(`ROP_MEM_QUEUE_SIZE) + OCACHE_BATCH_SEL_BITS);

// Core request tag bits
localparam OCACHE_TAG_WIDTH	    = OCACHE_TAG_ID_BITS;

// Memory request data bits
localparam OCACHE_MEM_DATA_WIDTH= (OCACHE_LINE_SIZE * 8);

// Memory request tag bits
`ifdef OCACHE_ENABLE
localparam OCACHE_MEM_TAG_WIDTH= `CACHE_CLUSTER_MEM_TAG_WIDTH(`OCACHE_MSHR_SIZE, `OCACHE_NUM_BANKS, `NUM_OCACHES);
`else
localparam OCACHE_MEM_TAG_WIDTH	= `CACHE_CLUSTER_BYPASS_TAG_WIDTH(OCACHE_NUM_REQS, OCACHE_LINE_SIZE, OCACHE_WORD_SIZE, OCACHE_TAG_WIDTH, `NUM_ROP_UNITS, `NUM_OCACHES);
`endif

/////////////////////////////// L1 Definitions ////////////////////////////////

localparam L1_MEM_TAG_WIDTH     =  `MAX(`MAX(`MAX(`MAX(ICACHE_MEM_TAG_WIDTH, DCACHE_MEM_TAG_WIDTH),
                                    (`TCACHE_ENABLED ? TCACHE_MEM_TAG_WIDTH : 0)),
                                    (`RCACHE_ENABLED ? RCACHE_MEM_TAG_WIDTH : 0)),
                                    (`OCACHE_ENABLED ? OCACHE_MEM_TAG_WIDTH : 0));

localparam NUM_L1_OUTPUTS       = (2 + `EXT_TEX_ENABLED + `EXT_RASTER_ENABLED + `EXT_ROP_ENABLED);

/////////////////////////////// L2 Definitions ////////////////////////////////

// Word size in bytes
localparam L2_WORD_SIZE	        = `L1_BLOCK_SIZE;
localparam L2_ADDR_WIDTH	    = (32 - `CLOG2(L2_WORD_SIZE));

// Block size in bytes
`ifdef L2_ENABLE
localparam L2_LINE_SIZE	        = `MEM_BLOCK_SIZE;
`else
localparam L2_LINE_SIZE	        = L2_WORD_SIZE;
`endif

// Input request size
localparam L2_NUM_REQS	        = NUM_L1_OUTPUTS;

// Core request tag bits
localparam L2_TAG_WIDTH	        = L1_MEM_TAG_WIDTH;

// Memory request data bits
localparam L2_MEM_DATA_WIDTH	= (L2_LINE_SIZE * 8);

// Memory request tag bits
`ifdef L2_ENABLE
localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L2_MSHR_SIZE, `L2_NUM_BANKS, L2_NUM_REQS, L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
`else
localparam L2_MEM_TAG_WIDTH     = `CACHE_NC_BYPASS_TAG_WIDTH(L2_NUM_REQS, L2_LINE_SIZE, L2_WORD_SIZE, L2_TAG_WIDTH);
`endif

/////////////////////////////// L3 Definitions ////////////////////////////////

// Word size in bytes
localparam L3_WORD_SIZE	        = L2_LINE_SIZE;
localparam L3_ADDR_WIDTH	    = (32 - `CLOG2(L3_WORD_SIZE));

// Block size in bytes
`ifdef L3_ENABLE
localparam L3_LINE_SIZE	        = `MEM_BLOCK_SIZE;
`else
localparam L3_LINE_SIZE	        = L3_WORD_SIZE;
`endif

// Input request size
localparam L3_NUM_REQS	        = `NUM_CLUSTERS;

// Core request tag bits
localparam L3_TAG_WIDTH	        = L2_MEM_TAG_WIDTH;

// Memory request data bits
localparam L3_MEM_DATA_WIDTH	= (L3_LINE_SIZE * 8);

// Memory request tag bits
`ifdef L3_ENABLE
localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_MEM_TAG_WIDTH(`L3_MSHR_SIZE, `L3_NUM_BANKS, L3_NUM_REQS, L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
`else
localparam L3_MEM_TAG_WIDTH     = `CACHE_NC_BYPASS_TAG_WIDTH(L3_NUM_REQS, L3_LINE_SIZE, L3_WORD_SIZE, L3_TAG_WIDTH);
`endif

endpackage

`endif
