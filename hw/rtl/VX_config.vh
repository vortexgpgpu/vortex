`ifndef VX_CONFIG
`define VX_CONFIG

`include "VX_user_config.vh"

`ifndef NUM_CORES
`define NUM_CORES 1
`endif

`ifndef NUM_CLUSTERS
`define NUM_CLUSTERS 1
`endif

`ifndef NUM_WARPS
`define NUM_WARPS 8
`endif

`ifndef NUM_THREADS
`define NUM_THREADS 4
`endif

`ifndef NUM_BARRIERS
`define NUM_BARRIERS 4
`endif

// `define SINGLE_CORE_BENCH

`ifndef GLOBAL_BLOCK_SIZE_BYTES
`define GLOBAL_BLOCK_SIZE_BYTES 16
`endif

// ========================= Dcache Configurable Knobs ========================

// Size of cache in bytes
`ifndef DCACHE_SIZE_BYTES
`define DCACHE_SIZE_BYTES 2048
`endif

// Size of line inside a bank in bytes
`ifndef DBANK_LINE_SIZE_BYTES
`define DBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Size of line inside a bank in bits
`define DBANK_LINE_SIZE (`DBANK_LINE_SIZE_BYTES * 8)

// Number of banks {1, 2, 4, 8,...}
`ifndef DNUM_BANKS
`define DNUM_BANKS 8
`endif

// Size of a word in bytes
`ifndef DWORD_SIZE_BYTES
`define DWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef DNUM_REQUESTS
`define DNUM_REQUESTS `NUM_THREADS
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef DSTAGE_1_CYCLES
`define DSTAGE_1_CYCLES 1
`endif

// Core Request Queue Size
`ifndef DREQQ_SIZE
`define DREQQ_SIZE `NUM_WARPS
`endif

// Miss Reserv Queue Knob
`ifndef DMRVQ_SIZE
`define DMRVQ_SIZE (`NUM_WARPS*`NUM_THREADS)
`endif

// Dram Fill Rsp Queue Size
`ifndef DDFPQ_SIZE
`define DDFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef DSNRQ_SIZE
`define DSNRQ_SIZE 32
`endif

// Core Writeback Queue Size
`ifndef DCWBQ_SIZE
`define DCWBQ_SIZE `DREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef DDWBQ_SIZE
`define DDWBQ_SIZE 4
`endif

// Dram Fill Req Queue Size
`ifndef DDFQQ_SIZE
`define DDFQQ_SIZE `DREQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef DLLVQ_SIZE
`define DLLVQ_SIZE 0
`endif

// Fill Forward SNP Queue
`ifndef DFFSQ_SIZE
`define DFFSQ_SIZE 32
`endif

// Prefetcher
`ifndef DPRFQ_SIZE
`define DPRFQ_SIZE 32
`endif

`ifndef DPRFQ_STRIDE
`define DPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef DFILL_INVALIDAOR_SIZE
`define DFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef DSIMULATED_DRAM_LATENCY_CYCLES
`define DSIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ========================== Icache Configurable Knobs =======================

// Size of cache in bytes
`ifndef ICACHE_SIZE_BYTES
`define ICACHE_SIZE_BYTES 4096
`endif

// Size of line inside a bank in bytes
`ifndef IBANK_LINE_SIZE_BYTES
`define IBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef INUM_BANKS
`define INUM_BANKS 8
`endif

// Size of a word in bytes
`ifndef IWORD_SIZE_BYTES
`define IWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef INUM_REQUESTS
`define INUM_REQUESTS 1
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef ISTAGE_1_CYCLES
`define ISTAGE_1_CYCLES 1
`endif

// Core Request Queue Size
`ifndef IREQQ_SIZE
`define IREQQ_SIZE `NUM_WARPS
`endif

// Miss Reserv Queue Knob
`ifndef IMRVQ_SIZE
`define IMRVQ_SIZE `IREQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef IDFPQ_SIZE
`define IDFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef ISNRQ_SIZE
`define ISNRQ_SIZE 32
`endif

// Core Writeback Queue Size
`ifndef ICWBQ_SIZE
`define ICWBQ_SIZE `IREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef IDWBQ_SIZE
`define IDWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef IDFQQ_SIZE
`define IDFQQ_SIZE `IREQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef ILLVQ_SIZE
`define ILLVQ_SIZE 16
`endif

// Fill Forward SNP Queue
`ifndef IFFSQ_SIZE
`define IFFSQ_SIZE 8
`endif

// Prefetcher
`ifndef IPRFQ_SIZE
`define IPRFQ_SIZE 32
`endif

`ifndef IPRFQ_STRIDE
`define IPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef IFILL_INVALIDAOR_SIZE
`define IFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef ISIMULATED_DRAM_LATENCY_CYCLES
`define ISIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// =========================== SM Configurable Knobs ==========================

// Size of cache in bytes
`ifndef SCACHE_SIZE_BYTES
`define SCACHE_SIZE_BYTES 1024
`endif

// Size of line inside a bank in bytes
`ifndef SBANK_LINE_SIZE_BYTES
`define SBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef SNUM_BANKS
`define SNUM_BANKS 8
`endif

// Size of a word in bytes
`ifndef SWORD_SIZE_BYTES
`define SWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef SNUM_REQUESTS
`define SNUM_REQUESTS `NUM_THREADS
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef SSTAGE_1_CYCLES
`define SSTAGE_1_CYCLES 1
`endif

// Core Request Queue Size
`ifndef SREQQ_SIZE
`define SREQQ_SIZE `NUM_WARPS
`endif

// Miss Reserv Queue Knob
`ifndef SMRVQ_SIZE
`define SMRVQ_SIZE `SREQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef SDFPQ_SIZE
`define SDFPQ_SIZE 0
`endif

// Snoop Req Queue
`ifndef SSNRQ_SIZE
`define SSNRQ_SIZE 16
`endif

// Core Writeback Queue Size
`ifndef SCWBQ_SIZE
`define SCWBQ_SIZE `SREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef SDWBQ_SIZE
`define SDWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef SDFQQ_SIZE
`define SDFQQ_SIZE 16
`endif

// Lower Level Cache Hit Queue Size
`ifndef SLLVQ_SIZE
`define SLLVQ_SIZE 16
`endif

// Fill Forward SNP Queue
`ifndef SFFSQ_SIZE
`define SFFSQ_SIZE 16
`endif

// Prefetcher
`ifndef SPRFQ_SIZE
`define SPRFQ_SIZE 4
`endif

`ifndef SPRFQ_STRIDE
`define SPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef SFILL_INVALIDAOR_SIZE
`define SFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef SSIMULATED_DRAM_LATENCY_CYCLES
`define SSIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ======================== L2cache Configurable Knobs ========================

// Size of cache in bytes
`ifndef L2CACHE_SIZE_BYTES
`define L2CACHE_SIZE_BYTES 4096
`endif

// Size of line inside a bank in bytes
`ifndef L2BANK_LINE_SIZE_BYTES
`define L2BANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef L2NUM_BANKS
`define L2NUM_BANKS 8
`endif

// Size of a word in bytes
`ifndef L2WORD_SIZE_BYTES
`define L2WORD_SIZE_BYTES (`L2BANK_LINE_SIZE_BYTES)
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef L2NUM_REQUESTS
`define L2NUM_REQUESTS (2*`NUM_CORES_PER_CLUSTER)
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef L2STAGE_1_CYCLES
`define L2STAGE_1_CYCLES 1
`endif

// Core Request Queue Size
`ifndef L2REQQ_SIZE
`define L2REQQ_SIZE 32
`endif

// Miss Reserv Queue Knob
`ifndef L2MRVQ_SIZE
`define L2MRVQ_SIZE 32
`endif

// Dram Fill Rsp Queue Size
`ifndef L2DFPQ_SIZE
`define L2DFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef L2SNRQ_SIZE
`define L2SNRQ_SIZE 32
`endif

// Core Writeback Queue Size
`ifndef L2CWBQ_SIZE
`define L2CWBQ_SIZE `L2REQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef L2DWBQ_SIZE
`define L2DWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef L2DFQQ_SIZE
`define L2DFQQ_SIZE `L2REQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef L2LLVQ_SIZE
`define L2LLVQ_SIZE 32
`endif

// Fill Forward SNP Queue
`ifndef L2FFSQ_SIZE
`define L2FFSQ_SIZE 32
`endif

// Prefetcher
`ifndef L2PRFQ_SIZE
`define L2PRFQ_SIZE 32
`endif

`ifndef L2PRFQ_STRIDE
`define L2PRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef L2FILL_INVALIDAOR_SIZE
`define L2FILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef L2SIMULATED_DRAM_LATENCY_CYCLES
`define L2SIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ======================== L3cache Configurable Knobs ========================

// Size of cache in bytes
`ifndef L3CACHE_SIZE_BYTES
`define L3CACHE_SIZE_BYTES 8192
`endif

// Size of line inside a bank in bytes
`ifndef L3BANK_LINE_SIZE_BYTES
`define L3BANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef L3NUM_BANKS
`define L3NUM_BANKS 8
`endif

// Size of a word in bytes
`ifndef L3WORD_SIZE_BYTES
`define L3WORD_SIZE_BYTES (`L3BANK_LINE_SIZE_BYTES)
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef L3NUM_REQUESTS
`define L3NUM_REQUESTS (`NUM_CLUSTERS)
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef L3STAGE_1_CYCLES
`define L3STAGE_1_CYCLES 1
`endif

// Core Request Queue Size
`ifndef L3REQQ_SIZE
`define L3REQQ_SIZE 32
`endif

// Miss Reserv Queue Knob
`ifndef L3MRVQ_SIZE
`define L3MRVQ_SIZE `L3REQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef L3DFPQ_SIZE
`define L3DFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef L3SNRQ_SIZE
`define L3SNRQ_SIZE 32
`endif

// Core Writeback Queue Size
`ifndef L3CWBQ_SIZE
`define L3CWBQ_SIZE `L3REQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef L3DWBQ_SIZE
`define L3DWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef L3DFQQ_SIZE
`define L3DFQQ_SIZE `L3REQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef L3LLVQ_SIZE
`define L3LLVQ_SIZE 0
`endif

// Fill Forward SNP Queue
`ifndef L3FFSQ_SIZE
`define L3FFSQ_SIZE 8
`endif

// Prefetcher
`ifndef L3PRFQ_SIZE
`define L3PRFQ_SIZE 32
`endif

`ifndef L3PRFQ_STRIDE
`define L3PRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef L3FILL_INVALIDAOR_SIZE
`define L3FILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef L3SIMULATED_DRAM_LATENCY_CYCLES
`define L3SIMULATED_DRAM_LATENCY_CYCLES 2
`endif

 // VX_CONFIG
`endif
