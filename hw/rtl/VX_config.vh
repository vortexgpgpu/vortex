`ifndef VX_CONFIG
`define VX_CONFIG

`ifndef NUM_CLUSTERS
`define NUM_CLUSTERS 1
`endif

`ifndef NUM_CORES
`define NUM_CORES 1
`endif

`ifndef NUM_WARPS
`define NUM_WARPS 4
`endif

`ifndef NUM_THREADS
`define NUM_THREADS 4
`endif

`ifndef NUM_BARRIERS
`define NUM_BARRIERS 4
`endif

`ifndef L2_ENABLE
`define L2_ENABLE 0
`endif

`ifndef L3_ENABLE
`define L3_ENABLE 0
`endif

`ifndef SM_ENABLE
`define SM_ENABLE 1
`endif

`ifndef MEM_BLOCK_SIZE
`define MEM_BLOCK_SIZE 64
`endif

`ifndef L1_BLOCK_SIZE
`define L1_BLOCK_SIZE ((`L2_ENABLE || `L3_ENABLE) ? 16 : `MEM_BLOCK_SIZE)
`endif

`ifndef STARTUP_ADDR
`define STARTUP_ADDR 32'h80000000
`endif

`ifndef IO_BASE_ADDR
`define IO_BASE_ADDR 32'hFF000000
`endif

`ifndef IO_ADDR_SIZE
`define IO_ADDR_SIZE (32'hFFFFFFFF - `IO_BASE_ADDR + 1)
`endif

`ifndef IO_COUT_ADDR
`define IO_COUT_ADDR (32'hFFFFFFFF - `MEM_BLOCK_SIZE + 1)
`endif

`ifndef IO_COUT_SIZE
`define IO_COUT_SIZE `MEM_BLOCK_SIZE
`endif

`ifndef IO_CSR_ADDR
`define IO_CSR_ADDR `IO_BASE_ADDR
`endif

`ifndef SMEM_BASE_ADDR
`define SMEM_BASE_ADDR `IO_BASE_ADDR
`endif

`ifndef EXT_M_DISABLE
`define EXT_M_ENABLE
`endif

`ifndef EXT_F_DISABLE
`define EXT_F_ENABLE
`endif

// Device identification
`define VENDOR_ID           0
`define ARCHITECTURE_ID     0
`define IMPLEMENTATION_ID   0

///////////////////////////////////////////////////////////////////////////////

`ifndef LATENCY_IMUL
`define LATENCY_IMUL 3
`endif

`ifndef LATENCY_FNCP
`define LATENCY_FNCP 2
`endif

`ifndef LATENCY_FMA
`define LATENCY_FMA 4
`endif

`ifndef LATENCY_FDIV
`ifdef ALTERA_S10
`define LATENCY_FDIV 34
`else
`define LATENCY_FDIV 15
`endif
`endif

`ifndef LATENCY_FSQRT
`ifdef ALTERA_S10
`define LATENCY_FSQRT 25
`else
`define LATENCY_FSQRT 10
`endif
`endif

`ifndef LATENCY_FDIVSQRT
`define LATENCY_FDIVSQRT 32
`endif

`ifndef LATENCY_FCVT
`define LATENCY_FCVT 5
`endif

`define RESET_DELAY 6

// CSR Addresses //////////////////////////////////////////////////////////////

// User Floating-Point CSRs
`define CSR_FFLAGS      12'h001
`define CSR_FRM         12'h002
`define CSR_FCSR        12'h003

`define CSR_SATP        12'h180

`define CSR_PMPCFG0     12'h3A0
`define CSR_PMPADDR0    12'h3B0

`define CSR_MSTATUS     12'h300
`define CSR_MISA        12'h301
`define CSR_MEDELEG     12'h302
`define CSR_MIDELEG     12'h303
`define CSR_MIE         12'h304
`define CSR_MTVEC       12'h305

`define CSR_MEPC        12'h341

// Machine Performance-monitoring counters
`define CSR_MPM_BASE                12'hB00
`define CSR_MPM_BASE_H              12'hB80
// PERF: pipeline
`define CSR_MCYCLE                  12'hB00
`define CSR_MCYCLE_H                12'hB80
`define CSR_MPM_RESERVED            12'hB01
`define CSR_MPM_RESERVED_H          12'hB81
`define CSR_MINSTRET                12'hB02
`define CSR_MINSTRET_H              12'hB82
`define CSR_MPM_IBUF_ST             12'hB03
`define CSR_MPM_IBUF_ST_H           12'hB83
`define CSR_MPM_SCRB_ST             12'hB04
`define CSR_MPM_SCRB_ST_H           12'hB84
`define CSR_MPM_ALU_ST              12'hB05
`define CSR_MPM_ALU_ST_H            12'hB85
`define CSR_MPM_LSU_ST              12'hB06
`define CSR_MPM_LSU_ST_H            12'hB86
`define CSR_MPM_CSR_ST              12'hB07
`define CSR_MPM_CSR_ST_H            12'hB87
`define CSR_MPM_FPU_ST              12'hB08
`define CSR_MPM_FPU_ST_H            12'hB88
`define CSR_MPM_GPU_ST              12'hB09
`define CSR_MPM_GPU_ST_H            12'hB89
// PERF: icache
`define CSR_MPM_ICACHE_READS        12'hB0A     // total reads
`define CSR_MPM_ICACHE_READS_H      12'hB8A
`define CSR_MPM_ICACHE_MISS_R       12'hB0B     // total misses
`define CSR_MPM_ICACHE_MISS_R_H     12'hB8B
`define CSR_MPM_ICACHE_PIPE_ST      12'hB0C     // pipeline stalls
`define CSR_MPM_ICACHE_PIPE_ST_H    12'hB8C
`define CSR_MPM_ICACHE_CRSP_ST      12'hB0D     // core response stalls
`define CSR_MPM_ICACHE_CRSP_ST_H    12'hB8D
// PERF: dcache
`define CSR_MPM_DCACHE_READS        12'hB0E     // total reads
`define CSR_MPM_DCACHE_READS_H      12'hB8E
`define CSR_MPM_DCACHE_WRITES       12'hB0F     // total writes
`define CSR_MPM_DCACHE_WRITES_H     12'hB8F 
`define CSR_MPM_DCACHE_MISS_R       12'hB10     // read misses
`define CSR_MPM_DCACHE_MISS_R_H     12'hB90
`define CSR_MPM_DCACHE_MISS_W       12'hB11     // write misses
`define CSR_MPM_DCACHE_MISS_W_H     12'hB91
`define CSR_MPM_DCACHE_BANK_ST      12'hB12     // bank conflicts stalls
`define CSR_MPM_DCACHE_BANK_ST_H    12'hB92
`define CSR_MPM_DCACHE_MSHR_ST      12'hB13     // MSHR stalls
`define CSR_MPM_DCACHE_MSHR_ST_H    12'hB93
`define CSR_MPM_DCACHE_PIPE_ST      12'hB14     // pipeline stalls
`define CSR_MPM_DCACHE_PIPE_ST_H    12'hB94
`define CSR_MPM_DCACHE_CRSP_ST      12'hB15     // core response stalls
`define CSR_MPM_DCACHE_CRSP_ST_H    12'hB95
// PERF: smem
`define CSR_MPM_SMEM_READS          12'hB16     // total reads
`define CSR_MPM_SMEM_READS_H        12'hB96
`define CSR_MPM_SMEM_WRITES         12'hB17     // total writes
`define CSR_MPM_SMEM_WRITES_H       12'hB97
`define CSR_MPM_SMEM_BANK_ST        12'hB18     // bank conflicts stalls
`define CSR_MPM_SMEM_BANK_ST_H      12'hB98
// PERF: memory
`define CSR_MPM_MEM_READS           12'hB19     // memory reads
`define CSR_MPM_MEM_READS_H         12'hB99
`define CSR_MPM_MEM_WRITES          12'hB1A     // memory writes
`define CSR_MPM_MEM_WRITES_H        12'hB9A
`define CSR_MPM_MEM_ST              12'hB1B     // memory request stalls
`define CSR_MPM_MEM_ST_H            12'hB9B
`define CSR_MPM_MEM_LAT             12'hB1C     // memory latency (total)
`define CSR_MPM_MEM_LAT_H           12'hB9C

// Machine Information Registers
`define CSR_MVENDORID   12'hF11
`define CSR_MARCHID     12'hF12
`define CSR_MIMPID      12'hF13
`define CSR_MHARTID     12'hF14

// User SIMT CSRs
`define CSR_WTID        12'hCC0
`define CSR_LTID        12'hCC1
`define CSR_GTID        12'hCC2
`define CSR_LWID        12'hCC3
`define CSR_GWID        `CSR_MHARTID
`define CSR_GCID        12'hCC5
`define CSR_TMASK       12'hCC4

// Machine SIMT CSRs
`define CSR_NT          12'hFC0
`define CSR_NW          12'hFC1
`define CSR_NC          12'hFC2

////////// Texture Units //////////////////////////////////////////////////////

`define NUM_TEX_UNITS       2

`define CSR_TEX_STATES      7
`define CSR_TEX_BEGIN(x)    (12'hFD0 + (x) * `CSR_TEX_STATES)

`define CSR_TEX_ADDR(x)     (`CSR_TEX_BEGIN(x) + 12'h00)
`define CSR_TEX_FORMAT(x)   (`CSR_TEX_BEGIN(x) + 12'h01)
`define CSR_TEX_WRAP(x)     (`CSR_TEX_BEGIN(x) + 12'h02)
`define CSR_TEX_FILTER(x)   (`CSR_TEX_BEGIN(x) + 12'h03)
`define CSR_TEX_MIPOFF(x)   (`CSR_TEX_BEGIN(x) + 12'h04)
`define CSR_TEX_WIDTH(x)    (`CSR_TEX_BEGIN(x) + 12'h05)
`define CSR_TEX_HEIGHT(x)   (`CSR_TEX_BEGIN(x) + 12'h06)

// Pipeline Queues ////////////////////////////////////////////////////////////

// Size of Instruction Buffer
`ifndef IBUF_SIZE
`define IBUF_SIZE 2
`endif

// Size of LSU Request Queue
`ifndef LSUQ_SIZE
`define LSUQ_SIZE (`NUM_WARPS * 2)
`endif

// Size of FPU Request Queue
`ifndef FPUQ_SIZE
`define FPUQ_SIZE 8
`endif

// Icache Configurable Knobs //////////////////////////////////////////////////

// Size of cache in bytes
`ifndef ICACHE_SIZE
`define ICACHE_SIZE 16384
`endif

// Core Request Queue Size
`ifndef ICACHE_CREQ_SIZE
`define ICACHE_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef ICACHE_CRSQ_SIZE
`define ICACHE_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef ICACHE_MSHR_SIZE
`define ICACHE_MSHR_SIZE `NUM_WARPS
`endif

// Memory Request Queue Size
`ifndef ICACHE_MREQ_SIZE
`define ICACHE_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef ICACHE_MRSQ_SIZE
`define ICACHE_MRSQ_SIZE 0
`endif

// Dcache Configurable Knobs //////////////////////////////////////////////////

// Size of cache in bytes
`ifndef DCACHE_SIZE
`define DCACHE_SIZE 16384
`endif

// Number of banks
`ifndef DCACHE_NUM_BANKS
`define DCACHE_NUM_BANKS `NUM_THREADS
`endif

// Number of ports per bank
`ifndef DCACHE_NUM_PORTS
`define DCACHE_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef DCACHE_CREQ_SIZE
`define DCACHE_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef DCACHE_CRSQ_SIZE
`define DCACHE_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef DCACHE_MSHR_SIZE
`define DCACHE_MSHR_SIZE `LSUQ_SIZE
`endif

// Memory Request Queue Size
`ifndef DCACHE_MREQ_SIZE
`define DCACHE_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef DCACHE_MRSQ_SIZE
`define DCACHE_MRSQ_SIZE 0
`endif

// SM Configurable Knobs //////////////////////////////////////////////////////

// per thread stack size
`ifndef STACK_LOG2_SIZE
`define STACK_LOG2_SIZE 10
`endif
`define STACK_SIZE (1 << `STACK_LOG2_SIZE)

// Size of cache in bytes
`ifndef SMEM_SIZE
`define SMEM_SIZE (`STACK_SIZE * `NUM_WARPS * `NUM_THREADS)
`endif

// Number of banks
`ifndef SMEM_NUM_BANKS
`define SMEM_NUM_BANKS `NUM_THREADS
`endif

// Core Request Queue Size
`ifndef SMEM_CREQ_SIZE
`define SMEM_CREQ_SIZE 2
`endif

// Core Response Queue Size
`ifndef SMEM_CRSQ_SIZE
`define SMEM_CRSQ_SIZE 2
`endif

// L2cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L2_CACHE_SIZE
`define L2_CACHE_SIZE 131072
`endif

// Number of banks
`ifndef L2_NUM_BANKS
`define L2_NUM_BANKS `MIN(`NUM_CORES, 4)
`endif

// Number of ports per bank
`ifndef L2_NUM_PORTS
`define L2_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef L2_CREQ_SIZE
`define L2_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef L2_CRSQ_SIZE
`define L2_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef L2_MSHR_SIZE
`define L2_MSHR_SIZE 16
`endif

// Memory Request Queue Size
`ifndef L2_MREQ_SIZE
`define L2_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef L2_MRSQ_SIZE
`define L2_MRSQ_SIZE 0
`endif

// L3cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L3_CACHE_SIZE
`define L3_CACHE_SIZE 1048576
`endif

// Number of banks
`ifndef L3_NUM_BANKS
`define L3_NUM_BANKS `MIN(`NUM_CLUSTERS, 4)
`endif

// Number of ports per bank
`ifndef L3_NUM_PORTS
`define L3_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef L3_CREQ_SIZE
`define L3_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef L3_CRSQ_SIZE
`define L3_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef L3_MSHR_SIZE
`define L3_MSHR_SIZE 16
`endif

// Memory Request Queue Size
`ifndef L3_MREQ_SIZE
`define L3_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef L3_MRSQ_SIZE
`define L3_MRSQ_SIZE 0
`endif

`endif
