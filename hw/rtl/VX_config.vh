`ifndef VX_CONFIG
`define VX_CONFIG

`include "VX_user_config.vh"

`ifndef NUM_CLUSTERS
`define NUM_CLUSTERS 1
`endif

`ifndef NUM_CORES
`define NUM_CORES 4
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
`define L2_ENABLE (`NUM_CORES >= 4)
`endif

`ifndef L3_ENABLE
`define L3_ENABLE (`NUM_CLUSTERS >= 4)
`endif

`ifndef GLOBAL_BLOCK_SIZE
`define GLOBAL_BLOCK_SIZE 64
`endif

`ifndef L1_BLOCK_SIZE
`define L1_BLOCK_SIZE (`NUM_THREADS * 4)
`endif

`ifndef STARTUP_ADDR
`define STARTUP_ADDR 32'h80000000
`endif

`ifndef SHARED_MEM_BASE_ADDR
`define SHARED_MEM_BASE_ADDR 32'h6FFFF000
`endif

`ifndef IO_BUS_BASE_ADDR
`define IO_BUS_BASE_ADDR 32'hFF000000
`endif

`ifndef IO_BUS_ADDR_COUT
`define IO_BUS_ADDR_COUT 32'hFFFFFFFC
`endif

`ifndef FRAME_BUFFER_BASE_ADDR
`define FRAME_BUFFER_BASE_ADDR 32'hFF000000
`endif

`ifndef FRAME_BUFFER_WIDTH
`define FRAME_BUFFER_WIDTH 16'd1920
`endif

`ifndef FRAME_BUFFER_HEIGHT
`define FRAME_BUFFER_HEIGHT 16'd1080
`endif

`define FRAME_BUFFER_SIZE (FRAME_BUFFER_WIDTH * FRAME_BUFFER_HEIGHT)

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

`ifndef LATENCY_FNONCOMP
`define LATENCY_FNONCOMP 1
`endif

`ifndef LATENCY_FADDMUL
`define LATENCY_FADDMUL 3
`endif

`ifndef LATENCY_FMADD
`define LATENCY_FMADD 4
`endif

`ifndef LATENCY_FDIV
`define LATENCY_FDIV 15
`endif

`ifndef LATENCY_FSQRT
`define LATENCY_FSQRT 10
`endif

`ifndef LATENCY_ITOF
`define LATENCY_ITOF 7
`endif

`ifndef LATENCY_FTOI
`define LATENCY_FTOI 3
`endif

`ifndef LATENCY_FDIVSQRT
`define LATENCY_FDIVSQRT 10
`endif

`ifndef LATENCY_FCONV
`define LATENCY_FCONV 3
`endif

// CSR Addresses //////////////////////////////////////////////////////////////

// User Floating-Point CSRs
`define CSR_FFLAGS      12'h001
`define CSR_FRM         12'h002
`define CSR_FCSR        12'h003

// SIMT CSRs
`define CSR_LTID        12'h020
`define CSR_LWID        12'h021
`define CSR_GTID        12'h022
`define CSR_GWID        12'h023
`define CSR_GCID        12'h024
`define CSR_NT          12'h025
`define CSR_NW          12'h026
`define CSR_NC          12'h027

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

// Machine Counter/Timers
`define CSR_MCYCLE      12'hB00
`define CSR_MCYCLE_H    12'hB80
`define CSR_MINSTRET    12'hB02
`define CSR_MINSTRET_H  12'hB82

// Machine Performance-monitoring counters
// PERF: pipeline
`define CSR_MPM_ICACHE_ST   12'hB03
`define CSR_MPM_ICACHE_ST_H 12'hB83
`define CSR_MPM_IBUF_ST     12'hB04
`define CSR_MPM_IBUF_ST_H   12'hB84
`define CSR_MPM_SCRB_ST     12'hB05
`define CSR_MPM_SCRB_ST_H   12'hB85
`define CSR_MPM_ALU_ST      12'hB06
`define CSR_MPM_ALU_ST_H    12'hB86
`define CSR_MPM_LSU_ST      12'hB07
`define CSR_MPM_LSU_ST_H    12'hB87
`define CSR_MPM_CSR_ST      12'hB08
`define CSR_MPM_CSR_ST_H    12'hB88
`define CSR_MPM_MUL_ST      12'hB09
`define CSR_MPM_MUL_ST_H    12'hB89
`define CSR_MPM_FPU_ST      12'hB0A
`define CSR_MPM_FPU_ST_H    12'hB8A
`define CSR_MPM_GPU_ST      12'hB0B
`define CSR_MPM_GPU_ST_H    12'hB8B
// PERF: icache
`define CSR_MPM_ICACHE_MISS_R       12'hB0C     // read misses
`define CSR_MPM_ICACHE_MISS_R_H     12'hB8C
`define CSR_MPM_ICACHE_DREQ_ST      12'hB0D     // dram request stalls
`define CSR_MPM_ICACHE_DREQ_ST_H    12'hB8D
`define CSR_MPM_ICACHE_CRSP_ST      12'hB0E     // core response stalls
`define CSR_MPM_ICACHE_CRSP_ST_H    12'hB8E
`define CSR_MPM_ICACHE_MSHR_ST      12'hB0F     // MSHR stalls
`define CSR_MPM_ICACHE_MSHR_ST_H    12'hB8F
`define CSR_MPM_ICACHE_PIPE_ST      12'hB10     // pipeline stalls
`define CSR_MPM_ICACHE_PIPE_ST_H    12'hB90
`define CSR_MPM_ICACHE_READS        12'hB11     // total reads
`define CSR_MPM_ICACHE_READS_H      12'hB91
// PERF: dcache
`define CSR_MPM_DCACHE_MISS_R       12'hB12     // read misses
`define CSR_MPM_DCACHE_MISS_R_H     12'hB92
`define CSR_MPM_DCACHE_MISS_W       12'hB13     // write misses
`define CSR_MPM_DCACHE_MISS_W_H     12'hB93
`define CSR_MPM_DCACHE_DREQ_ST      12'hB14     // dram request stalls
`define CSR_MPM_DCACHE_DREQ_ST_H    12'hB94
`define CSR_MPM_DCACHE_CRSP_ST      12'hB15     // core response stalls
`define CSR_MPM_DCACHE_CRSP_ST_H    12'hB95
`define CSR_MPM_DCACHE_MSHR_ST      12'hB16     // MSHR stalls
`define CSR_MPM_DCACHE_MSHR_ST_H    12'hB96
`define CSR_MPM_DCACHE_PIPE_ST      12'hB17     // pipeline stalls
`define CSR_MPM_DCACHE_PIPE_ST_H    12'hB97
`define CSR_MPM_DCACHE_READS        12'hB18     // total reads
`define CSR_MPM_DCACHE_READS_H      12'hB98
`define CSR_MPM_DCACHE_WRITES       12'hB19     // total writes
`define CSR_MPM_DCACHE_WRITES_H     12'hB99 
`define CSR_MPM_DCACHE_EVICTS       12'hB1A     // total evictions
`define CSR_MPM_DCACHE_EVICTS_H     12'hB9A 
// PERF: memory
`define CSR_MPM_DRAM_LAT    12'hB1B     // dram latency (total)
`define CSR_MPM_DRAM_LAT_H  12'hB9B
`define CSR_MPM_DRAM_REQ    12'hB1C     // dram requests
`define CSR_MPM_DRAM_REQ_H  12'hB9C
`define CSR_MPM_DRAM_RSP    12'hB1D     // dram responses
`define CSR_MPM_DRAM_RSP_H  12'hB9D

// Machine Information Registers
`define CSR_MVENDORID   12'hF11
`define CSR_MARCHID     12'hF12
`define CSR_MIMPID      12'hF13
`define CSR_MHARTID     12'hF14

// Pipeline Queues ////////////////////////////////////////////////////////////

// Size of instruction queue
`ifndef IBUF_SIZE
`define IBUF_SIZE 4
`endif

// Size of LSU Request Queue
`ifndef LSUQ_SIZE
`define LSUQ_SIZE 8
`endif

// Size of MUL Request Queue
`ifndef MULQ_SIZE
`define MULQ_SIZE 4
`endif

// Size of FPU Request Queue
`ifndef FPUQ_SIZE
`define FPUQ_SIZE 4
`endif

// Icache Configurable Knobs //////////////////////////////////////////////////

// Size of cache in bytes
`ifndef ICACHE_SIZE
`define ICACHE_SIZE 4096
`endif

// Core Request Queue Size
`ifndef ICREQ_SIZE
`define ICREQ_SIZE 4
`endif

// Core Response Queue Size
`ifndef ICRSQ_SIZE
`define ICRSQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef IMSHR_SIZE
`define IMSHR_SIZE `NUM_WARPS
`endif

// DRAM Request Queue Size
`ifndef IDREQ_SIZE
`define IDREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef IDRSQ_SIZE
`define IDRSQ_SIZE 4
`endif

// Dcache Configurable Knobs //////////////////////////////////////////////////

// Size of cache in bytes
`ifndef DCACHE_SIZE
`define DCACHE_SIZE 8192
`endif

// Number of banks
`ifndef DNUM_BANKS
`define DNUM_BANKS `NUM_THREADS
`endif

// Core Request Queue Size
`ifndef DCREQ_SIZE
`define DCREQ_SIZE 4
`endif

// Core Response Queue Size
`ifndef DCRSQ_SIZE
`define DCRSQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef DMSHR_SIZE
`define DMSHR_SIZE `LSUQ_SIZE
`endif

// DRAM Request Queue Size
`ifndef DDREQ_SIZE
`define DDREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef DDRSQ_SIZE
`define DDRSQ_SIZE 4
`endif

// Snoop Request Queue Size
`ifndef DSREQ_SIZE
`define DSREQ_SIZE 4
`endif

// Snoop Response Queue Size
`ifndef DSRSQ_SIZE
`define DSRSQ_SIZE 4
`endif

// SM Configurable Knobs //////////////////////////////////////////////////////

// Size of cache in bytes
`ifndef SMEM_SIZE
`define SMEM_SIZE 4096
`endif

// Number of banks
`ifndef SNUM_BANKS
`define SNUM_BANKS `NUM_THREADS
`endif

// Core Request Queue Size
`ifndef SCREQ_SIZE
`define SCREQ_SIZE 4
`endif

// Core Response Queue Size
`ifndef SCRSQ_SIZE
`define SCRSQ_SIZE 4
`endif

// L2cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L2CACHE_SIZE
`define L2CACHE_SIZE 131072
`endif

// Number of banks
`ifndef L2NUM_BANKS
`define L2NUM_BANKS `MIN(`NUM_CORES, 4)
`endif

// Core Request Queue Size
`ifndef L2CREQ_SIZE
`define L2CREQ_SIZE 4
`endif

// Core Response Queue Size
`ifndef L2CRSQ_SIZE
`define L2CRSQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef L2MSHR_SIZE
`define L2MSHR_SIZE 8
`endif

// DRAM Request Queue Size
`ifndef L2DREQ_SIZE
`define L2DREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef L2DRSQ_SIZE
`define L2DRSQ_SIZE 4
`endif

// Snoop Request Queue Size
`ifndef L2SREQ_SIZE
`define L2SREQ_SIZE 4
`endif

// Snoop Response Queue Size
`ifndef L2SRSQ_SIZE
`define L2SRSQ_SIZE 4
`endif

// L3cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L3CACHE_SIZE
`define L3CACHE_SIZE 262144
`endif

// Number of banks
`ifndef L3NUM_BANKS
`define L3NUM_BANKS `MIN(`NUM_CLUSTERS, 4)
`endif

// Core Request Queue Size
`ifndef L3CREQ_SIZE
`define L3CREQ_SIZE 4
`endif

// Core Response Queue Size
`ifndef L3CRSQ_SIZE
`define L3CRSQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef L3MSHR_SIZE
`define L3MSHR_SIZE 8
`endif

// DRAM Request Queue Size
`ifndef L3DREQ_SIZE
`define L3DREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef L3DRSQ_SIZE
`define L3DRSQ_SIZE 4
`endif

// Snoop Request Queue Size
`ifndef L3SREQ_SIZE
`define L3SREQ_SIZE 4
`endif

// Snoop Response Queue Size
`ifndef L3SRSQ_SIZE
`define L3SRSQ_SIZE 4
`endif

`endif
