`ifndef VX_CONFIG
`define VX_CONFIG

`include "VX_user_config.vh"

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

`ifndef GLOBAL_BLOCK_SIZE
`define GLOBAL_BLOCK_SIZE 64
`endif

`ifndef L1_BLOCK_SIZE
`define L1_BLOCK_SIZE (`NUM_THREADS * 4)
`endif

`ifndef STARTUP_ADDR
`define STARTUP_ADDR 32'h80000000
`endif

`ifndef IO_BUS_BASE_ADDR
`define IO_BUS_BASE_ADDR 32'hFF000000
`endif

`ifndef SHARED_MEM_BASE_ADDR
`define SHARED_MEM_BASE_ADDR `IO_BUS_BASE_ADDR
`endif

`ifndef SHARED_MEM_BASE_ADDR_ALIGN
`define SHARED_MEM_BASE_ADDR_ALIGN 64
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
`define LATENCY_FCVT 4
`endif

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

// Machine Counter/Timers
`define CSR_CYCLE       12'hC00
`define CSR_CYCLE_H     12'hC80
`define CSR_INSTRET     12'hC02
`define CSR_INSTRET_H   12'hC82

// Machine Performance-monitoring counters
// PERF: pipeline
`define CSR_MPM_IBUF_ST     12'hB03
`define CSR_MPM_IBUF_ST_H   12'hB83
`define CSR_MPM_SCRB_ST     12'hB04
`define CSR_MPM_SCRB_ST_H   12'hB84
`define CSR_MPM_ALU_ST      12'hB05
`define CSR_MPM_ALU_ST_H    12'hB85
`define CSR_MPM_LSU_ST      12'hB06
`define CSR_MPM_LSU_ST_H    12'hB86
`define CSR_MPM_CSR_ST      12'hB07
`define CSR_MPM_CSR_ST_H    12'hB87
`define CSR_MPM_FPU_ST      12'hB08
`define CSR_MPM_FPU_ST_H    12'hB88
`define CSR_MPM_GPU_ST      12'hB09
`define CSR_MPM_GPU_ST_H    12'hB89
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
`define CSR_MPM_SMEM_READS      12'hB16     // total reads
`define CSR_MPM_SMEM_READS_H    12'hB96
`define CSR_MPM_SMEM_WRITES     12'hB17     // total writes
`define CSR_MPM_SMEM_WRITES_H   12'hB97
`define CSR_MPM_SMEM_BANK_ST    12'hB18     // bank conflicts stalls
`define CSR_MPM_SMEM_BANK_ST_H  12'hB98
// PERF: memory
`define CSR_MPM_DRAM_READS      12'hB19     // dram reads
`define CSR_MPM_DRAM_READS_H    12'hB99
`define CSR_MPM_DRAM_WRITES     12'hB1A     // dram writes
`define CSR_MPM_DRAM_WRITES_H   12'hB9A
`define CSR_MPM_DRAM_ST         12'hB1B     // dram request stalls
`define CSR_MPM_DRAM_ST_H       12'hB9B
`define CSR_MPM_DRAM_LAT        12'hB1C     // dram latency (total)
`define CSR_MPM_DRAM_LAT_H      12'hB9C

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

// Machine SIMT CSRs
`define CSR_NT          12'hFC0
`define CSR_NW          12'hFC1
`define CSR_NC          12'hFC2

// Pipeline Queues ////////////////////////////////////////////////////////////

// Size of LSU Request Queue
`ifndef LSUQ_SIZE
`define LSUQ_SIZE 8
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
`ifndef ICREQ_SIZE
`define ICREQ_SIZE 4
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
`define DCACHE_SIZE 16384
`endif

// Number of banks
`ifndef DNUM_BANKS
`define DNUM_BANKS `NUM_THREADS
`endif

// Number of bank ports
`ifndef DNUM_PORTS
`define DNUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef DCREQ_SIZE
`define DCREQ_SIZE 4
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
`define DDRSQ_SIZE `MAX(4, (`DNUM_BANKS * 2))
`endif

// SM Configurable Knobs //////////////////////////////////////////////////////

// per thread stack size
`ifndef STACK_SIZE
`define STACK_SIZE 1024
`endif

// Size of cache in bytes
`ifndef SMEM_SIZE
`define SMEM_SIZE (`STACK_SIZE * `NUM_WARPS * `NUM_THREADS)
`endif

// Number of banks
`ifndef SNUM_BANKS
`define SNUM_BANKS `NUM_THREADS
`endif

// Core Request Queue Size
`ifndef SCREQ_SIZE
`define SCREQ_SIZE 4
`endif

// L2cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L2CACHE_SIZE
`define L2CACHE_SIZE 65536
`endif

// Number of banks
`ifndef L2NUM_BANKS
`define L2NUM_BANKS `MIN(`NUM_CORES, 4)
`endif

// Core Request Queue Size
`ifndef L2CREQ_SIZE
`define L2CREQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef L2MSHR_SIZE
`define L2MSHR_SIZE 16
`endif

// DRAM Request Queue Size
`ifndef L2DREQ_SIZE
`define L2DREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef L2DRSQ_SIZE
`define L2DRSQ_SIZE `MAX(4, (`L2NUM_BANKS * 2))
`endif

// L3cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L3CACHE_SIZE
`define L3CACHE_SIZE 131072
`endif

// Number of banks
`ifndef L3NUM_BANKS
`define L3NUM_BANKS `MIN(`NUM_CLUSTERS, 4)
`endif

// Core Request Queue Size
`ifndef L3CREQ_SIZE
`define L3CREQ_SIZE 4
`endif

// Miss Handling Register Size
`ifndef L3MSHR_SIZE
`define L3MSHR_SIZE 16
`endif

// DRAM Request Queue Size
`ifndef L3DREQ_SIZE
`define L3DREQ_SIZE 4
`endif

// DRAM Response Queue Size
`ifndef L3DRSQ_SIZE
`define L3DRSQ_SIZE `MAX(4, (`L3NUM_BANKS * 2))
`endif

`endif
