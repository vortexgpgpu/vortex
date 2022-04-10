`ifndef VX_CONFIG_VH
`define VX_CONFIG_VH

`ifndef XLEN
`define XLEN 32
`endif

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

`ifdef L2_ENABLE
    `define L2_ENABLED   1
`else
    `define L2_ENABLED   0
`endif

`ifdef L3_ENABLE
    `define L3_ENABLED   1
`else
    `define L3_ENABLED   0
`endif

`ifdef L1_DISABLE
    `define ICACHE_DISABLE
    `define DCACHE_DISABLE
    `define TCACHE_DISABLE
`endif

`ifndef MEM_BLOCK_SIZE
`define MEM_BLOCK_SIZE 64
`endif

`ifndef L1_BLOCK_SIZE
`ifdef L1_DISABLE
`define L1_BLOCK_SIZE ((`L2_ENABLED || `L3_ENABLED) ? 4 : `MEM_BLOCK_SIZE)
`else
`define L1_BLOCK_SIZE ((`L2_ENABLED || `L3_ENABLED) ? 16 : `MEM_BLOCK_SIZE)
`endif
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

`ifndef STACK_BASE_ADDR
`define STACK_BASE_ADDR `IO_BASE_ADDR
`endif

`ifndef STACK_SIZE
`define STACK_SIZE 8192
`endif

// ISA Extensions /////////////////////////////////////////////////////////////

`ifndef EXT_M_DISABLE
`define EXT_M_ENABLE
`endif

`ifndef EXT_F_DISABLE
`define EXT_F_ENABLE
`endif

`ifdef EXT_GFX_ENABLE
`define EXT_TEX_ENABLE
`define EXT_RASTER_ENABLE
`define EXT_ROP_ENABLE
`define EXT_IMADD_ENABLE
`endif

`define ISA_STD_A           0
`define ISA_STD_C           2
`define ISA_STD_D           3
`define ISA_STD_E           4
`define ISA_STD_F           5
`define ISA_STD_H           7
`define ISA_STD_I           8
`define ISA_STD_N           13
`define ISA_STD_Q           16
`define ISA_STD_S           18
`define ISA_STD_U           20

`define ISA_EXT_TEX         0
`define ISA_EXT_RASTER      1
`define ISA_EXT_ROP         2
`define ISA_EXT_IMADD       3

`ifdef EXT_A_ENABLE
    `define EXT_A_ENABLED   1
`else
    `define EXT_A_ENABLED   0
`endif

`ifdef EXT_C_ENABLE
    `define EXT_C_ENABLED   1
`else
    `define EXT_C_ENABLED   0
`endif

`ifdef EXT_D_ENABLE
    `define EXT_D_ENABLED   1
`else
    `define EXT_D_ENABLED   0
`endif

`ifdef EXT_F_ENABLE
    `define EXT_F_ENABLED   1
`else
    `define EXT_F_ENABLED   0
`endif

`ifdef EXT_M_ENABLE
    `define EXT_M_ENABLED   1
`else
    `define EXT_M_ENABLED   0
`endif

`ifdef EXT_TEX_ENABLE
    `define EXT_TEX_ENABLED 1
`else
    `define EXT_TEX_ENABLED 0
`endif

`ifdef EXT_RASTER_ENABLE
    `define EXT_RASTER_ENABLED 1
`else
    `define EXT_RASTER_ENABLED 0
`endif

`ifdef EXT_ROP_ENABLE
    `define EXT_ROP_ENABLED 1
`else
    `define EXT_ROP_ENABLED 0
`endif

`ifdef EXT_IMADD_ENABLE
    `define EXT_IMADD_ENABLED 1
`else
    `define EXT_IMADD_ENABLED 0
`endif

`define ISA_X_ENABLED  ( `EXT_TEX_ENABLED       \
                       | `EXT_RASTER_ENABLED    \
                       | `EXT_ROP_ENABLED       \
                       | `EXT_IMADD_ENABLED     \
                       )

`define MISA_EXT    (`EXT_TEX_ENABLED << `ISA_EXT_TEX)        \
                  | (`EXT_RASTER_ENABLED << `ISA_EXT_RASTER)  \
                  | (`EXT_ROP_ENABLED << `ISA_EXT_ROP)        \
                  | (`EXT_IMADD_ENABLED << `ISA_EXT_IMADD)

`define MISA_STD  (`EXT_A_ENABLED <<  0) /* A - Atomic Instructions extension */ \
                | (0 <<  1) /* B - Tentatively reserved for Bit operations extension */ \
                | (`EXT_C_ENABLED <<  2) /* C - Compressed extension */ \
                | (`EXT_D_ENABLED <<  3) /* D - Double precsision floating-point extension */ \
                | (0 <<  4) /* E - RV32E base ISA */ \
                | (`EXT_F_ENABLED << 5) /* F - Single precsision floating-point extension */ \
                | (0 <<  6) /* G - Additional standard extensions present */ \
                | (0 <<  7) /* H - Hypervisor mode implemented */ \
                | (1 <<  8) /* I - RV32I/64I/128I base ISA */ \
                | (0 <<  9) /* J - Reserved */ \
                | (0 << 10) /* K - Reserved */ \
                | (0 << 11) /* L - Tentatively reserved for Bit operations extension */ \
                | (`EXT_M_ENABLED << 12) /* M - Integer Multiply/Divide extension */ \
                | (0 << 13) /* N - User level interrupts supported */ \
                | (0 << 14) /* O - Reserved */ \
                | (0 << 15) /* P - Tentatively reserved for Packed-SIMD extension */ \
                | (0 << 16) /* Q - Quad-precision floating-point extension */ \
                | (0 << 17) /* R - Reserved */ \
                | (0 << 18) /* S - Supervisor mode implemented */ \
                | (0 << 19) /* T - Tentatively reserved for Transactional Memory extension */ \
                | (1 << 20) /* U - User mode implemented */ \
                | (0 << 21) /* V - Tentatively reserved for Vector extension */ \
                | (0 << 22) /* W - Reserved */ \
                | (`ISA_X_ENABLED << 23) /* X - Non-standard extensions present */ \
                | (0 << 24) /* Y - Reserved */ \
                | (0 << 25) /* Z - Reserved */

// Device identification //////////////////////////////////////////////////////

`define VENDOR_ID           0
`define ARCHITECTURE_ID     0
`define IMPLEMENTATION_ID   0

// Pipeline latencies /////////////////////////////////////////////////////////

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

// Texture Unit Request Queue
`ifndef TEXQ_SIZE
`define TEXQ_SIZE (`NUM_WARPS * 2)
`endif

// RASTER memeory queue size
`ifndef RASTER_MEM_QUEUE_SIZE    
`define RASTER_MEM_QUEUE_SIZE 8
`endif

// ROP memeory queue size
`ifndef ROP_MEM_QUEUE_SIZE    
`define ROP_MEM_QUEUE_SIZE 8
`endif

// Icache Configurable Knobs //////////////////////////////////////////////////

// Enable/disable cache
`ifndef ICACHE_DISABLE
`define ICACHE_ENABLE
`endif
`ifdef ICACHE_ENABLE
    `define ICACHE_ENABLED 1
`else
    `define ICACHE_ENABLED 0
`endif

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

// Number of associative ways
`ifndef ICACHE_NUM_WAYS
`define ICACHE_NUM_WAYS 2
`endif

// Dcache Configurable Knobs //////////////////////////////////////////////////

// Enable/disable cache
`ifndef DCACHE_DISABLE
`define DCACHE_ENABLE
`endif
`ifdef DCACHE_ENABLE
    `define DCACHE_ENABLED 1
`else
    `define DCACHE_ENABLED 0
`endif

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

// Number of associative ways
`ifndef DCACHE_NUM_WAYS
`define DCACHE_NUM_WAYS 2
`endif

// Tcache Configurable Knobs //////////////////////////////////////////////////

// Enable/disable cache
`ifndef TCACHE_DISABLE
`define TCACHE_ENABLE
`endif
`ifdef TCACHE_ENABLE
    `define TCACHE_ENABLED 1
`else
    `define TCACHE_ENABLED 0
`endif

// Size of cache in bytes
`ifndef TCACHE_SIZE
`define TCACHE_SIZE 8192
`endif

// Number of banks
`ifndef TCACHE_NUM_BANKS
`define TCACHE_NUM_BANKS `NUM_THREADS
`endif

// Number of ports per bank
`ifndef TCACHE_NUM_PORTS
`define TCACHE_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef TCACHE_CREQ_SIZE
`define TCACHE_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef TCACHE_CRSQ_SIZE
`define TCACHE_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef TCACHE_MSHR_SIZE
`define TCACHE_MSHR_SIZE (`NUM_THREADS * 4)
`endif

// Memory Request Queue Size
`ifndef TCACHE_MREQ_SIZE
`define TCACHE_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef TCACHE_MRSQ_SIZE
`define TCACHE_MRSQ_SIZE 0
`endif

// Number of associative ways
`ifndef TCACHE_NUM_WAYS
`define TCACHE_NUM_WAYS 2
`endif

// Rcache Configurable Knobs //////////////////////////////////////////////////

// Enable/disable cache
`ifndef RCACHE_DISABLE
`define RCACHE_ENABLE
`endif
`ifdef RCACHE_ENABLE
    `define RCACHE_ENABLED 1
`else
    `define RCACHE_ENABLED 0
`endif

// Size of cache in bytes
`ifndef RCACHE_SIZE
`define RCACHE_SIZE 4096
`endif

// Number of banks
`ifndef RCACHE_NUM_BANKS
`define RCACHE_NUM_BANKS 1
`endif

// Number of ports per bank
`ifndef RCACHE_NUM_PORTS
`define RCACHE_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef RCACHE_CREQ_SIZE
`define RCACHE_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef RCACHE_CRSQ_SIZE
`define RCACHE_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef RCACHE_MSHR_SIZE
`define RCACHE_MSHR_SIZE 8
`endif

// Memory Request Queue Size
`ifndef RCACHE_MREQ_SIZE
`define RCACHE_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef RCACHE_MRSQ_SIZE
`define RCACHE_MRSQ_SIZE 0
`endif

// Number of associative ways
`ifndef RCACHE_NUM_WAYS
`define RCACHE_NUM_WAYS 2
`endif

// Ocache Configurable Knobs //////////////////////////////////////////////////

// Enable/disable cache
`ifndef OCACHE_DISABLE
`define OCACHE_ENABLE
`endif
`ifdef OCACHE_ENABLE
    `define OCACHE_ENABLED 1
`else
    `define OCACHE_ENABLED 0
`endif

// Size of cache in bytes
`ifndef OCACHE_SIZE
`define OCACHE_SIZE 4096
`endif

// Number of banks
`ifndef OCACHE_NUM_BANKS
`define OCACHE_NUM_BANKS `NUM_THREADS
`endif

// Number of ports per bank
`ifndef OCACHE_NUM_PORTS
`define OCACHE_NUM_PORTS 1
`endif

// Core Request Queue Size
`ifndef OCACHE_CREQ_SIZE
`define OCACHE_CREQ_SIZE 0
`endif

// Core Response Queue Size
`ifndef OCACHE_CRSQ_SIZE
`define OCACHE_CRSQ_SIZE 2
`endif

// Miss Handling Register Size
`ifndef OCACHE_MSHR_SIZE
`define OCACHE_MSHR_SIZE 8
`endif

// Memory Request Queue Size
`ifndef OCACHE_MREQ_SIZE
`define OCACHE_MREQ_SIZE 4
`endif

// Memory Response Queue Size
`ifndef OCACHE_MRSQ_SIZE
`define OCACHE_MRSQ_SIZE 0
`endif

// Number of associative ways
`ifndef OCACHE_NUM_WAYS
`define OCACHE_NUM_WAYS 2
`endif

// SM Configurable Knobs //////////////////////////////////////////////////////

`ifndef SM_DISABLE
`define SM_ENABLE
`endif

`ifdef SM_ENABLE
    `define SM_ENABLED   1
`else
    `define SM_ENABLED   0
`endif

// per thread stack size
`ifndef SMEM_LOCAL_SIZE
`define SMEM_LOCAL_SIZE 1024
`endif

// Size of storage in bytes
`ifndef SMEM_SIZE
`define SMEM_SIZE (`SMEM_LOCAL_SIZE * `NUM_WARPS * `NUM_THREADS)
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
`define L2_NUM_BANKS ((`NUM_CORES < 4) ? `NUM_CORES : 4)
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

// Number of associative ways
`ifndef L2_NUM_WAYS
`define L2_NUM_WAYS 4
`endif

// L3cache Configurable Knobs /////////////////////////////////////////////////

// Size of cache in bytes
`ifndef L3_CACHE_SIZE
`define L3_CACHE_SIZE 1048576
`endif

// Number of banks
`ifndef L3_NUM_BANKS
`define L3_NUM_BANKS ((`NUM_CLUSTERS < 4) ? `NUM_CORES : 4)
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

// Number of associative ways
`ifndef L3_NUM_WAYS
`define L3_NUM_WAYS 4
`endif

`endif
