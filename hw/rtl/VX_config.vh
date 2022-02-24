`ifndef VX_CONFIG
`define VX_CONFIG

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

`define ISA_X_ENABLED   (`EXT_TEX_ENABLED       \
                       | `EXT_RASTER_ENABLED    \
                       | `EXT_ROP_ENABLED)

`define MISA_EXT    (`EXT_TEX_ENABLED << `ISA_EXT_TEX)        \
                  | (`EXT_RASTER_ENABLED << `ISA_EXT_RASTER)  \
                  | (`EXT_ROP_ENABLED << `ISA_EXT_ROP)

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

// CSR Addresses //////////////////////////////////////////////////////////////

// User Floating-Point CSRs
`define CSR_FFLAGS                  12'h001
`define CSR_FRM                     12'h002
`define CSR_FCSR                    12'h003

`define CSR_SATP                    12'h180

`define CSR_PMPCFG0                 12'h3A0
`define CSR_PMPADDR0                12'h3B0

`define CSR_MSTATUS                 12'h300
`define CSR_MISA                    12'h301
`define CSR_MEDELEG                 12'h302
`define CSR_MIDELEG                 12'h303
`define CSR_MIE                     12'h304
`define CSR_MTVEC                   12'h305

`define CSR_MEPC                    12'h341

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
// PERF: decode
`define CSR_MPM_LOADS               12'hB0A
`define CSR_MPM_LOADS_H             12'hB8A
`define CSR_MPM_STORES              12'hB0B
`define CSR_MPM_STORES_H            12'hB8B
`define CSR_MPM_BRANCHES            12'hB0C
`define CSR_MPM_BRANCHES_H          12'hB8C
// PERF: icache
`define CSR_MPM_ICACHE_READS        12'hB0D     // total reads
`define CSR_MPM_ICACHE_READS_H      12'hB8D
`define CSR_MPM_ICACHE_MISS_R       12'hB0E     // read misses
`define CSR_MPM_ICACHE_MISS_R_H     12'hB8E
// PERF: dcache
`define CSR_MPM_DCACHE_READS        12'hB0F     // total reads
`define CSR_MPM_DCACHE_READS_H      12'hB8F
`define CSR_MPM_DCACHE_WRITES       12'hB10     // total writes
`define CSR_MPM_DCACHE_WRITES_H     12'hB90
`define CSR_MPM_DCACHE_MISS_R       12'hB11     // read misses
`define CSR_MPM_DCACHE_MISS_R_H     12'hB91
`define CSR_MPM_DCACHE_MISS_W       12'hB12     // write misses
`define CSR_MPM_DCACHE_MISS_W_H     12'hB92
`define CSR_MPM_DCACHE_BANK_ST      12'hB13     // bank conflicts
`define CSR_MPM_DCACHE_BANK_ST_H    12'hB93
`define CSR_MPM_DCACHE_MSHR_ST      12'hB14     // MSHR stalls
`define CSR_MPM_DCACHE_MSHR_ST_H    12'hB94
// PERF: smem
`define CSR_MPM_SMEM_READS          12'hB15     // total reads
`define CSR_MPM_SMEM_READS_H        12'hB95
`define CSR_MPM_SMEM_WRITES         12'hB16     // total writes
`define CSR_MPM_SMEM_WRITES_H       12'hB96
`define CSR_MPM_SMEM_BANK_ST        12'hB17     // bank conflicts
`define CSR_MPM_SMEM_BANK_ST_H      12'hB97
// PERF: memory
`define CSR_MPM_MEM_READS           12'hB18     // memory reads
`define CSR_MPM_MEM_READS_H         12'hB98
`define CSR_MPM_MEM_WRITES          12'hB19     // memory writes
`define CSR_MPM_MEM_WRITES_H        12'hB99
`define CSR_MPM_MEM_LAT             12'hB1A     // memory latency
`define CSR_MPM_MEM_LAT_H           12'hB9A
// PERF: texunit
`define CSR_MPM_TEX_READS           12'hB1B     // texture accesses
`define CSR_MPM_TEX_READS_H         12'hB9B
`define CSR_MPM_TEX_LAT             12'hB1C     // texture latency
`define CSR_MPM_TEX_LAT_H           12'hB9C

// Machine Information Registers
`define CSR_MVENDORID               12'hF11
`define CSR_MARCHID                 12'hF12
`define CSR_MIMPID                  12'hF13
`define CSR_MHARTID                 12'hF14

// Vortex GPGU CSRs
`define CSR_WTID                    12'hCC0
`define CSR_LTID                    12'hCC1
`define CSR_GTID                    12'hCC2
`define CSR_LWID                    12'hCC3
`define CSR_GWID                    `CSR_MHARTID
`define CSR_GCID                    12'hCC5
`define CSR_TMASK                   12'hCC4
`define CSR_NT                      12'hFC0
`define CSR_NW                      12'hFC1
`define CSR_NC                      12'hFC2

// Texture Units //////////////////////////////////////////////////////////////

`define TEX_STAGE_COUNT             2
`define TEX_SUBPIXEL_BITS           8

`define TEX_DIM_BITS                15
`define TEX_LOD_MAX                 `TEX_DIM_BITS
`define TEX_LOD_BITS                4

`define TEX_FXD_BITS                32
`define TEX_FXD_FRAC                (`TEX_DIM_BITS+`TEX_SUBPIXEL_BITS)

`define TEX_STATE_ADDR              0
`define TEX_STATE_WIDTH             1
`define TEX_STATE_HEIGHT            2
`define TEX_STATE_FORMAT            3
`define TEX_STATE_FILTER            4
`define TEX_STATE_WRAPU             5
`define TEX_STATE_WRAPV             6
`define TEX_STATE_MIPOFF(lod)       (7+(lod))
`define TEX_STATE_COUNT             (`TEX_STATE_MIPOFF(`TEX_LOD_MAX)+1)

`define CSR_TEX_STATE_BEGIN         12'h7C0
`define CSR_TEX_STAGE               `CSR_TEX_STATE_BEGIN
`define CSR_TEX_ADDR                (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_ADDR)
`define CSR_TEX_WIDTH               (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_WIDTH)
`define CSR_TEX_HEIGHT              (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_HEIGHT)
`define CSR_TEX_FORMAT              (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_FORMAT)
`define CSR_TEX_FILTER              (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_FILTER)
`define CSR_TEX_WRAPU               (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_WRAPU)
`define CSR_TEX_WRAPV               (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_WRAPV)
`define CSR_TEX_MIPOFF(lod)         (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_MIPOFF(lod))
`define CSR_TEX_STATE_END           (`CSR_TEX_STATE_BEGIN+1+`TEX_STATE_COUNT)

`define CSR_TEX_STATE(addr)         ((addr) - `CSR_TEX_ADDR)

// Raster Units ///////////////////////////////////////////////////////////////

`define RASTER_STATE_PIDX_ADDR      0
`define RASTER_STATE_PIDX_SIZE      1
`define RASTER_STATE_PBUF_ADDR      2
`define RASTER_STATE_PBUF_STRIDE    3
`define RASTER_STATE_TILE_XY        4
`define RASTER_STATE_TILE_WH        5
`define RASTER_STATE_COUNT          6

`define CSR_RASTER_STATE_BEGIN      `CSR_TEX_STATE_END
`define CSR_RASTER_PIDX_ADDR        (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_PIDX_ADDR)
`define CSR_RASTER_PIDX_SIZE        (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_PIDX_SIZE)
`define CSR_RASTER_PBUF_ADDR        (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_PBUF_ADDR)
`define CSR_RASTER_PBUF_STRIDE      (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_PBUF_STRIDE)
`define CSR_RASTER_TILE_XY          (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_TILE_XY)
`define CSR_RASTER_TILE_WH          (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_TILE_WH)
`define CSR_RASTER_STATE_END        (`CSR_RASTER_STATE_BEGIN+`RASTER_STATE_COUNT)

`define CSR_RASTER_STATE(addr)      ((addr) - `CSR_RASTER_STATE_BEGIN)

// Render Output Units ////////////////////////////////////////////////////////

`define ROP_STATE_ZBUF_ADDR         0
`define ROP_STATE_ZBUF_PITCH        1
`define ROP_STATE_CBUF_ADDR         2
`define ROP_STATE_CBUF_PITCH        3
`define ROP_STATE_ZFUNC             4
`define ROP_STATE_SFUNC             5
`define ROP_STATE_ZPASS             6
`define ROP_STATE_ZFAIL             7
`define ROP_STATE_SFAIL             8
`define ROP_STATE_BLEND_MODE        9
`define ROP_STATE_BLEND_RGB         10
`define ROP_STATE_BLEND_ALPHA       11
`define ROP_STATE_BLEND_CONST       12
`define ROP_STATE_LOGIC_OP          13
`define ROP_STATE_COUNT             14

`define CSR_ROP_STATE_BEGIN         `CSR_RASTER_STATE_END
`define CSR_ROP_ZBUF_ADDR           (`CSR_ROP_STATE_BEGIN+`ROP_STATE_ZBUF_ADDR)
`define CSR_ROP_ZBUF_PITCH          (`CSR_ROP_STATE_BEGIN+`ROP_STATE_ZBUF_PITCH)
`define CSR_ROP_CBUF_ADDR           (`CSR_ROP_STATE_BEGIN+`ROP_STATE_CBUF_ADDR)
`define CSR_ROP_CBUF_PITCH          (`CSR_ROP_STATE_BEGIN+`ROP_STATE_CBUF_PITCH)
`define CSR_ROP_ZFUNC               (`CSR_ROP_STATE_BEGIN+`ROP_STATE_ZFUNC)
`define CSR_ROP_SFUNC               (`CSR_ROP_STATE_BEGIN+`ROP_STATE_SFUNC)
`define CSR_ROP_ZPASS               (`CSR_ROP_STATE_BEGIN+`ROP_STATE_ZPASS)
`define CSR_ROP_ZFAIL               (`CSR_ROP_STATE_BEGIN+`ROP_STATE_ZFAIL)
`define CSR_ROP_SFAIL               (`CSR_ROP_STATE_BEGIN+`ROP_STATE_SFAIL)
`define CSR_ROP_BLEND_MODE          (`CSR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_MODE)
`define CSR_ROP_BLEND_RGB           (`CSR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_RGB)
`define CSR_ROP_BLEND_APLHA         (`CSR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_ALPHA)
`define CSR_ROP_BLEND_CONST         (`CSR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_CONST)
`define CSR_ROP_LOGIC_OP            (`CSR_ROP_STATE_BEGIN+`ROP_STATE_LOGIC_OP)
`define CSR_ROP_STATE_END           (`CSR_ROP_STATE_BEGIN+`ROP_STATE_COUNT)

`define CSR_ROP_STATE(addr)         ((addr) - `CSR_ROP_STATE_BEGIN)

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

// Tcache Configurable Knobs //////////////////////////////////////////////////

// Size of cache in bytes
`ifndef TCACHE_SIZE
`define TCACHE_SIZE 16384
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

// SM Configurable Knobs //////////////////////////////////////////////////////

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

`endif