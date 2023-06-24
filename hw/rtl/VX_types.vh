`ifndef VX_TYPES_VH
`define VX_TYPES_VH

// Device configuration registers

`define NUM_DCRS                    4096
`define DCR_BITS                    12

`define DCR_BASE_STATE_BEGIN        12'h001
`define DCR_BASE_STARTUP_ADDR0      12'h001
`define DCR_BASE_STARTUP_ADDR1      12'h002
`define DCR_BASE_MPM_CLASS          12'h003
`define DCR_BASE_STATE_END          12'h004

`define DCR_BASE_STATE(addr)        ((addr) - `DCR_BASE_STATE_BEGIN)
`define DCR_BASE_STATE_COUNT        (`DCR_BASE_STATE_END-`DCR_BASE_STATE_BEGIN)

// Machine Performance-monitoring counters classes

`define DCR_MPM_CLASS_NONE          0           
`define DCR_MPM_CLASS_CORE          1
`define DCR_MPM_CLASS_MEM           2
`define DCR_MPM_CLASS_TEX           3
`define DCR_MPM_CLASS_RASTER        4
`define DCR_MPM_CLASS_ROP           5

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

`define CSR_MNSTATUS                12'h744

`define CSR_MPM_BASE                12'hB00
`define CSR_MPM_BASE_H              12'hB80

// Machine Performance-monitoring core counters
// PERF: Standard
`define CSR_MCYCLE                  12'hB00
`define CSR_MCYCLE_H                12'hB80
`define CSR_MPM_RESERVED            12'hB01
`define CSR_MPM_RESERVED_H          12'hB81
`define CSR_MINSTRET                12'hB02
`define CSR_MINSTRET_H              12'hB82
// PERF: pipeline
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
// PERF: memory
`define CSR_MPM_IFETCHES            12'hB0A
`define CSR_MPM_IFETCHES_H          12'hB8A
`define CSR_MPM_LOADS               12'hB0B
`define CSR_MPM_LOADS_H             12'hB8B
`define CSR_MPM_STORES              12'hB0C
`define CSR_MPM_STORES_H            12'hB8C
`define CSR_MPM_IFETCH_LAT          12'hB0D
`define CSR_MPM_IFETCH_LAT_H        12'hB8D
`define CSR_MPM_LOAD_LAT            12'hB0E 
`define CSR_MPM_LOAD_LAT_H          12'hB8E

// Machine Performance-monitoring memory counters
// PERF: icache
`define CSR_MPM_ICACHE_READS        12'hB03     // total reads
`define CSR_MPM_ICACHE_READS_H      12'hB83
`define CSR_MPM_ICACHE_MISS_R       12'hB04     // read misses
`define CSR_MPM_ICACHE_MISS_R_H     12'hB84
// PERF: dcache
`define CSR_MPM_DCACHE_READS        12'hB05     // total reads
`define CSR_MPM_DCACHE_READS_H      12'hB85
`define CSR_MPM_DCACHE_WRITES       12'hB06     // total writes
`define CSR_MPM_DCACHE_WRITES_H     12'hB86
`define CSR_MPM_DCACHE_MISS_R       12'hB07     // read misses
`define CSR_MPM_DCACHE_MISS_R_H     12'hB87
`define CSR_MPM_DCACHE_MISS_W       12'hB08     // write misses
`define CSR_MPM_DCACHE_MISS_W_H     12'hB88
`define CSR_MPM_DCACHE_BANK_ST      12'hB09     // bank conflicts
`define CSR_MPM_DCACHE_BANK_ST_H    12'hB89
`define CSR_MPM_DCACHE_MSHR_ST      12'hB0A     // MSHR stalls
`define CSR_MPM_DCACHE_MSHR_ST_H    12'hB8A
// PERF: smem
`define CSR_MPM_SMEM_READS          12'hB0B     // memory reads
`define CSR_MPM_SMEM_READS_H        12'hB8B
`define CSR_MPM_SMEM_WRITES         12'hB0C     // memory writes
`define CSR_MPM_SMEM_WRITES_H       12'hB8C
`define CSR_MPM_SMEM_BANK_ST        12'hB0D     // bank conflicts
`define CSR_MPM_SMEM_BANK_ST_H      12'hB8D
// PERF: l2cache
`define CSR_MPM_L2CACHE_READS       12'hB0E     // total reads
`define CSR_MPM_L2CACHE_READS_H     12'hB8E
`define CSR_MPM_L2CACHE_WRITES      12'hB0F     // total writes
`define CSR_MPM_L2CACHE_WRITES_H    12'hB8F
`define CSR_MPM_L2CACHE_MISS_R      12'hB10     // read misses
`define CSR_MPM_L2CACHE_MISS_R_H    12'hB90
`define CSR_MPM_L2CACHE_MISS_W      12'hB11     // write misses
`define CSR_MPM_L2CACHE_MISS_W_H    12'hB91
`define CSR_MPM_L2CACHE_BANK_ST     12'hB12     // bank conflicts
`define CSR_MPM_L2CACHE_BANK_ST_H   12'hB92
`define CSR_MPM_L2CACHE_MSHR_ST     12'hB13     // MSHR stalls
`define CSR_MPM_L2CACHE_MSHR_ST_H   12'hB93
// PERF: l3cache
`define CSR_MPM_L3CACHE_READS       12'hB14     // total reads
`define CSR_MPM_L3CACHE_READS_H     12'hB94
`define CSR_MPM_L3CACHE_WRITES      12'hB15     // total writes
`define CSR_MPM_L3CACHE_WRITES_H    12'hB95
`define CSR_MPM_L3CACHE_MISS_R      12'hB16     // read misses
`define CSR_MPM_L3CACHE_MISS_R_H    12'hB96
`define CSR_MPM_L3CACHE_MISS_W      12'hB17     // write misses
`define CSR_MPM_L3CACHE_MISS_W_H    12'hB97
`define CSR_MPM_L3CACHE_BANK_ST     12'hB18     // bank conflicts
`define CSR_MPM_L3CACHE_BANK_ST_H   12'hB98
`define CSR_MPM_L3CACHE_MSHR_ST     12'hB19     // MSHR stalls
`define CSR_MPM_L3CACHE_MSHR_ST_H   12'hB99
// PERF: memory
`define CSR_MPM_MEM_READS           12'hB1A     // total reads
`define CSR_MPM_MEM_READS_H         12'hB9A
`define CSR_MPM_MEM_WRITES          12'hB1B     // total writes
`define CSR_MPM_MEM_WRITES_H        12'hB9B
`define CSR_MPM_MEM_LAT             12'hB1C     // memory latency
`define CSR_MPM_MEM_LAT_H           12'hB9C

// Machine Performance-monitoring texture counters
// PERF: texture unit
`define CSR_MPM_TEX_READS           12'hB03     // texture accesses
`define CSR_MPM_TEX_READS_H         12'hB83
`define CSR_MPM_TEX_LAT             12'hB04     // texture latency
`define CSR_MPM_TEX_LAT_H           12'hB84
`define CSR_MPM_TEX_STALL           12'hB05     // texture latency
`define CSR_MPM_TEX_STALL_H         12'hB85
// PERF: texture cache
`define CSR_MPM_TCACHE_READS        12'hB06     // total reads
`define CSR_MPM_TCACHE_READS_H      12'hB86
`define CSR_MPM_TCACHE_MISS_R       12'hB07     // read misses
`define CSR_MPM_TCACHE_MISS_R_H     12'hB87
`define CSR_MPM_TCACHE_BANK_ST      12'hB08     // bank stalls
`define CSR_MPM_TCACHE_BANK_ST_H    12'hB88
`define CSR_MPM_TCACHE_MSHR_ST      12'hB09     // MSHR stalls
`define CSR_MPM_TCACHE_MSHR_ST_H    12'hB89
// PERF: pipeline
`define CSR_MPM_TEX_ISSUE_ST        12'hB0A     // issue stalls
`define CSR_MPM_TEX_ISSUE_ST_H      12'hB8A

// Machine Performance-monitoring raster counters
// PERF: raster unit
`define CSR_MPM_RASTER_READS        12'hB03     // raster accesses
`define CSR_MPM_RASTER_READS_H      12'hB83
`define CSR_MPM_RASTER_LAT          12'hB04     // raster latency
`define CSR_MPM_RASTER_LAT_H        12'hB84
`define CSR_MPM_RASTER_STALL        12'hB05     // raster stall cycles
`define CSR_MPM_RASTER_STALL_H      12'hB85
// PERF: raster cache
`define CSR_MPM_RCACHE_READS        12'hB06     // total reads
`define CSR_MPM_RCACHE_READS_H      12'hB86
`define CSR_MPM_RCACHE_MISS_R       12'hB07     // read misses
`define CSR_MPM_RCACHE_MISS_R_H     12'hB87
`define CSR_MPM_RCACHE_BANK_ST      12'hB08     // bank stalls
`define CSR_MPM_RCACHE_BANK_ST_H    12'hB88
`define CSR_MPM_RCACHE_MSHR_ST      12'hB09     // MSHR stalls
`define CSR_MPM_RCACHE_MSHR_ST_H    12'hB89
// PERF: pipeline
`define CSR_MPM_RASTER_ISSUE_ST     12'hB0A     // issue stalls
`define CSR_MPM_RASTER_ISSUE_ST_H   12'hB8A

// Machine Performance-monitoring rop counters
// PERF: rop unit
`define CSR_MPM_ROP_READS           12'hB03     // rop memory reads
`define CSR_MPM_ROP_READS_H         12'hB83
`define CSR_MPM_ROP_WRITES          12'hB04     // rop memory writes
`define CSR_MPM_ROP_WRITES_H        12'hB84
`define CSR_MPM_ROP_LAT             12'hB05     // rop memory latency
`define CSR_MPM_ROP_LAT_H           12'hB85
`define CSR_MPM_ROP_STALL           12'hB06     // rop stall cycles
`define CSR_MPM_ROP_STALL_H         12'hB86
// PERF: rop cache
`define CSR_MPM_OCACHE_READS        12'hB07     // total reads
`define CSR_MPM_OCACHE_READS_H      12'hB87
`define CSR_MPM_OCACHE_WRITES       12'hB08     // total writes
`define CSR_MPM_OCACHE_WRITES_H     12'hB88
`define CSR_MPM_OCACHE_MISS_R       12'hB09     // read misses
`define CSR_MPM_OCACHE_MISS_R_H     12'hB89
`define CSR_MPM_OCACHE_MISS_W       12'hB0A     // write misses
`define CSR_MPM_OCACHE_MISS_W_H     12'hB8A
`define CSR_MPM_OCACHE_BANK_ST      12'hB0B     // bank stalls
`define CSR_MPM_OCACHE_BANK_ST_H    12'hB8B
`define CSR_MPM_OCACHE_MSHR_ST      12'hB0C     // MSHR stalls
`define CSR_MPM_OCACHE_MSHR_ST_H    12'hB8C
// PERF: pipeline
`define CSR_MPM_ROP_ISSUE_ST        12'hB0D     // issue stalls
`define CSR_MPM_ROP_ISSUE_ST_H      12'hB8D

// Machine Information Registers

`define CSR_MVENDORID               12'hF11
`define CSR_MARCHID                 12'hF12
`define CSR_MIMPID                  12'hF13
`define CSR_MHARTID                 12'hF14

// GPGU CSRs

`define CSR_THREAD_ID               12'hCC0
`define CSR_WARP_ID                 12'hCC1
`define CSR_CORE_ID                 12'hCC2
`define CSR_CLUSTER_ID              12'hCC3
`define CSR_TMASK                   12'hCC4

`define CSR_NUM_THREADS             12'hFC0
`define CSR_NUM_WARPS               12'hFC1
`define CSR_NUM_CORES               12'hFC2
`define CSR_NUM_CLUSTERS            12'hFC3

// Raster unit CSRs

`define CSR_RASTER_BEGIN            12'h7C0
`define CSR_RASTER_POS_MASK         (`CSR_RASTER_BEGIN+0)
`define CSR_RASTER_BCOORD_X0        (`CSR_RASTER_BEGIN+1)
`define CSR_RASTER_BCOORD_X1        (`CSR_RASTER_BEGIN+2)
`define CSR_RASTER_BCOORD_X2        (`CSR_RASTER_BEGIN+3)
`define CSR_RASTER_BCOORD_X3        (`CSR_RASTER_BEGIN+4)
`define CSR_RASTER_BCOORD_Y0        (`CSR_RASTER_BEGIN+5)
`define CSR_RASTER_BCOORD_Y1        (`CSR_RASTER_BEGIN+6)
`define CSR_RASTER_BCOORD_Y2        (`CSR_RASTER_BEGIN+7)
`define CSR_RASTER_BCOORD_Y3        (`CSR_RASTER_BEGIN+8)
`define CSR_RASTER_BCOORD_Z0        (`CSR_RASTER_BEGIN+9)
`define CSR_RASTER_BCOORD_Z1        (`CSR_RASTER_BEGIN+10)
`define CSR_RASTER_BCOORD_Z2        (`CSR_RASTER_BEGIN+11)
`define CSR_RASTER_BCOORD_Z3        (`CSR_RASTER_BEGIN+12)
`define CSR_RASTER_END              (`CSR_RASTER_BEGIN+13)
`define CSR_RASTER_COUNT            (`CSR_RASTER_END-`CSR_RASTER_BEGIN)

// ROP unit CSRs

`define CSR_ROP_BEGIN               `CSR_RASTER_END
`define CSR_ROP_RT_IDX              (`CSR_ROP_BEGIN+0)
`define CSR_ROP_SAMPLE_IDX          (`CSR_ROP_BEGIN+1)
`define CSR_ROP_END                 (`CSR_ROP_BEGIN+2)
`define CSR_ROP_COUNT               (`CSR_ROP_END-`CSR_ROP_BEGIN)

// Texture unit CSRs

`define CSR_TEX_BEGIN               `CSR_ROP_END
`define CSR_TEX_END                 (`CSR_TEX_BEGIN+0)
`define CSR_TEX_COUNT               (`CSR_TEX_END-`CSR_TEX_BEGIN)

// Texture Units //////////////////////////////////////////////////////////////

`define TEX_STAGE_COUNT             2
`define TEX_STAGE_BITS              1

`define TEX_SUBPIXEL_BITS           8

`define TEX_DIM_BITS                15
`define TEX_LOD_MAX                 `TEX_DIM_BITS
`define TEX_LOD_BITS                4

`define TEX_FXD_BITS                32
`define TEX_FXD_FRAC                (`TEX_DIM_BITS+`TEX_SUBPIXEL_BITS)

`define TEX_FILTER_POINT            0
`define TEX_FILTER_BILINEAR         1
`define TEX_FILTER_BITS             1

`define TEX_WRAP_CLAMP              0
`define TEX_WRAP_REPEAT             1
`define TEX_WRAP_MIRROR             2

`define TEX_FORMAT_A8R8G8B8         0
`define TEX_FORMAT_R5G6B5           1
`define TEX_FORMAT_A1R5G5B5         2
`define TEX_FORMAT_A4R4G4B4         3
`define TEX_FORMAT_A8L8             4
`define TEX_FORMAT_L8               5
`define TEX_FORMAT_A8               6

`define DCR_TEX_STATE_BEGIN         (`DCR_BASE_STATE_END)
`define DCR_TEX_STAGE               (`DCR_TEX_STATE_BEGIN+0)
`define DCR_TEX_ADDR                (`DCR_TEX_STATE_BEGIN+1)
`define DCR_TEX_LOGDIM              (`DCR_TEX_STATE_BEGIN+2)
`define DCR_TEX_FORMAT              (`DCR_TEX_STATE_BEGIN+3)
`define DCR_TEX_FILTER              (`DCR_TEX_STATE_BEGIN+4)
`define DCR_TEX_WRAP                (`DCR_TEX_STATE_BEGIN+5)
`define DCR_TEX_MIPOFF(lod)         (`DCR_TEX_STATE_BEGIN+6+lod)
`define DCR_TEX_STATE_END           (`DCR_TEX_MIPOFF(`TEX_LOD_MAX)+1)

`define DCR_TEX_STATE(addr)         ((addr) - `DCR_TEX_STATE_BEGIN)
`define DCR_TEX_STATE_COUNT         (`DCR_TEX_STATE_END-`DCR_TEX_STATE_BEGIN)

// Raster Units ///////////////////////////////////////////////////////////////

`define RASTER_DIM_BITS             15
`define RASTER_STRIDE_BITS          16
`define RASTER_PID_BITS             16
`define RASTER_TILECNT_BITS         (2 * (`RASTER_DIM_BITS - `RASTER_TILE_LOGSIZE) + 1)

`define DCR_RASTER_STATE_BEGIN      `DCR_TEX_STATE_END
`define DCR_RASTER_TBUF_ADDR        (`DCR_RASTER_STATE_BEGIN+0)
`define DCR_RASTER_TILE_COUNT       (`DCR_RASTER_STATE_BEGIN+1)
`define DCR_RASTER_PBUF_ADDR        (`DCR_RASTER_STATE_BEGIN+2)
`define DCR_RASTER_PBUF_STRIDE      (`DCR_RASTER_STATE_BEGIN+3)
`define DCR_RASTER_SCISSOR_X        (`DCR_RASTER_STATE_BEGIN+4)
`define DCR_RASTER_SCISSOR_Y        (`DCR_RASTER_STATE_BEGIN+5)
`define DCR_RASTER_STATE_END        (`DCR_RASTER_STATE_BEGIN+6)

`define DCR_RASTER_STATE(addr)      ((addr) - `DCR_RASTER_STATE_BEGIN)
`define DCR_RASTER_STATE_COUNT      (`DCR_RASTER_STATE_END-`DCR_RASTER_STATE_BEGIN)

// Render Output Units ////////////////////////////////////////////////////////

`define ROP_DIM_BITS                15

`define ROP_PITCH_BITS              (`ROP_DIM_BITS + `CLOG2(4) + 1)

`define ROP_DEPTH_BITS              24 
`define ROP_DEPTH_MASK              ((1 << `ROP_DEPTH_BITS) - 1)

`define ROP_STENCIL_BITS            8
`define ROP_STENCIL_MASK            ((1 << `ROP_STENCIL_BITS) - 1)

`define ROP_DEPTH_FUNC_ALWAYS       0
`define ROP_DEPTH_FUNC_NEVER        1
`define ROP_DEPTH_FUNC_LESS         2
`define ROP_DEPTH_FUNC_LEQUAL       3
`define ROP_DEPTH_FUNC_EQUAL        4
`define ROP_DEPTH_FUNC_GEQUAL       5
`define ROP_DEPTH_FUNC_GREATER      6
`define ROP_DEPTH_FUNC_NOTEQUAL     7
`define ROP_DEPTH_FUNC_BITS         3

`define ROP_STENCIL_OP_KEEP         0 
`define ROP_STENCIL_OP_ZERO         1
`define ROP_STENCIL_OP_REPLACE      2
`define ROP_STENCIL_OP_INCR         3
`define ROP_STENCIL_OP_DECR         4
`define ROP_STENCIL_OP_INVERT       5
`define ROP_STENCIL_OP_INCR_WRAP    6
`define ROP_STENCIL_OP_DECR_WRAP    7
`define ROP_STENCIL_OP_BITS         3

`define ROP_BLEND_MODE_ADD          0
`define ROP_BLEND_MODE_SUB          1
`define ROP_BLEND_MODE_REV_SUB      2
`define ROP_BLEND_MODE_MIN          3
`define ROP_BLEND_MODE_MAX          4
`define ROP_BLEND_MODE_LOGICOP      5
`define ROP_BLEND_MODE_BITS         3

`define ROP_BLEND_FUNC_ZERO                   0 
`define ROP_BLEND_FUNC_ONE                    1
`define ROP_BLEND_FUNC_SRC_RGB                2
`define ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB      3
`define ROP_BLEND_FUNC_DST_RGB                4
`define ROP_BLEND_FUNC_ONE_MINUS_DST_RGB      5
`define ROP_BLEND_FUNC_SRC_A                  6
`define ROP_BLEND_FUNC_ONE_MINUS_SRC_A        7
`define ROP_BLEND_FUNC_DST_A                  8
`define ROP_BLEND_FUNC_ONE_MINUS_DST_A        9
`define ROP_BLEND_FUNC_CONST_RGB              10
`define ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB    11
`define ROP_BLEND_FUNC_CONST_A                12
`define ROP_BLEND_FUNC_ONE_MINUS_CONST_A      13
`define ROP_BLEND_FUNC_ALPHA_SAT              14
`define ROP_BLEND_FUNC_BITS                   4

`define ROP_LOGIC_OP_CLEAR          0
`define ROP_LOGIC_OP_AND            1
`define ROP_LOGIC_OP_AND_REVERSE    2
`define ROP_LOGIC_OP_COPY           3
`define ROP_LOGIC_OP_AND_INVERTED   4
`define ROP_LOGIC_OP_NOOP           5
`define ROP_LOGIC_OP_XOR            6
`define ROP_LOGIC_OP_OR             7
`define ROP_LOGIC_OP_NOR            8
`define ROP_LOGIC_OP_EQUIV          9
`define ROP_LOGIC_OP_INVERT         10
`define ROP_LOGIC_OP_OR_REVERSE     11
`define ROP_LOGIC_OP_COPY_INVERTED  12
`define ROP_LOGIC_OP_OR_INVERTED    13
`define ROP_LOGIC_OP_NAND           14
`define ROP_LOGIC_OP_SET            15
`define ROP_LOGIC_OP_BITS           4

`define DCR_ROP_STATE_BEGIN         `DCR_RASTER_STATE_END
`define DCR_ROP_CBUF_ADDR           (`DCR_ROP_STATE_BEGIN+0)
`define DCR_ROP_CBUF_PITCH          (`DCR_ROP_STATE_BEGIN+1)
`define DCR_ROP_CBUF_WRITEMASK      (`DCR_ROP_STATE_BEGIN+2)
`define DCR_ROP_ZBUF_ADDR           (`DCR_ROP_STATE_BEGIN+3)
`define DCR_ROP_ZBUF_PITCH          (`DCR_ROP_STATE_BEGIN+4)
`define DCR_ROP_DEPTH_FUNC          (`DCR_ROP_STATE_BEGIN+5)
`define DCR_ROP_DEPTH_WRITEMASK     (`DCR_ROP_STATE_BEGIN+6)
`define DCR_ROP_STENCIL_FUNC        (`DCR_ROP_STATE_BEGIN+7)
`define DCR_ROP_STENCIL_ZPASS       (`DCR_ROP_STATE_BEGIN+8)
`define DCR_ROP_STENCIL_ZFAIL       (`DCR_ROP_STATE_BEGIN+9)
`define DCR_ROP_STENCIL_FAIL        (`DCR_ROP_STATE_BEGIN+10)
`define DCR_ROP_STENCIL_REF         (`DCR_ROP_STATE_BEGIN+11)
`define DCR_ROP_STENCIL_MASK        (`DCR_ROP_STATE_BEGIN+12)
`define DCR_ROP_STENCIL_WRITEMASK   (`DCR_ROP_STATE_BEGIN+13)
`define DCR_ROP_BLEND_MODE          (`DCR_ROP_STATE_BEGIN+14)
`define DCR_ROP_BLEND_FUNC          (`DCR_ROP_STATE_BEGIN+15)
`define DCR_ROP_BLEND_CONST         (`DCR_ROP_STATE_BEGIN+16)
`define DCR_ROP_LOGIC_OP            (`DCR_ROP_STATE_BEGIN+17)
`define DCR_ROP_STATE_END           (`DCR_ROP_STATE_BEGIN+18)

`define DCR_ROP_STATE(addr)         ((addr) - `DCR_ROP_STATE_BEGIN)
`define DCR_ROP_STATE_COUNT         (`DCR_ROP_STATE_END-`DCR_ROP_STATE_BEGIN)

`endif // VX_TYPES_VH
