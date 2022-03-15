`ifndef VX_TYPES
`define VX_TYPES

// Device configuration registers 
`define NUM_DCRS                    4096
`define DCR_BITS                    12

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

// GPGU CSRs
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

// Texture unit CSRs
`define CSR_TEX_BEGIN               12'h7C0
`define CSR_TEX_STAGE               (`CSR_TEX_BEGIN+0)
`define CSR_TEX_END                 (`CSR_TEX_BEGIN+1)

// Raster unit CSRs
`define CSR_RASTER_BEGIN            `CSR_TEX_END
`define CSR_RASTER_FRAG             (`CSR_RASTER_BEGIN+0)
`define CSR_RASTER_X_Y              (`CSR_RASTER_BEGIN+1)
`define CSR_RASTER_MASK_PID         (`CSR_RASTER_BEGIN+2)
`define CSR_RASTER_BCOORD_X         (`CSR_RASTER_BEGIN+3)
`define CSR_RASTER_BCOORD_Y         (`CSR_RASTER_BEGIN+4)
`define CSR_RASTER_BCOORD_Z         (`CSR_RASTER_BEGIN+5)
`define CSR_RASTER_GRAD_X           (`CSR_RASTER_BEGIN+6)
`define CSR_RASTER_GRAD_Y           (`CSR_RASTER_BEGIN+7)
`define CSR_RASTER_END              (`CSR_RASTER_BEGIN+8)

// ROP unit CSRs
`define CSR_ROP_BEGIN               `CSR_RASTER_END
`define CSR_ROP_DST_IDX             (`CSR_ROP_BEGIN+0)
`define CSR_ROP_DST_POS             (`CSR_ROP_BEGIN+1)
`define CSR_ROP_SAMPLE_MASK         (`CSR_ROP_BEGIN+2)
`define CSR_ROP_END                 (`CSR_ROP_BEGIN+3)

// Texture Units //////////////////////////////////////////////////////////////

`define TEX_STAGE_COUNT             2
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

`define TEX_STATE_ADDR              0
`define TEX_STATE_LOGDIM            1
`define TEX_STATE_FORMAT            2
`define TEX_STATE_FILTER            3
`define TEX_STATE_WRAP              4
`define TEX_STATE_MIPOFF(lod)       (5+(lod))
`define TEX_STATE_COUNT             (`TEX_STATE_MIPOFF(`TEX_LOD_MAX)+1)

`define DCR_TEX_STATE_BEGIN         12'h100
`define DCR_TEX_STAGE               `DCR_TEX_STATE_BEGIN
`define DCR_TEX_ADDR                (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_ADDR)
`define DCR_TEX_LOGDIM              (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_LOGDIM)
`define DCR_TEX_FORMAT              (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_FORMAT)
`define DCR_TEX_FILTER              (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_FILTER)
`define DCR_TEX_WRAP                (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_WRAP)
`define DCR_TEX_MIPOFF(lod)         (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_MIPOFF(lod))
`define DCR_TEX_STATE_END           (`DCR_TEX_STATE_BEGIN+1+`TEX_STATE_COUNT)

`define DCR_TEX_STATE(addr)         ((addr) - `DCR_TEX_ADDR)

// Raster Units ///////////////////////////////////////////////////////////////

`define RASTER_DIM_BITS             15
`define RASTER_TILE_LOGSIZE         6
`define RASTER_BLOCK_LOGSIZE        2

`define RASTER_STATE_TBUF_ADDR      0
`define RASTER_STATE_TILE_COUNT     1
`define RASTER_STATE_PBUF_ADDR      2
`define RASTER_STATE_PBUF_STRIDE    3
`define RASTER_STATE_COUNT          4

`define DCR_RASTER_STATE_BEGIN      `DCR_TEX_STATE_END
`define DCR_RASTER_TBUF_ADDR        (`DCR_RASTER_STATE_BEGIN+`RASTER_STATE_TBUF_ADDR)
`define DCR_RASTER_TILE_COUNT       (`DCR_RASTER_STATE_BEGIN+`RASTER_STATE_TILE_COUNT)
`define DCR_RASTER_PBUF_ADDR        (`DCR_RASTER_STATE_BEGIN+`RASTER_STATE_PBUF_ADDR)
`define DCR_RASTER_PBUF_STRIDE      (`DCR_RASTER_STATE_BEGIN+`RASTER_STATE_PBUF_STRIDE)
`define DCR_RASTER_STATE_END        (`DCR_RASTER_STATE_BEGIN+`RASTER_STATE_COUNT)

`define DCR_RASTER_STATE(addr)      ((addr) - `DCR_RASTER_STATE_BEGIN)

// Render Output Units ////////////////////////////////////////////////////////

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

`define ROP_BLEND_MODE_LOGICOP      0  // deprecated!
`define ROP_BLEND_MODE_ADD          1
`define ROP_BLEND_MODE_SUB          2
`define ROP_BLEND_MODE_REV_SUB      3
`define ROP_BLEND_MODE_MIN          4
`define ROP_BLEND_MODE_MAX          5
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

`define ROP_LOGIC_OP_COPY           0
`define ROP_LOGIC_OP_CLEAR          1
`define ROP_LOGIC_OP_AND            2
`define ROP_LOGIC_OP_AND_REVERSE    3
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

`define ROP_STATE_CBUF_ADDR         0
`define ROP_STATE_CBUF_PITCH        1
`define ROP_STATE_CBUF_MASK         2
`define ROP_STATE_ZBUF_ADDR         3
`define ROP_STATE_ZBUF_PITCH        4
`define ROP_STATE_DEPTH_FUNC        5
`define ROP_STATE_DEPTH_MASK        6
`define ROP_STATE_STENCIL_FUNC      7
`define ROP_STATE_STENCIL_ZPASS     8
`define ROP_STATE_STENCIL_ZFAIL     9
`define ROP_STATE_STENCIL_FAIL      10
`define ROP_STATE_STENCIL_MASK      11
`define ROP_STATE_STENCIL_REF       12
`define ROP_STATE_BLEND_MODE        13
`define ROP_STATE_BLEND_FUNC        14
`define ROP_STATE_BLEND_CONST       15
`define ROP_STATE_LOGIC_OP          16
`define ROP_STATE_COUNT             17

`define DCR_ROP_STATE_BEGIN         `DCR_RASTER_STATE_END
`define DCR_ROP_CBUF_ADDR           (`DCR_ROP_STATE_BEGIN+`ROP_STATE_CBUF_ADDR)
`define DCR_ROP_CBUF_PITCH          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_CBUF_PITCH)
`define DCR_ROP_CBUF_MASK           (`DCR_ROP_STATE_BEGIN+`ROP_STATE_CBUF_MASK)
`define DCR_ROP_ZBUF_ADDR           (`DCR_ROP_STATE_BEGIN+`ROP_STATE_ZBUF_ADDR)
`define DCR_ROP_ZBUF_PITCH          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_ZBUF_PITCH)
`define DCR_ROP_DEPTH_FUNC          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_DEPTH_FUNC)
`define DCR_ROP_DEPTH_MASK          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_DEPTH_MASK)
`define DCR_ROP_STENCIL_FUNC        (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_FUNC)
`define DCR_ROP_STENCIL_ZPASS       (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_ZPASS)
`define DCR_ROP_STENCIL_ZFAIL       (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_ZFAIL)
`define DCR_ROP_STENCIL_FAIL        (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_FAIL)
`define DCR_ROP_STENCIL_MASK        (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_MASK)
`define DCR_ROP_STENCIL_REF         (`DCR_ROP_STATE_BEGIN+`ROP_STATE_STENCIL_REF)
`define DCR_ROP_BLEND_MODE          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_MODE)
`define DCR_ROP_BLEND_FUNC          (`DCR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_FUNC)
`define DCR_ROP_BLEND_CONST         (`DCR_ROP_STATE_BEGIN+`ROP_STATE_BLEND_CONST)
`define DCR_ROP_LOGIC_OP            (`DCR_ROP_STATE_BEGIN+`ROP_STATE_LOGIC_OP)
`define DCR_ROP_STATE_END           (`DCR_ROP_STATE_BEGIN+`ROP_STATE_COUNT)

`define DCR_ROP_STATE(addr)         ((addr) - `DCR_ROP_STATE_BEGIN)

`endif