`ifndef VX_DEFINE
`define VX_DEFINE

`include "VX_config.vh"

// `define QUEUE_FORCE_MLAB 1
// `define SYN 1
// `define ASIC 1
// `define SYN_FUNC 1

///////////////////////////////////////////////////////////////////////////////

`define DEBUG_BEGIN /* verilator lint_off UNUSED */ 

`define DEBUG_END   /* verilator lint_on UNUSED */     

`define IGNORE_WARNINGS_BEGIN /* verilator lint_off UNUSED */ \
                              /* verilator lint_off PINCONNECTEMPTY */ \
                              /* verilator lint_off DECLFILENAME */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on DECLFILENAME */

`define STRINGIFY(x) `"x`"

`define STATIC_ASSERT(cond, msg)    \
    generate                        \
        if (!(cond)) $error(msg);   \
    endgenerate

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > x) ? 1 : 0))
`define LOG2UP(x)   ((x > 1) ? $clog2(x) : 1)

`define MIN(x, y)   ((x < y)  ? x : y);
`define MAX(x, y)   ((x > y)  ? x : y);

///////////////////////////////////////////////////////////////////////////////

`define NW_BITS (`LOG2UP(`NUM_WARPS))

`define NT_BITS (`LOG2UP(`NUM_THREADS))

`define NC_BITS (`LOG2UP(`NUM_CORES))

`define NUM_GPRS 32

`define CSR_ADDR_SIZE 12

`define CSR_WIDTH 12

///////////////////////////////////////////////////////////////////////////////

`define INST_R      7'd051
`define INST_L      7'd003
`define INST_ALU    7'd019
`define INST_S      7'd035
`define INST_B      7'd099
`define INST_LUI    7'd055
`define INST_AUIPC  7'd023
`define INST_JAL    7'd111
`define INST_JALR   7'd103
`define INST_SYS    7'd115
`define INST_GPGPU  7'h06b

`define RS2_IMMED   1
`define RS2_REG     0

`define BR_NO       3'h0
`define BR_EQ       3'h1
`define BR_NE       3'h2
`define BR_LT       3'h3
`define BR_GT       3'h4
`define BR_LTU      3'h5
`define BR_GTU      3'h6

`define ALU_NO      5'd15
`define ALU_ADD     5'd00
`define ALU_SUB     5'd01
`define ALU_SLLA    5'd02
`define ALU_SLT     5'd03
`define ALU_SLTU    5'd04
`define ALU_XOR     5'd05
`define ALU_SRL     5'd06
`define ALU_SRA     5'd07
`define ALU_OR      5'd08
`define ALU_AND     5'd09
`define ALU_SUBU    5'd10
`define ALU_LUI     5'd11
`define ALU_AUIPC   5'd12
`define ALU_CSR_RW  5'd13
`define ALU_CSR_RS  5'd14
`define ALU_CSR_RC  5'd15
`define ALU_MUL     5'd16
`define ALU_MULH    5'd17
`define ALU_MULHSU  5'd18
`define ALU_MULHU   5'd19
`define ALU_DIV     5'd20
`define ALU_DIVU    5'd21
`define ALU_REM     5'd22
`define ALU_REMU    5'd23

`define WB_NO       2'h0
`define WB_ALU      2'h1
`define WB_MEM      2'h2
`define WB_JAL      2'h3

///////////////////////////////////////////////////////////////////////////////

// Core request tag width    pc,  wb, rd,  warp_num
`define CORE_REQ_TAG_WIDTH  (32 + 2 + 5 + `NW_BITS)

// TAG sharing enable        rd,  warp_num
`define CORE_TAG_ID_BITS    (5 + `NW_BITS)

////////////////////////// Dcache Configurable Knobs //////////////////////////

// DRAM request data bits
`define DDRAM_LINE_WIDTH    (`DBANK_LINE_SIZE * 8)

// DRAM request address bits
`define DDRAM_ADDR_WIDTH    (32 - `CLOG2(`DBANK_LINE_SIZE))

// DRAM request tag bits
`define DDRAM_TAG_WIDTH     `DDRAM_ADDR_WIDTH

////////////////////////// Icache Configurable Knobs //////////////////////////

// DRAM request data bits
`define IDRAM_LINE_WIDTH    (`IBANK_LINE_SIZE * 8)

// DRAM request address bits
`define IDRAM_ADDR_WIDTH    (32 - `CLOG2(`IBANK_LINE_SIZE))

// DRAM request tag bits
`define IDRAM_TAG_WIDTH     `IDRAM_ADDR_WIDTH

////////////////////////// SM Configurable Knobs //////////////////////////////

// DRAM request data bits
`define SDRAM_LINE_WIDTH    (`SBANK_LINE_SIZE * 8)

// DRAM request address bits
`define SDRAM_ADDR_WIDTH    (32 - `CLOG2(`SBANK_LINE_SIZE))

// DRAM request tag bits
`define SDRAM_TAG_WIDTH     `SDRAM_ADDR_WIDTH

////////////////////////// L2cache Configurable Knobs /////////////////////////

// DRAM request data bits
`define L2DRAM_LINE_WIDTH   (`L2BANK_LINE_SIZE * 8)

// DRAM request address bits
`define L2DRAM_ADDR_WIDTH   (32 - `CLOG2(`L2BANK_LINE_SIZE))

// DRAM request tag bits
`define L2DRAM_TAG_WIDTH    (`L2_ENABLE ? `L2DRAM_ADDR_WIDTH : (`L2DRAM_ADDR_WIDTH+`CLOG2(`NUM_CORES*2)))

////////////////////////// L3cache Configurable Knobs /////////////////////////

// DRAM request data bits
`define L3DRAM_LINE_WIDTH   (`L3BANK_LINE_SIZE * 8)

// DRAM request address bits
`define L3DRAM_ADDR_WIDTH   (32 - `CLOG2(`L3BANK_LINE_SIZE))

// DRAM request tag bits
`define L3DRAM_TAG_WIDTH    ((`NUM_CLUSTERS > 1) ? `L3DRAM_ADDR_WIDTH : `L2DRAM_TAG_WIDTH)

 // VX_DEFINE
`endif
