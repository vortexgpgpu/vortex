`ifndef VX_DEFINE
`define VX_DEFINE

`include "VX_platform.vh"
`include "VX_config.vh"
`include "VX_scope.vh"

///////////////////////////////////////////////////////////////////////////////

`define QUEUE_FORCE_MLAB 1
// `define SYNTHESIS 1
// `define ASIC 1

///////////////////////////////////////////////////////////////////////////////

`define NW_BITS     `LOG2UP(`NUM_WARPS)

`define NT_BITS     `LOG2UP(`NUM_THREADS)

`define NC_BITS     `LOG2UP(`NUM_CORES)

`define NB_BITS     `LOG2UP(`NUM_BARRIERS)

`define REQS_BITS   `LOG2UP(NUM_REQUESTS)

`define NUM_REGS    32

`define NR_BITS    `LOG2UP(`NUM_REGS)

`define CSR_ADDR_SIZE 12

`define CSR_WIDTH   12

`define ISTAG_BITS  `LOG2UP(`ISSUEQ_SIZE)

`define LATENCY_IDIV 23
`define LATENCY_IMUL 2

`define LATENCY_FMULADD  2
`define LATENCY_FDIVSQRT 2
`define LATENCY_FCONV    2
`define LATENCY_FNONCOMP 1

///////////////////////////////////////////////////////////////////////////////

`define INST_LUI    7'b0110111
`define INST_AUIPC  7'b0010111
`define INST_JAL    7'b1101111
`define INST_JALR   7'b1100111
`define INST_B      7'b1100011 // branch instructions
`define INST_L      7'b0000011 // load instructions
`define INST_S      7'b0100011 // store instructions
`define INST_I      7'b0010011 // immediate instructions
`define INST_R      7'b0110011 // register instructions
`define INST_F      7'b0001111 // Fence instructions
`define INST_SYS    7'b1110011 // system instructions

`define INST_FL     7'b0000111 // float load instruction
`define INST_FS     7'b0100111 // float store  instruction
`define INST_FMADD  7'b1000011  
`define INST_FMSUB  7'b1000111
`define INST_FNMSUB 7'b1001011
`define INST_FNMADD 7'b1001111 
`define INST_FCI    7'b1010011 // float common instructions

`define INST_GPU    7'b1101011

`define BYTEEN_SB   3'h0 
`define BYTEEN_SH   3'h1
`define BYTEEN_SW   3'h2
`define BYTEEN_UB   3'h4
`define BYTEEN_UH   3'h5
`define BYTEEN_BITS 3
`define BYTEEN_TYPE(x) x[1:0]

`define BR_EQ       4'h0
`define BR_NE       4'h1
`define BR_LT       4'h2
`define BR_GE       4'h3
`define BR_LTU      4'h4 
`define BR_GEU      4'h5 
`define BR_JAL      4'h6
`define BR_JALR     4'h7
`define BR_ECALL    4'h8
`define BR_EBREAK   4'h9
`define BR_MRET     4'hA
`define BR_SRET     4'hB
`define BR_DRET     4'hC
`define BR_BITS     4

`define OP_BITS     5

`define ALU_ADD     5'h00
`define ALU_SUB     5'h01
`define ALU_SLL     5'h02
`define ALU_SRL     5'h03
`define ALU_SRA     5'h04
`define ALU_SLT     5'h05
`define ALU_SLTU    5'h06
`define ALU_XOR     5'h07
`define ALU_OR      5'h08
`define ALU_AND     5'h09
`define ALU_LUI     5'h0A
`define ALU_AUIPC   5'h0B
`define ALU_BEQ     {1'b1, `BR_EQ}
`define ALU_BNE     {1'b1, `BR_NE}
`define ALU_BLT     {1'b1, `BR_LT}
`define ALU_BGE     {1'b1, `BR_GE}
`define ALU_BLTU    {1'b1, `BR_LTU}
`define ALU_BGEU    {1'b1, `BR_GEU}
`define ALU_JAL     {1'b1, `BR_JAL}
`define ALU_JALR    {1'b1, `BR_JALR}
`define ALU_ECALL   {1'b1, `BR_ECALL}
`define ALU_EBREAK  {1'b1, `BR_EBREAK}
`define ALU_MRET    {1'b1, `BR_MRET}
`define ALU_SRET    {1'b1, `BR_SRET}
`define ALU_DRET    {1'b1, `BR_DRET}
`define ALU_OTHER   5'h1F
`define ALU_BITS    5
`define ALU_OP(x)   x[`ALU_BITS-1:0]
`define BR_OP(x)    x[`BR_BITS-1:0]
`define IS_BR_OP(x) x[4]

`define LSU_LB      {1'b0, `BYTEEN_SB}
`define LSU_LH      {1'b0, `BYTEEN_SH}
`define LSU_LW      {1'b0, `BYTEEN_SW}
`define LSU_LBU     {1'b0, `BYTEEN_UB}
`define LSU_LHU     {1'b0, `BYTEEN_UH}
`define LSU_SB      {1'b1, `BYTEEN_SB}
`define LSU_SH      {1'b1, `BYTEEN_SH}
`define LSU_SW      {1'b1, `BYTEEN_SW}
`define LSU_SBU     {1'b1, `BYTEEN_UB}
`define LSU_SHU     {1'b1, `BYTEEN_UH}
`define LSU_BITS    4
`define LSU_RW(x)   x[3]
`define LSU_BE(x)   x[2:0]

`define CSR_RW      2'h0
`define CSR_RS      2'h1
`define CSR_RC      2'h2
`define CSR_OTHER   2'h3
`define CSR_BITS    2
`define CSR_OP(x)   x[`CSR_BITS-1:0]

`define MUL_MUL     3'h0
`define MUL_MULH    3'h1
`define MUL_MULHSU  3'h2
`define MUL_MULHU   3'h3
`define MUL_DIV     3'h4
`define MUL_DIVU    3'h5
`define MUL_REM     3'h6
`define MUL_REMU    3'h7
`define MUL_BITS    3
`define MUL_OP(x)   x[`MUL_BITS-1:0]
`define IS_DIV_OP(x) x[2]

`define FPU_ADD     5'h00 
`define FPU_SUB     5'h01 
`define FPU_MUL     5'h02 
`define FPU_DIV     5'h03 
`define FPU_SQRT    5'h04 
`define FPU_MADD    5'h05 
`define FPU_MSUB    5'h06   
`define FPU_NMSUB   5'h07   
`define FPU_NMADD   5'h08 
`define FPU_SGNJ    5'h09  // FSGNJ
`define FPU_SGNJN   5'h0A  // FSGNJN
`define FPU_SGNJX   5'h0B  // FSGNJX
`define FPU_MIN     5'h0C  // FMIN.S 
`define FPU_MAX     5'h0D  // FMAX.S
`define FPU_CVTWS   5'h0E  // FCVT.W.S
`define FPU_CVTWUS  5'h0F  // FCVT.WU.S
`define FPU_CVTSW   5'h10  // FCVT.S.W
`define FPU_CVTSWU  5'h11  // FCVT.S.WU
`define FPU_MVXW    5'h12  // MOV FP from fpReg to integer reg
`define FPU_MVWX    5'h13  // MOV FP from integer reg to fpReg
`define FPU_CLASS   5'h14  
`define FPU_CMP     5'h15
`define FPU_OTHER   5'h1f
`define FPU_BITS    5
`define FPU_OP(x)   x[`FPU_BITS-1:0]

`define FRM_RNE    3'b000  // round to nearest even
`define FRM_RTZ    3'b001  // round to zero
`define FRM_RDN    3'b010  // round to -inf
`define FRM_RUP    3'b011  // round to +inf
`define FRM_RMM    3'b100  // round to nearest max magnitude
`define FRM_DYN    3'b111  // dynamic mode
`define FRM_BITS   3
`define FFG_BITS   5

`define GPU_TMC     3'h0
`define GPU_WSPAWN  3'h1 
`define GPU_SPLIT   3'h2
`define GPU_JOIN    3'h3
`define GPU_BAR     3'h4
`define GPU_OTHER   3'h7
`define GPU_BITS    3
`define GPU_OP(x)   x[`GPU_BITS-1:0]

`define EX_NOP      3'h0
`define EX_ALU      3'h1
`define EX_LSU      3'h2
`define EX_CSR      3'h3
`define EX_MUL      3'h4
`define EX_FPU      3'h5
`define EX_GPU      3'h6
`define EX_BITS     3

`define NUM_EXS     6
`define NE_BITS     `LOG2UP(`NUM_EXS)

///////////////////////////////////////////////////////////////////////////////

`ifdef EXT_M_ENABLE
    `define ISA_EXT_M (1 << 12)
`else
    `define ISA_EXT_M 0
`endif

`ifdef EXT_F_ENABLE
    `define ISA_EXT_F (1 << 5)
`else
    `define ISA_EXT_F 0
`endif

`define ISA_CODE  (0 <<  0) // A - Atomic Instructions extension \
                | (0 <<  1) // B - Tentatively reserved for Bit operations extension  \
                | (0 <<  2) // C - Compressed extension \
                | (0 <<  3) // D - Double precsision floating-point extension \
                | (0 <<  4) // E - RV32E base ISA \
                |`ISA_EXT_F // F - Single precsision floating-point extension \
                | (0 <<  6) // G - Additional standard extensions present \
                | (0 <<  7) // H - Hypervisor mode implemented \
                | (1 <<  8) // I - RV32I/64I/128I base ISA \
                | (0 <<  9) // J - Reserved \
                | (0 << 10) // K - Reserved \
                | (0 << 11) // L - Tentatively reserved for Bit operations extension \
                |`ISA_EXT_M // M - Integer Multiply/Divide extension \
                | (0 << 13) // N - User level interrupts supported \
                | (0 << 14) // O - Reserved \
                | (0 << 15) // P - Tentatively reserved for Packed-SIMD extension \
                | (0 << 16) // Q - Quad-precision floating-point extension \
                | (0 << 17) // R - Reserved \
                | (0 << 18) // S - Supervisor mode implemented \
                | (0 << 19) // T - Tentatively reserved for Transactional Memory extension \
                | (1 << 20) // U - User mode implemented \
                | (0 << 21) // V - Tentatively reserved for Vector extension \
                | (0 << 22) // W - Reserved \
                | (1 << 23) // X - Non-standard extensions present \
                | (0 << 24) // Y - Reserved \
                | (0 << 25) // Z - Reserved

///////////////////////////////////////////////////////////////////////////////

`ifdef DBG_CORE_REQ_INFO          // pc,  wb, rd,        warp_num
`define DEBUG_CORE_REQ_MDATA_WIDTH  (32 + 1 + `NR_BITS + `NW_BITS)
`else
`define DEBUG_CORE_REQ_MDATA_WIDTH  0
`endif

////////////////////////// Dcache Configurable Knobs //////////////////////////

// Cache ID
`define DCACHE_ID           (((`L3_ENABLE && `L2_ENABLE) ? 2 : `L2_ENABLE ? 1 : 0) + (CORE_ID * 3) + 0)

// TAG sharing enable       
`define DCORE_TAG_ID_BITS   `ISTAG_BITS

// Core request tag bits
`define DCORE_TAG_WIDTH     (`DEBUG_CORE_REQ_MDATA_WIDTH + `DCORE_TAG_ID_BITS)
 
// DRAM request data bits
`define DDRAM_LINE_WIDTH    (`DBANK_LINE_SIZE * 8)

// DRAM request address bits
`define DDRAM_ADDR_WIDTH    (32 - `CLOG2(`DBANK_LINE_SIZE))

// DRAM byte enable bits
`define DDRAM_BYTEEN_WIDTH  `DBANK_LINE_SIZE

// DRAM request tag bits
`define DDRAM_TAG_WIDTH     `DDRAM_ADDR_WIDTH

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define DNUM_REQUESTS       `NUM_THREADS

// Snoop request tag bits
`define DSNP_TAG_WIDTH      ((`NUM_CORES > 1) ? `LOG2UP(`L2SNRQ_SIZE) : `L2SNP_TAG_WIDTH)

////////////////////////// Icache Configurable Knobs //////////////////////////

// Cache ID
`define ICACHE_ID           (((`L3_ENABLE && `L2_ENABLE) ? 2 : `L2_ENABLE ? 1 : 0) + (CORE_ID * 3) + 1)

// Core request address bits
`define ICORE_ADDR_WIDTH    (32-`CLOG2(`IWORD_SIZE))

// Core request byte enable bits
`define ICORE_BYTEEN_WIDTH `DWORD_SIZE

// TAG sharing enable       
`define ICORE_TAG_ID_BITS   `NW_BITS

// Core request tag bits
`define ICORE_TAG_WIDTH     (`DEBUG_CORE_REQ_MDATA_WIDTH + `ICORE_TAG_ID_BITS)

// DRAM request data bits
`define IDRAM_LINE_WIDTH    (`IBANK_LINE_SIZE * 8)

// DRAM request address bits
`define IDRAM_ADDR_WIDTH    (32 - `CLOG2(`IBANK_LINE_SIZE))

// DRAM byte enable bits
`define IDRAM_BYTEEN_WIDTH  `IBANK_LINE_SIZE

// DRAM request tag bits
`define IDRAM_TAG_WIDTH     `IDRAM_ADDR_WIDTH

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define INUM_REQUESTS       1

////////////////////////// SM Configurable Knobs //////////////////////////////

// Cache ID
`define SCACHE_ID           (((`L3_ENABLE && `L2_ENABLE) ? 2 : `L2_ENABLE ? 1 : 0) + (CORE_ID * 3) + 2)

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define SNUM_REQUESTS       `NUM_THREADS

// DRAM request address bits
`define SDRAM_ADDR_WIDTH    (32 - `CLOG2(`SBANK_LINE_SIZE))

// DRAM request tag bits
`define SDRAM_TAG_WIDTH     `SDRAM_ADDR_WIDTH

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define SNUM_REQUESTS       `NUM_THREADS

////////////////////////// L2cache Configurable Knobs /////////////////////////

// Cache ID
`define L2CACHE_ID          (`L3_ENABLE ? 1 : 0)

// Core request tag bits
`define L2CORE_TAG_WIDTH    (`DCORE_TAG_WIDTH + `CLOG2(`NUM_CORES))

// DRAM request data bits
`define L2DRAM_LINE_WIDTH   (`L2_ENABLE ? (`L2BANK_LINE_SIZE * 8) : `DDRAM_LINE_WIDTH)

// DRAM request address bits
`define L2DRAM_ADDR_WIDTH   (`L2_ENABLE ? (32 - `CLOG2(`L2BANK_LINE_SIZE)) : `DDRAM_ADDR_WIDTH)

// DRAM byte enable bits
`define L2DRAM_BYTEEN_WIDTH (`L2_ENABLE ? `L2BANK_LINE_SIZE : `DDRAM_BYTEEN_WIDTH)

// DRAM request tag bits
`define L2DRAM_TAG_WIDTH    (`L2_ENABLE ? `L2DRAM_ADDR_WIDTH : (`L2DRAM_ADDR_WIDTH+`CLOG2(`NUM_CORES*2)))

// Snoop request tag bits
`define L2SNP_TAG_WIDTH     (`L3_ENABLE ? `LOG2UP(`L3SNRQ_SIZE) : `L3SNP_TAG_WIDTH)

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define L2NUM_REQUESTS      (2 * `NUM_CORES)

////////////////////////// L3cache Configurable Knobs /////////////////////////

// Cache ID
`define L3CACHE_ID          0

// Core request tag bits
`define L3CORE_TAG_WIDTH    (`L2CORE_TAG_WIDTH + `CLOG2(`NUM_CLUSTERS))

// DRAM request data bits
`define L3DRAM_LINE_WIDTH   (`L3_ENABLE ? (`L3BANK_LINE_SIZE * 8) : `L2DRAM_LINE_WIDTH)

// DRAM request address bits
`define L3DRAM_ADDR_WIDTH   (`L3_ENABLE ? (32 - `CLOG2(`L3BANK_LINE_SIZE)) : `L2DRAM_ADDR_WIDTH)

// DRAM byte enable bits
`define L3DRAM_BYTEEN_WIDTH (`L3_ENABLE ? `L3BANK_LINE_SIZE : `L2DRAM_BYTEEN_WIDTH)

// DRAM request tag bits
`define L3DRAM_TAG_WIDTH    (`L3_ENABLE ? `L3DRAM_ADDR_WIDTH : `L2DRAM_TAG_WIDTH)

// Snoop request tag bits
`define L3SNP_TAG_WIDTH     16 

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`define L3NUM_REQUESTS      `NUM_CLUSTERS

///////////////////////////////////////////////////////////////////////////////

`define VX_DRAM_BYTEEN_WIDTH    `L3DRAM_BYTEEN_WIDTH   
`define VX_DRAM_ADDR_WIDTH      `L3DRAM_ADDR_WIDTH
`define VX_DRAM_LINE_WIDTH      `L3DRAM_LINE_WIDTH
`define VX_DRAM_TAG_WIDTH       `L3DRAM_TAG_WIDTH
`define VX_SNP_TAG_WIDTH        `L3SNP_TAG_WIDTH    
`define VX_CORE_TAG_WIDTH       `L3CORE_TAG_WIDTH 
`define VX_CSR_ID_WIDTH         `LOG2UP(`NUM_CLUSTERS * `NUM_CORES)

`define DRAM_TO_BYTE_ADDR(x)     {x, (32-$bits(x))'(0)}

///////////////////////////////////////////////////////////////////////////////

typedef struct packed {
    logic [`NW_BITS-1:0]    warp_num;
    logic [`NUM_THREADS-1:0] thread_mask;
    logic [31:0]            curr_PC;
    logic [`NR_BITS-1:0]    rd;
    logic                   rd_is_fp;
    logic                   wb;
} is_data_t;

`endif
