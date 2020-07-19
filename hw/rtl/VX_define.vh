`ifndef VX_DEFINE
`define VX_DEFINE

`include "VX_config.vh"
`include "VX_scope.vh"

`define QUEUE_FORCE_MLAB 1

// `define SYNTHESIS 1
// `define ASIC 1

///////////////////////////////////////////////////////////////////////////////

`ifndef NDEBUG
    `define DEBUG_BLOCK(x) /* verilator lint_off UNUSED */ \
                           x \
                           /* verilator lint_on UNUSED */
`else
    `define DEBUG_BLOCK(x)
`endif

`define DEBUG_BEGIN /* verilator lint_off UNUSED */ 

`define DEBUG_END   /* verilator lint_on UNUSED */     

`define IGNORE_WARNINGS_BEGIN /* verilator lint_off UNUSED */ \
                              /* verilator lint_off PINCONNECTEMPTY */ \
                              /* verilator lint_off DECLFILENAME */

`define IGNORE_WARNINGS_END   /* verilator lint_on UNUSED */ \
                              /* verilator lint_on PINCONNECTEMPTY */ \
                              /* verilator lint_on DECLFILENAME */

`define UNUSED_VAR(x) /* verilator lint_off UNUSED */ \
                      wire [$bits(x)-1:0] __``x``__ = x; \
                      /* verilator lint_on UNUSED */

`define UNUSED_PIN(x)  /* verilator lint_off PINCONNECTEMPTY */ \
                       . x () \
                       /* verilator lint_on PINCONNECTEMPTY */

`define STRINGIFY(x) `"x`"

`define STATIC_ASSERT(cond, msg)    \
    generate                        \
        if (!(cond)) $error(msg);   \
    endgenerate

`define CLOG2(x)    $clog2(x)
`define FLOG2(x)    ($clog2(x) - (((1 << $clog2(x)) > (x)) ? 1 : 0))
`define LOG2UP(x)   (((x) > 1) ? $clog2(x) : 1)
`define ISPOW2(x)   (((x) != 0) && (0 == ((x) & ((x) - 1))))

`define MIN(x, y)   ((x < y) ? (x) : (y))
`define MAX(x, y)   ((x > y) ? (x) : (y))

`define UP(x)       (((x) > 0) ? x : 1)

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

`define DIV_LATENCY 2

`define MUL_LATENCY 2

///////////////////////////////////////////////////////////////////////////////

`define INST_LUI    7'b0110111
`define INST_AUIPC  7'b0010111
`define INST_JAL    7'b1101111
`define INST_JALR   7'b1100111
`define INST_B      7'b1100011
`define INST_L      7'b0000011
`define INST_S      7'b0100011
`define INST_I      7'b0010011
`define INST_R      7'b0110011
`define INST_F      7'b0001111
`define INST_SYS    7'b1110011
`define INST_GPU    7'b1101011

`define OP_BITS     4

`define ALU_ADD     4'h0
`define ALU_SUB     4'h1
`define ALU_SLL     4'h2
`define ALU_SRL     4'h3
`define ALU_SRA     4'h4
`define ALU_SLT     4'h5
`define ALU_SLTU    4'h6
`define ALU_XOR     4'h7
`define ALU_OR      4'h8
`define ALU_AND     4'h9
`define ALU_LUI     4'hA
`define ALU_AUIPC   4'hB
`define ALU_OTHER   4'hF
`define ALU_BITS    4
`define ALU_OP(x)   x[`ALU_BITS-1:0]

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
`define BR_OTHER    4'hF
`define BR_BITS     4
`define BR_OP(x)    x[`BR_BITS-1:0]

`define BYTEEN_SB   3'h0 
`define BYTEEN_SH   3'h1
`define BYTEEN_SW   3'h2
`define BYTEEN_UB   3'h4
`define BYTEEN_UH   3'h5
`define BYTEEN_BITS 3
`define LSU_BITS    4
`define LSU_RW(x)   x[3]
`define LSU_BE(x)   x[2:0]

`define CSR_RW      2'h0
`define CSR_RS      2'h1
`define CSR_RC      2'h2
`define CSR_OTHER   2'h3
`define CSR_BITS    2
`define CSR_OP(x)   x[`CSR_BITS-1:0]

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
`define EX_BR       3'h2
`define EX_MUL      3'h3
`define EX_LSU      3'h4
`define EX_FPU      3'h5
`define EX_CSR      3'h6
`define EX_GPU      3'h7
`define EX_BITS     3

`define WB_NO       2'h0
`define WB_ALU      2'h1
`define WB_MEM      2'h2
`define WB_JAL      2'h3
`define WB_BITS     2

///////////////////////////////////////////////////////////////////////////////

`define ISA_CODE  (0 <<  0) // A - Atomic Instructions extension \
                | (0 <<  1) // B - Tentatively reserved for Bit operations extension  \
                | (0 <<  2) // C - Compressed extension \
                | (0 <<  3) // D - Double precsision floating-point extension \
                | (0 <<  4) // E - RV32E base ISA \
                | (0 <<  5) // F - Single precsision floating-point extension \
                | (0 <<  6) // G - Additional standard extensions present \
                | (0 <<  7) // H - Hypervisor mode implemented \
                | (1 <<  8) // I - RV32I/64I/128I base ISA \
                | (0 <<  9) // J - Reserved \
                | (0 << 10) // K - Reserved \
                | (0 << 11) // L - Tentatively reserved for Bit operations extension \
                | (1 << 12) // M - Integer Multiply/Divide extension \
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

`ifdef DBG_CORE_REQ_INFO          // pc,  wb,        rd,        warp_num
`define DEBUG_CORE_REQ_MDATA_WIDTH  (32 + `WB_BITS + `NR_BITS + `NW_BITS)
`else
`define DEBUG_CORE_REQ_MDATA_WIDTH  0
`endif

////////////////////////// Dcache Configurable Knobs //////////////////////////

// Cache ID
`define DCACHE_ID           (((`L3_ENABLE && `L2_ENABLE) ? 2 : `L2_ENABLE ? 1 : 0) + (CORE_ID * 3) + 0)

// TAG sharing enable       
`define DCORE_TAG_ID_BITS   `LOG2UP(`DCREQ_SIZE)

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
`define ICORE_TAG_ID_BITS   `LOG2UP(`ICREQ_SIZE)

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

task print_ex_type;
    input [`EX_BITS-1:0] ex;
    begin     
        case (ex)
            `EX_ALU: $write("ALU");
            `EX_BR:  $write("BR");            
            `EX_LSU: $write("LSU");
            `EX_CSR: $write("CSR");
            `EX_MUL: $write("MUL");
            `EX_FPU: $write("FPU");
            `EX_GPU: $write("GPU");
            default: $write("NOP");
        endcase
    end      
endtask

task print_instr_op;
  input [`EX_BITS-1:0] ex;
  input [`OP_BITS-1:0] op;
  begin
      case (ex)
        `EX_ALU: begin
            case (`ALU_BITS'(op))
                `ALU_ADD:   $write("ADD");
                `ALU_SUB:   $write("SUB");
                `ALU_SLL:   $write("SLL");
                `ALU_SRL:   $write("SRL");
                `ALU_SRA:   $write("SRA");
                `ALU_SLT:   $write("SLT");
                `ALU_SLTU:  $write("SLTU");
                `ALU_XOR:   $write("XOR");
                `ALU_OR:    $write("OR");
                `ALU_AND:   $write("AND");
                `ALU_LUI:   $write("LUI");
                `ALU_AUIPC: $write("AUIPC");
                default:    $write("?");
            endcase
        end
        `EX_BR: begin
            case (`BR_BITS'(op))
                `BR_EQ:     $write("EQ");
                `BR_NE:     $write("NE");
                `BR_LT:     $write("LT");
                `BR_GE:     $write("GE");
                `BR_LTU:    $write("LTU");
                `BR_GEU:    $write("GEU");           
                `BR_JAL:    $write("JAL");
                `BR_JALR:   $write("JALR");
                `BR_ECALL:  $write("ECALL");
                `BR_EBREAK: $write("EBREAK");    
                `BR_MRET:   $write("MRET");    
                `BR_SRET:   $write("SRET");    
                `BR_DRET:   $write("DRET");    
                default:    $write("?");              
            endcase
        end 
        `EX_MUL: begin
            case (`MUL_BITS'(op))
                `MUL_MUL:   $write("MUL");
                `MUL_MULH:  $write("MULH");
                `MUL_MULHSU: $write("MULHSU");
                `MUL_MULHU: $write("MULHU");
                `MUL_DIV:   $write("DIV");
                `MUL_DIVU:  $write("DIVU");
                `MUL_REM:   $write("REM");
                `MUL_REMU:  $write("REMU");
                default:    $write("?");
            endcase
        end
        `EX_LSU: begin
            case (`LSU_BITS'(op))
                4'b0000: $write("LB");
                4'b0001: $write("LH");
                4'b0010: $write("LW");
                4'b0100: $write("LBU");
                4'b0101: $write("LHU");
                4'b1000: $write("SB");
                4'b1001: $write("SH");
                4'b1010: $write("SW");
                4'b1100: $write("SBU");
                4'b1101: $write("SHU");
                default: $write("?");
            endcase
        end
        `EX_CSR: begin
            case (`CSR_BITS'(op))
                `CSR_RW: $write("CSRW");
                `CSR_RS: $write("CSRS");
                `CSR_RC: $write("CSRC");
                default: $write("?");
            endcase
        end
        `EX_GPU: begin
            case (`GPU_BITS'(op))
                `GPU_TMC:   $write("TMC");
                `GPU_WSPAWN: $write("WSPAWN");
                `GPU_SPLIT: $write("SPLIT");
                `GPU_JOIN:  $write("JOIN");
                `GPU_BAR:   $write("BAR");
                default:    $write("?");
            endcase
        end    
        default:;    
    endcase        
  end
endtask

task print_wb;
    input [`WB_BITS-1:0] wb;
    begin     
        case (wb)
            `WB_ALU: $write("ALU");
            `WB_MEM: $write("MEM");
            `WB_JAL: $write("JAL");
            default: $write("NO");
        endcase
    end      
endtask

`endif
