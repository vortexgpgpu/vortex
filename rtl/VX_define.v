`ifndef VX_DEFINE
`define VX_DEFINE

`include "./VX_define_synth.v"

`define NT_M1 (`NT-1)

// NW_M1 is actually log2(NW)
`define NW_M1 (`CLOG2(`NW))

// Uncomment the below line if NW=1
// `define ONLY

// `define SYN 1
// `define ASIC 1
// `define SYN_FUNC 1

`define NUM_BARRIERS 4

`define R_INST 7'd51
`define L_INST 7'd3
`define ALU_INST 7'd19
`define S_INST 7'd35
`define B_INST 7'd99
`define LUI_INST 7'd55
`define AUIPC_INST 7'd23
`define JAL_INST 7'd111
`define JALR_INST 7'd103
`define SYS_INST 7'd115
`define GPGPU_INST 7'h6b


`define WB_ALU 2'h1
`define WB_MEM 2'h2
`define WB_JAL 2'h3
`define NO_WB  2'h0


`define RS2_IMMED 1
`define RS2_REG 0


`define NO_MEM_READ  3'h7
`define LB_MEM_READ  3'h0
`define LH_MEM_READ  3'h1
`define LW_MEM_READ  3'h2
`define LBU_MEM_READ 3'h4
`define LHU_MEM_READ 3'h5


`define NO_MEM_WRITE 3'h7
`define SB_MEM_WRITE 3'h0
`define SH_MEM_WRITE 3'h1
`define SW_MEM_WRITE 3'h2


`define NO_BRANCH 3'h0
`define BEQ 3'h1
`define BNE 3'h2
`define BLT 3'h3
`define BGT 3'h4
`define BLTU 3'h5
`define BGTU 3'h6


`define NO_ALU 5'd15
`define ADD 5'd0
`define SUB 5'd1
`define SLLA 5'd2
`define SLT 5'd3
`define SLTU 5'd4
`define XOR 5'd5
`define SRL 5'd6
`define SRA 5'd7
`define OR 5'd8
`define AND 5'd9
`define SUBU 5'd10
`define LUI_ALU 5'd11
`define AUIPC_ALU 5'd12
`define CSR_ALU_RW 5'd13
`define CSR_ALU_RS 5'd14
`define CSR_ALU_RC 5'd15
`define MUL 5'd16
`define MULH 5'd17
`define MULHSU 5'd18
`define MULHU 5'd19
`define DIV 5'd20
`define DIVU 5'd21
`define REM 5'd22
`define REMU 5'd23

// WRITEBACK
`define WB_ALU 2'h1
`define WB_MEM 2'h2
`define WB_JAL 2'h3
`define NO_WB  2'h0


// JAL
`define JUMP 1'h1
`define NO_JUMP 1'h0

// STALLS
`define STALL 1'h1
`define NO_STALL 1'h0


`define TAKEN 1'b1
`define NOT_TAKEN 1'b0


`define ZERO_REG 5'h0

`define CLOG2(x) \
   (x <= 2) ? 1 : \
   (x <= 4) ? 2 : \
   (x <= 8) ? 3 : \
   (x <= 16) ? 4 : \
   (x <= 32) ? 5 : \
   (x <= 64) ? 6 : \
   (x <= 128) ? 7 : \
   (x <= 256) ? 8 : \
   (x <= 512) ? 9 : \
   (x <= 1024) ? 10 : \
   -199


`define NUMBER_CORES (`NUMBER_CORES_PER_CLUSTER*`NUMBER_CLUSTERS)

// `define SINGLE_CORE_BENCH 0
`define GLOBAL_BLOCK_SIZE_BYTES 16
// ========================================= Dcache Configurable Knobs =========================================

// General Cache Knobs
   // Size of line inside a bank in bytes
   `define DBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
   // Number of banks {1, 2, 4, 8,...}
   `define DNUMBER_BANKS 8
   // Size of a word in bytes
   `define DWORD_SIZE_BYTES 4
   // Number of Word requests per cycle {1, 2, 4, 8, ...}
   `define DNUMBER_REQUESTS `NT
   // Number of cycles to complete stage 1 (read from memory)
   `define DSTAGE_1_CYCLES 4
   // Function ID
   `define DFUNC_ID 0

   // Bank Number of words in a line
   `define DBANK_LINE_SIZE_WORDS (`DBANK_LINE_SIZE_BYTES / `DWORD_SIZE_BYTES)
   `define DBANK_LINE_SIZE_RNG `DBANK_LINE_SIZE_WORDS-1:0
// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

   // Core Request Queue Size
   `define DREQQ_SIZE `NW
   // Miss Reserv Queue Knob
   `define DMRVQ_SIZE (`NW*`NT)
   // Dram Fill Rsp Queue Size
   `define DDFPQ_SIZE 2
   // Snoop Req Queue
   `define DSNRQ_SIZE 8

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
   // Core Writeback Queue Size
   `define DCWBQ_SIZE `DREQQ_SIZE
   // Dram Writeback Queue Size
   `define DDWBQ_SIZE 4
   // Dram Fill Req Queue Size
   `define DDFQQ_SIZE `DREQQ_SIZE
   // Lower Level Cache Hit Queue Size
   `define DLLVQ_SIZE 0
   // Fill Forward SNP Queue
   `define DFFSQ_SIZE 8

  // Fill Invalidator Size {Fill invalidator must be active}
  `define DFILL_INVALIDAOR_SIZE 16

// Dram knobs
   `define DSIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= Dcache Configurable Knobs =========================================


// ========================================= Icache Configurable Knobs =========================================

// General Cache Knobs
   // Size of line inside a bank in bytes
   `define IBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
   // Number of banks {1, 2, 4, 8,...}
   `define INUMBER_BANKS 8
   // Size of a word in bytes
   `define IWORD_SIZE_BYTES 4
   // Number of Word requests per cycle {1, 2, 4, 8, ...}
   `define INUMBER_REQUESTS 1
   // Number of cycles to complete stage 1 (read from memory)
   `define ISTAGE_1_CYCLES 4
   // Function ID
   `define IFUNC_ID 1

   // Bank Number of words in a line
   `define IBANK_LINE_SIZE_WORDS (`IBANK_LINE_SIZE_BYTES / `IWORD_SIZE_BYTES)
   `define IBANK_LINE_SIZE_RNG `IBANK_LINE_SIZE_WORDS-1:0
// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

   // Core Request Queue Size
   `define IREQQ_SIZE `NW
   // Miss Reserv Queue Knob
   `define IMRVQ_SIZE `IREQQ_SIZE
   // Dram Fill Rsp Queue Size
   `define IDFPQ_SIZE 2
   // Snoop Req Queue
   `define ISNRQ_SIZE 8

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
   // Core Writeback Queue Size
   `define ICWBQ_SIZE `IREQQ_SIZE
   // Dram Writeback Queue Size
   `define IDWBQ_SIZE 0
   // Dram Fill Req Queue Size
   `define IDFQQ_SIZE `IREQQ_SIZE
   // Lower Level Cache Hit Queue Size
   `define ILLVQ_SIZE 0
   // Fill Forward SNP Queue
   `define IFFSQ_SIZE 8

  // Fill Invalidator Size {Fill invalidator must be active}
  `define IFILL_INVALIDAOR_SIZE 16

// Dram knobs
   `define ISIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= Icache Configurable Knobs =========================================

// ========================================= SM Configurable Knobs =========================================

// General Cache Knobs
   // Size of cache in bytes
   `define SCACHE_SIZE_BYTES 1024
   // Size of line inside a bank in bytes
   `define SBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
   // Number of banks {1, 2, 4, 8,...}
   `define SNUMBER_BANKS 8
   // Size of a word in bytes
   `define SWORD_SIZE_BYTES 4
   // Number of Word requests per cycle {1, 2, 4, 8, ...}
   `define SNUMBER_REQUESTS `NT
   // Number of cycles to complete stage 1 (read from memory)
   `define SSTAGE_1_CYCLES 2
   // Function ID
   `define SFUNC_ID 2

   // Bank Number of words in a line
   `define SBANK_LINE_SIZE_WORDS (`SBANK_LINE_SIZE_BYTES / `SWORD_SIZE_BYTES)
   `define SBANK_LINE_SIZE_RNG `SBANK_LINE_SIZE_WORDS-1:0
// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

   // Core Request Queue Size
   `define SREQQ_SIZE `NW
   // Miss Reserv Queue Knob
   `define SMRVQ_SIZE `SREQQ_SIZE
   // Dram Fill Rsp Queue Size
   `define SDFPQ_SIZE 0
   // Snoop Req Queue
   `define SSNRQ_SIZE 0

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
   // Core Writeback Queue Size
   `define SCWBQ_SIZE `SREQQ_SIZE
   // Dram Writeback Queue Size
   `define SDWBQ_SIZE 0
   // Dram Fill Req Queue Size
   `define SDFQQ_SIZE 0
   // Lower Level Cache Hit Queue Size
   `define SLLVQ_SIZE 0
   // Fill Forward SNP Queue
   `define SFFSQ_SIZE 0

  // Fill Invalidator Size {Fill invalidator must be active}
  `define SFILL_INVALIDAOR_SIZE 16

// Dram knobs
   `define SSIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= SM Configurable Knobs =========================================



// ========================================= L2cache Configurable Knobs =========================================

// General Cache Knobs
   // Size of line inside a bank in bytes
   `define LLBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
   // Number of banks {1, 2, 4, 8,...}
   `define LLNUMBER_BANKS 8
   // Size of a word in bytes
   `define LLWORD_SIZE_BYTES (`LLBANK_LINE_SIZE_BYTES)
   // Number of Word requests per cycle {1, 2, 4, 8, ...}
   `define LLNUMBER_REQUESTS (2*`NUMBER_CORES_PER_CLUSTER)
   // Number of cycles to complete stage 1 (read from memory)
   `define LLSTAGE_1_CYCLES 2
   // Function ID
   `define LLFUNC_ID 3

   // Bank Number of words in a line
   `define LLBANK_LINE_SIZE_WORDS (`LLBANK_LINE_SIZE_BYTES / `LLWORD_SIZE_BYTES)
   `define LLBANK_LINE_SIZE_RNG `LLBANK_LINE_SIZE_WORDS-1:0
// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

   // Core Request Queue Size
   `define LLREQQ_SIZE (2*`NUMBER_CORES_PER_CLUSTER)
   // Miss Reserv Queue Knob
   `define LLMRVQ_SIZE (`DNUMBER_BANKS*`NUMBER_CORES_PER_CLUSTER)
   // Dram Fill Rsp Queue Size
   `define LLDFPQ_SIZE 2
   // Snoop Req Queue
   `define LLSNRQ_SIZE 8

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
   // Core Writeback Queue Size
   `define LLCWBQ_SIZE `LLREQQ_SIZE
   // Dram Writeback Queue Size
   `define LLDWBQ_SIZE 4
   // Dram Fill Req Queue Size
   `define LLDFQQ_SIZE `LLREQQ_SIZE
   // Lower Level Cache Hit Queue Size
   `define LLLLVQ_SIZE 0
   // Fill Forward SNP Queue
   `define LLFFSQ_SIZE 8

  // Fill Invalidator Size {Fill invalidator must be active}
  `define LLFILL_INVALIDAOR_SIZE 16

// Dram knobs
   `define LLSIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= L2cache Configurable Knobs =========================================


// ========================================= L3cache Configurable Knobs =========================================
// General Cache Knobs
   // Size of cache in bytes
   `define L3CACHE_SIZE_BYTES 1024
   // Size of line inside a bank in bytes
   `define L3BANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
   // Number of banks {1, 2, 4, 8,...}
   `define L3NUMBER_BANKS 8
   // Size of a word in bytes
   `define L3WORD_SIZE_BYTES (`L3BANK_LINE_SIZE_BYTES)
   // Number of Word requests per cycle {1, 2, 4, 8, ...}
   `define L3NUMBER_REQUESTS (`NUMBER_CLUSTERS)
   // Number of cycles to complete stage 1 (read from memory)
   `define L3STAGE_1_CYCLES 2
   // Function ID
   `define L3FUNC_ID 3

   // Bank Number of words in a line
   `define L3BANK_LINE_SIZE_WORDS (`L3BANK_LINE_SIZE_BYTES / `L3WORD_SIZE_BYTES)
   `define L3BANK_LINE_SIZE_RNG `L3BANK_LINE_SIZE_WORDS-1:0
// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

   // Core Request Queue Size
   `define L3REQQ_SIZE (`NT*`NW*`NUMBER_CLUSTERS)
   // Miss Reserv Queue Knob
   `define L3MRVQ_SIZE `LLREQQ_SIZE
   // Dram Fill Rsp Queue Size
   `define L3DFPQ_SIZE 2
   // Snoop Req Queue
   `define L3SNRQ_SIZE 8

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
   // Core Writeback Queue Size
   `define L3CWBQ_SIZE `L3REQQ_SIZE
   // Dram Writeback Queue Size
   `define L3DWBQ_SIZE 4
   // Dram Fill Req Queue Size
   `define L3DFQQ_SIZE `L3REQQ_SIZE
   // Lower Level Cache Hit Queue Size
   `define L3LLVQ_SIZE 0
   // Fill Forward SNP Queue
   `define L3FFSQ_SIZE 8

  // Fill Invalidator Size {Fill invalidator must be active}
  `define L3FILL_INVALIDAOR_SIZE 16

// Dram knobs
   `define L3SIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= L3cache Configurable Knobs =========================================


`endif
