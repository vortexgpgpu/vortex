`ifndef VX_DEFINE
`define VX_DEFINE

`include "./VX_define_synth.v"

`ifndef NT
`define NT 4
`endif

`ifndef NW
`define NW 8
`endif

`ifndef NUMBER_CORES_PER_CLUSTER
`define NUMBER_CORES_PER_CLUSTER 1
`endif

`ifndef NUMBER_CLUSTERS
`define NUMBER_CLUSTERS 1
`endif

// `define QUEUE_FORCE_MLAB 1

`define NT_M1 (`NT-1)

// NW_M1 is actually log2(NW)
`define NW_M1 (`CLOG2(`NW))

// Uncomment the below line if NW=1
// `define ONLY

// `define SYN 1
// `define ASIC 1
// `define SYN_FUNC 1

`ifndef NUM_BARRIERS
`define NUM_BARRIERS 4
`endif

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


`define TAKEN 1'h1
`define NOT_TAKEN 1'h0


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


`ifndef NUMBER_CORES
`define NUMBER_CORES (`NUMBER_CORES_PER_CLUSTER*`NUMBER_CLUSTERS)
`endif

// `define SINGLE_CORE_BENCH

`ifndef GLOBAL_BLOCK_SIZE_BYTES
`define GLOBAL_BLOCK_SIZE_BYTES 16
`endif

// ========================================= Dcache Configurable Knobs =========================================

// General Cache Knobs

// Size of cache in bytes
`ifndef DCACHE_SIZE_BYTES
`define DCACHE_SIZE_BYTES 2048
`endif

// Size of line inside a bank in bytes
`ifndef DBANK_LINE_SIZE_BYTES
`define DBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef DNUMBER_BANKS
`define DNUMBER_BANKS 8
`endif

// Size of a word in bytes
`ifndef DWORD_SIZE_BYTES
`define DWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef DNUMBER_REQUESTS
`define DNUMBER_REQUESTS `NT
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef DSTAGE_1_CYCLES
`define DSTAGE_1_CYCLES 1
`endif

// Function ID
`ifndef DFUNC_ID
`define DFUNC_ID 0
`endif

// Bank Number of words in a line
`ifndef DBANK_LINE_SIZE_WORDS
`define DBANK_LINE_SIZE_WORDS (`DBANK_LINE_SIZE_BYTES / `DWORD_SIZE_BYTES)
`endif

// Bank Number of words range
`ifndef DBANK_LINE_SIZE_RNG
`define DBANK_LINE_SIZE_RNG `DBANK_LINE_SIZE_WORDS-1:0
`endif

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

// Core Request Queue Size
`ifndef DREQQ_SIZE
`define DREQQ_SIZE `NW
`endif

// Miss Reserv Queue Knob
`ifndef DMRVQ_SIZE
`define DMRVQ_SIZE (`NW*`NT)
`endif

// Dram Fill Rsp Queue Size
`ifndef DDFPQ_SIZE
`define DDFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef DSNRQ_SIZE
`define DSNRQ_SIZE 32
`endif

// Queues for writebacks Knobs {1, 2, 4, 8, ...}

// Core Writeback Queue Size
`ifndef DCWBQ_SIZE
`define DCWBQ_SIZE `DREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef DDWBQ_SIZE
`define DDWBQ_SIZE 4
`endif

// Dram Fill Req Queue Size
`ifndef DDFQQ_SIZE
`define DDFQQ_SIZE `DREQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef DLLVQ_SIZE
`define DLLVQ_SIZE 0
`endif

// Fill Forward SNP Queue
`ifndef DFFSQ_SIZE
`define DFFSQ_SIZE 32
`endif

// Prefetcher
`ifndef DPRFQ_SIZE
`define DPRFQ_SIZE 32
`endif

`ifndef DPRFQ_STRIDE
`define DPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef DFILL_INVALIDAOR_SIZE
`define DFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef DSIMULATED_DRAM_LATENCY_CYCLES
`define DSIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ========================================= Icache Configurable Knobs =========================================

// General Cache Knobs

// Size of cache in bytes
`ifndef ICACHE_SIZE_BYTES
`define ICACHE_SIZE_BYTES 4096
`endif

// Size of line inside a bank in bytes
`ifndef IBANK_LINE_SIZE_BYTES
`define IBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef INUMBER_BANKS
`define INUMBER_BANKS 8
`endif

// Size of a word in bytes
`ifndef IWORD_SIZE_BYTES
`define IWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef INUMBER_REQUESTS
`define INUMBER_REQUESTS 1
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef ISTAGE_1_CYCLES
`define ISTAGE_1_CYCLES 1
`endif

// Function ID
`ifndef IFUNC_ID
`define IFUNC_ID 1
`endif

// Bank Number of words in a line
`ifndef IBANK_LINE_SIZE_WORDS
`define IBANK_LINE_SIZE_WORDS (`IBANK_LINE_SIZE_BYTES / `IWORD_SIZE_BYTES)
`endif

// Bank Number of words range
`ifndef IBANK_LINE_SIZE_RNG
`define IBANK_LINE_SIZE_RNG `IBANK_LINE_SIZE_WORDS-1:0
`endif

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

// Core Request Queue Size
`ifndef IREQQ_SIZE
`define IREQQ_SIZE `NW
`endif

// Miss Reserv Queue Knob
`ifndef IMRVQ_SIZE
`define IMRVQ_SIZE `IREQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef IDFPQ_SIZE
`define IDFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef ISNRQ_SIZE
`define ISNRQ_SIZE 32
`endif

// Queues for writebacks Knobs {1, 2, 4, 8, ...}

// Core Writeback Queue Size
`ifndef ICWBQ_SIZE
`define ICWBQ_SIZE `IREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef IDWBQ_SIZE
`define IDWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef IDFQQ_SIZE
`define IDFQQ_SIZE `IREQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef ILLVQ_SIZE
`define ILLVQ_SIZE 16
`endif

// Fill Forward SNP Queue
`ifndef IFFSQ_SIZE
`define IFFSQ_SIZE 8
`endif

// Prefetcher
`ifndef IPRFQ_SIZE
`define IPRFQ_SIZE 32
`endif

`ifndef IPRFQ_STRIDE
`define IPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef IFILL_INVALIDAOR_SIZE
`define IFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef ISIMULATED_DRAM_LATENCY_CYCLES
`define ISIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ========================================= SM Configurable Knobs =========================================

// General Cache Knobs
// Size of cache in bytes
`ifndef SCACHE_SIZE_BYTES
`define SCACHE_SIZE_BYTES 1024
`endif

// Size of line inside a bank in bytes
`ifndef SBANK_LINE_SIZE_BYTES
`define SBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef SNUMBER_BANKS
`define SNUMBER_BANKS 8
`endif

// Size of a word in bytes
`ifndef SWORD_SIZE_BYTES
`define SWORD_SIZE_BYTES 4
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef SNUMBER_REQUESTS
`define SNUMBER_REQUESTS `NT
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef SSTAGE_1_CYCLES
`define SSTAGE_1_CYCLES 1
`endif

// Function ID
`ifndef SFUNC_ID
`define SFUNC_ID 2
`endif

// Bank Number of words in a line
`ifndef SBANK_LINE_SIZE_WORDS
`define SBANK_LINE_SIZE_WORDS (`SBANK_LINE_SIZE_BYTES / `SWORD_SIZE_BYTES)
`endif

`ifndef SBANK_LINE_SIZE_RNG
`define SBANK_LINE_SIZE_RNG `SBANK_LINE_SIZE_WORDS-1:0
`endif

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

// Core Request Queue Size
`ifndef SREQQ_SIZE
`define SREQQ_SIZE `NW
`endif

// Miss Reserv Queue Knob
`ifndef SMRVQ_SIZE
`define SMRVQ_SIZE `SREQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef SDFPQ_SIZE
`define SDFPQ_SIZE 0
`endif

// Snoop Req Queue
`ifndef SSNRQ_SIZE
`define SSNRQ_SIZE 16
`endif

// Queues for writebacks Knobs {1, 2, 4, 8, ...}

// Core Writeback Queue Size
`ifndef SCWBQ_SIZE
`define SCWBQ_SIZE `SREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef SDWBQ_SIZE
`define SDWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef SDFQQ_SIZE
`define SDFQQ_SIZE 16
`endif

// Lower Level Cache Hit Queue Size
`ifndef SLLVQ_SIZE
`define SLLVQ_SIZE 16
`endif

// Fill Forward SNP Queue
`ifndef SFFSQ_SIZE
`define SFFSQ_SIZE 16
`endif

// Prefetcher
`ifndef SPRFQ_SIZE
`define SPRFQ_SIZE 4
`endif

`ifndef SPRFQ_STRIDE
`define SPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef SFILL_INVALIDAOR_SIZE
`define SFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef SSIMULATED_DRAM_LATENCY_CYCLES
`define SSIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ========================================= L2cache Configurable Knobs =========================================

// General Cache Knobs

// Size of cache in bytes
`ifndef LLCACHE_SIZE_BYTES
`define LLCACHE_SIZE_BYTES 4096
`endif

// Size of line inside a bank in bytes
`ifndef LLBANK_LINE_SIZE_BYTES
`define LLBANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef LLNUMBER_BANKS
`define LLNUMBER_BANKS 8
`endif

// Size of a word in bytes
`ifndef LLWORD_SIZE_BYTES
`define LLWORD_SIZE_BYTES (`LLBANK_LINE_SIZE_BYTES)
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef LLNUMBER_REQUESTS
`define LLNUMBER_REQUESTS (2*`NUMBER_CORES_PER_CLUSTER)
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef LLSTAGE_1_CYCLES
`define LLSTAGE_1_CYCLES 1
`endif

// Function ID
`define LLFUNC_ID 3

// Bank Number of words in a line
`ifndef LLBANK_LINE_SIZE_WORDS
`define LLBANK_LINE_SIZE_WORDS (`LLBANK_LINE_SIZE_BYTES / `LLWORD_SIZE_BYTES)
`endif

`ifndef LLBANK_LINE_SIZE_RNG
`define LLBANK_LINE_SIZE_RNG `LLBANK_LINE_SIZE_WORDS-1:0
`endif

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

// Core Request Queue Size
`ifndef LLREQQ_SIZE
`define LLREQQ_SIZE 32
`endif

// Miss Reserv Queue Knob
`ifndef LLMRVQ_SIZE
`define LLMRVQ_SIZE 32
`endif

// Dram Fill Rsp Queue Size
`ifndef LLDFPQ_SIZE
`define LLDFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef LLSNRQ_SIZE
`define LLSNRQ_SIZE 32
`endif

// Queues for writebacks Knobs {1, 2, 4, 8, ...}

// Core Writeback Queue Size
`ifndef LLCWBQ_SIZE
`define LLCWBQ_SIZE `LLREQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef LLDWBQ_SIZE
`define LLDWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef LLDFQQ_SIZE
`define LLDFQQ_SIZE `LLREQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef LLLLVQ_SIZE
`define LLLLVQ_SIZE 32
`endif

// Fill Forward SNP Queue
`ifndef LLFFSQ_SIZE
`define LLFFSQ_SIZE 32
`endif

// Prefetcher
`ifndef LLPRFQ_SIZE
`define LLPRFQ_SIZE 32
`endif

`ifndef LLPRFQ_STRIDE
`define LLPRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef LLFILL_INVALIDAOR_SIZE
`define LLFILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef LLSIMULATED_DRAM_LATENCY_CYCLES
`define LLSIMULATED_DRAM_LATENCY_CYCLES 2
`endif

// ========================================= L3cache Configurable Knobs =========================================

// General Cache Knobs

// Size of cache in bytes
`ifndef L3CACHE_SIZE_BYTES
`define L3CACHE_SIZE_BYTES 8192
`endif

// Size of line inside a bank in bytes
`ifndef L3BANK_LINE_SIZE_BYTES
`define L3BANK_LINE_SIZE_BYTES `GLOBAL_BLOCK_SIZE_BYTES
`endif

// Number of banks {1, 2, 4, 8,...}
`ifndef L3NUMBER_BANKS
`define L3NUMBER_BANKS 8
`endif

// Size of a word in bytes
`ifndef L3WORD_SIZE_BYTES
`define L3WORD_SIZE_BYTES (`L3BANK_LINE_SIZE_BYTES)
`endif

// Number of Word requests per cycle {1, 2, 4, 8, ...}
`ifndef L3NUMBER_REQUESTS
`define L3NUMBER_REQUESTS (`NUMBER_CLUSTERS)
`endif

// Number of cycles to complete stage 1 (read from memory)
`ifndef L3STAGE_1_CYCLES
`define L3STAGE_1_CYCLES 1
`endif

// Function ID
`define L3FUNC_ID 3

// Bank Number of words in a line
`ifndef L3BANK_LINE_SIZE_WORDS
`define L3BANK_LINE_SIZE_WORDS (`L3BANK_LINE_SIZE_BYTES / `L3WORD_SIZE_BYTES)
`endif

`ifndef L3BANK_LINE_SIZE_RNG
`define L3BANK_LINE_SIZE_RNG `L3BANK_LINE_SIZE_WORDS-1:0
`endif

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

// Core Request Queue Size
`ifndef L3REQQ_SIZE
`define L3REQQ_SIZE 32
`endif

// Miss Reserv Queue Knob
`ifndef L3MRVQ_SIZE
`define L3MRVQ_SIZE `L3REQQ_SIZE
`endif

// Dram Fill Rsp Queue Size
`ifndef L3DFPQ_SIZE
`define L3DFPQ_SIZE 32
`endif

// Snoop Req Queue
`ifndef L3SNRQ_SIZE
`define L3SNRQ_SIZE 32
`endif

// Queues for writebacks Knobs {1, 2, 4, 8, ...}

// Core Writeback Queue Size
`ifndef L3CWBQ_SIZE
`define L3CWBQ_SIZE `L3REQQ_SIZE
`endif

// Dram Writeback Queue Size
`ifndef L3DWBQ_SIZE
`define L3DWBQ_SIZE 16
`endif

// Dram Fill Req Queue Size
`ifndef L3DFQQ_SIZE
`define L3DFQQ_SIZE `L3REQQ_SIZE
`endif

// Lower Level Cache Hit Queue Size
`ifndef L3LLVQ_SIZE
`define L3LLVQ_SIZE 0
`endif

// Fill Forward SNP Queue
`ifndef L3FFSQ_SIZE
`define L3FFSQ_SIZE 8
`endif

// Prefetcher
`ifndef L3PRFQ_SIZE
`define L3PRFQ_SIZE 32
`endif

`ifndef L3PRFQ_STRIDE
`define L3PRFQ_STRIDE 0
`endif

// Fill Invalidator Size {Fill invalidator must be active}
`ifndef L3FILL_INVALIDAOR_SIZE
`define L3FILL_INVALIDAOR_SIZE 32
`endif

// Dram knobs
`ifndef L3SIMULATED_DRAM_LATENCY_CYCLES
`define L3SIMULATED_DRAM_LATENCY_CYCLES 2
`endif

 // VX_DEFINE
`endif
