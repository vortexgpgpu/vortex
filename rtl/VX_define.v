`include "./VX_define_synth.v"



`define NT_M1 (`NT-1)

// NW_M1 is actually log2(NW)
`define NW_M1 (`CLOG2(`NW))

// Uncomment the below line if NW=1
// `define ONLY

// `define SYN 1
// `define ASIC 1
`define SYN_FUNC 1

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


// `define PARAM

//Cache configurations
//Cache configurations
 //Bytes
`define ICACHE_SIZE  1024
`define ICACHE_WAYS  2
//Bytes
`define ICACHE_BLOCK 16
`define ICACHE_BANKS 1
`define ICACHE_LOG_NUM_BANKS `CLOG2(`ICACHE_BANKS)

`define ICACHE_NUM_WORDS_PER_BLOCK (`ICACHE_BLOCK / (`ICACHE_BANKS * 4))
`define ICACHE_NUM_REQ    1
`define ICACHE_LOG_NUM_REQ `CLOG2(`ICACHE_NUM_REQ)

 //set this to 1 if CACHE_WAYS is 1
`define ICACHE_WAY_INDEX `CLOG2(`ICACHE_WAYS)
//`define ICACHE_WAY_INDEX 1
`define ICACHE_BLOCK_PER_BANK  (`ICACHE_BLOCK / `ICACHE_BANKS)

// Offset
`define ICACHE_OFFSET_NB (`CLOG2(`ICACHE_NUM_WORDS_PER_BLOCK))

`define ICACHE_ADDR_OFFSET_ST  (2+$clog2(`ICACHE_BANKS))
`define ICACHE_ADDR_OFFSET_ED  (`ICACHE_ADDR_OFFSET_ST+(`ICACHE_OFFSET_NB)-1)


`define ICACHE_ADDR_OFFSET_RNG `ICACHE_ADDR_OFFSET_ED:`ICACHE_ADDR_OFFSET_ST
`define ICACHE_OFFSET_SIZE_RNG (`CLOG2(`ICACHE_NUM_WORDS_PER_BLOCK)-1):0
`define ICACHE_OFFSET_ST 0
`define ICACHE_OFFSET_ED ($clog2(`ICACHE_NUM_WORDS_PER_BLOCK)-1)

// Index
// `define ICACHE_NUM_IND (`ICACHE_SIZE / (`ICACHE_WAYS * `ICACHE_BLOCK_PER_BANK))
`define ICACHE_NUM_IND (`ICACHE_SIZE / (`ICACHE_WAYS * `ICACHE_BLOCK))
`define ICACHE_IND_NB ($clog2(`ICACHE_NUM_IND))

`define ICACHE_IND_ST  (`ICACHE_ADDR_OFFSET_ED+1)
`define ICACHE_IND_ED  (`ICACHE_IND_ST+`ICACHE_IND_NB-1)

`define ICACHE_ADDR_IND_RNG `ICACHE_IND_ED:`ICACHE_IND_ST
`define ICACHE_IND_SIZE_RNG `ICACHE_IND_NB-1:0

`define ICACHE_IND_SIZE_START 0
`define ICACHE_IND_SIZE_END   `ICACHE_IND_NB-1


// Tag
`define ICACHE_ADDR_TAG_RNG 31:(`ICACHE_IND_ED+1)
`define ICACHE_TAG_SIZE_RNG (32-(`ICACHE_IND_ED+1)-1):0
`define ICACHE_TAG_SIZE_START 0
`define ICACHE_TAG_SIZE_END	  (32-(`ICACHE_IND_ED+1)-1)
`define ICACHE_ADDR_TAG_START  (`ICACHE_IND_ED+1)
`define ICACHE_ADDR_TAG_END    31

//Cache configurations
//Bytes
`define DCACHE_SIZE  4096
`define DCACHE_WAYS  2

//Bytes
`define DCACHE_BLOCK 64
`define DCACHE_BANKS 4
`define DCACHE_LOG_NUM_BANKS $clog2(`DCACHE_BANKS)
`define DCACHE_NUM_WORDS_PER_BLOCK (`DCACHE_BLOCK / (`DCACHE_BANKS * 4))
`define DCACHE_NUM_REQ    `NT
`define DCACHE_LOG_NUM_REQ $clog2(`DCACHE_NUM_REQ)

//set this to 1 if CACHE_WAYS is 1
`define DCACHE_WAY_INDEX $clog2(`DCACHE_WAYS)
//`define DCACHE_WAY_INDEX 1
`define DCACHE_BLOCK_PER_BANK  (`DCACHE_BLOCK / `DCACHE_BANKS)

// Offset
`define DCACHE_OFFSET_NB ($clog2(`DCACHE_NUM_WORDS_PER_BLOCK))

`define DCACHE_ADDR_OFFSET_ST  (2+$clog2(`DCACHE_BANKS))
`define DCACHE_ADDR_OFFSET_ED  (`DCACHE_ADDR_OFFSET_ST+(`DCACHE_OFFSET_NB)-1)


`define DCACHE_ADDR_OFFSET_RNG `DCACHE_ADDR_OFFSET_ED:`DCACHE_ADDR_OFFSET_ST
`define DCACHE_OFFSET_SIZE_RNG ($clog2(`DCACHE_NUM_WORDS_PER_BLOCK)-1):0
`define DCACHE_OFFSET_ST 0
`define DCACHE_OFFSET_ED ($clog2(`DCACHE_NUM_WORDS_PER_BLOCK)-1)

// Index
// `define DCACHE_NUM_IND (`DCACHE_SIZE / (`DCACHE_WAYS * `DCACHE_BLOCK_PER_BANK))
`define DCACHE_NUM_IND (`DCACHE_SIZE / (`DCACHE_WAYS * `DCACHE_BLOCK))
`define DCACHE_IND_NB ($clog2(`DCACHE_NUM_IND))

`define DCACHE_IND_ST  (`DCACHE_ADDR_OFFSET_ED+1)
`define DCACHE_IND_ED  (`DCACHE_IND_ST+`DCACHE_IND_NB-1)

`define DCACHE_ADDR_IND_RNG `DCACHE_IND_ED:`DCACHE_IND_ST
`define DCACHE_IND_SIZE_RNG `DCACHE_IND_NB-1:0

`define DCACHE_IND_SIZE_START 0
`define DCACHE_IND_SIZE_END   `DCACHE_IND_NB-1


// Tag
`define DCACHE_ADDR_TAG_RNG 31:(`DCACHE_IND_ED+1)
`define DCACHE_TAG_SIZE_RNG (32-(`DCACHE_IND_ED+1)-1):0
`define DCACHE_TAG_SIZE_START 0
`define DCACHE_TAG_SIZE_END	  (32-(`DCACHE_IND_ED+1)-1)
`define DCACHE_ADDR_TAG_START  (`DCACHE_IND_ED+1)
`define DCACHE_ADDR_TAG_END    31

// Mask
`define DCACHE_MEM_REQ_ADDR_MASK (32'hffffffff - (`DCACHE_BLOCK-1))
`define ICACHE_MEM_REQ_ADDR_MASK (32'hffffffff - (`ICACHE_BLOCK-1))



///////

//`define SHARED_MEMORY_SIZE 4096
`define SHARED_MEMORY_SIZE 8192
`define SHARED_MEMORY_BANKS 4
//`define SHARED_MEMORY_BYTES_PER_READ 16
//`define SHARED_MEMORY_HEIGHT ((`SHARED_MEMORY_SIZE) / (`SHARED_MEMORY_BANKS * `SHARED_MEMORY_BYTES_PER_READ))

//`define SHARED_MEMORY_SIZE 16384
//`define SHARED_MEMORY_BANKS 8
`define SHARED_MEMORY_BYTES_PER_READ 16
//`define SHARED_MEMORY_BITS_PER_BANK 3
`define SHARED_MEMORY_BITS_PER_BANK `CLOG2(`SHARED_MEMORY_BANKS)
`define SHARED_MEMORY_NUM_REQ    `NT
`define SHARED_MEMORY_WORDS_PER_READ (`SHARED_MEMORY_BYTES_PER_READ / 4)
`define SHARED_MEMORY_LOG_WORDS_PER_READ $clog2(`SHARED_MEMORY_WORDS_PER_READ)
`define SHARED_MEMORY_HEIGHT ((`SHARED_MEMORY_SIZE) / (`SHARED_MEMORY_BANKS * `SHARED_MEMORY_BYTES_PER_READ))

`define SHARED_MEMORY_BANK_OFFSET_ST  (2)
`define SHARED_MEMORY_BANK_OFFSET_ED  (2+$clog2(`SHARED_MEMORY_BANKS)-1)
`define SHARED_MEMORY_BLOCK_OFFSET_ST (`SHARED_MEMORY_BANK_OFFSET_ED + 1)
`define SHARED_MEMORY_BLOCK_OFFSET_ED (`SHARED_MEMORY_BLOCK_OFFSET_ST +`SHARED_MEMORY_LOG_WORDS_PER_READ-1)
`define SHARED_MEMORY_INDEX_OFFSET_ST  (`SHARED_MEMORY_BLOCK_OFFSET_ED + 1)
`define SHARED_MEMORY_INDEX_OFFSET_ED  (`SHARED_MEMORY_INDEX_OFFSET_ST + $clog2(`SHARED_MEMORY_HEIGHT)-1)
