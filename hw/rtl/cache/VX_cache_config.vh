`ifndef VX_CACHE_CONFIG
`define VX_CACHE_CONFIG

`include "VX_platform.vh"

`ifdef DBG_CACHE_REQ_INFO
`include "VX_define.vh"
`endif

`define REQ_TAG_WIDTH           `MAX(CORE_TAG_WIDTH, SNP_TAG_WIDTH)

`define REQS_BITS               `LOG2UP(NUM_REQS)

//                               tag              rw   byteen      tid
`define REQ_INST_META_WIDTH     (`REQ_TAG_WIDTH + 1  + WORD_SIZE + `REQS_BITS)

//                                data         metadata               word_sel                  is_snp  snp_inv 
`define MSHR_DATA_WIDTH     (`WORD_WIDTH + `REQ_INST_META_WIDTH + `UP(`WORD_SELECT_WIDTH) + 1     + 1)

`define BANK_BITS               `LOG2UP(NUM_BANKS)

`define WORD_WIDTH              (8 * WORD_SIZE)

`define BANK_LINE_WIDTH         (8 * BANK_LINE_SIZE)

`define BANK_SIZE               (CACHE_SIZE / NUM_BANKS)      
`define BANK_LINE_COUNT         (`BANK_SIZE / BANK_LINE_SIZE)
`define BANK_LINE_WORDS         (BANK_LINE_SIZE / WORD_SIZE)

// Offset select
`define OFFSET_ADDR_BITS        `CLOG2(WORD_SIZE)
`define OFFSET_ADDR_START       0
`define OFFSET_ADDR_END         (`OFFSET_ADDR_START+`OFFSET_ADDR_BITS-1)

// Word select
`define WORD_SELECT_BITS        `CLOG2(`BANK_LINE_WORDS)
`define WORD_SELECT_ADDR_START  (1+`OFFSET_ADDR_END)
`define WORD_SELECT_ADDR_END    (`WORD_SELECT_ADDR_START+`WORD_SELECT_BITS-1)

// Bank select
`define BANK_SELECT_BITS        `CLOG2(NUM_BANKS)
`define BANK_SELECT_ADDR_START  (1+`WORD_SELECT_ADDR_END)
`define BANK_SELECT_ADDR_END    (`BANK_SELECT_ADDR_START+`BANK_SELECT_BITS-1)

// Line select
`define LINE_SELECT_BITS        `CLOG2(`BANK_LINE_COUNT)
`define LINE_SELECT_ADDR_START  (1+`BANK_SELECT_ADDR_END)
`define LINE_SELECT_ADDR_END    (`LINE_SELECT_ADDR_START+`LINE_SELECT_BITS-1)

// Tag select
`define TAG_SELECT_BITS         (31-`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_START   (1+`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_END     31

`define WORD_SELECT_WIDTH       `CLOG2(`BANK_LINE_WORDS)

`define WORD_ADDR_WIDTH         (32-`CLOG2(WORD_SIZE))

`define DRAM_ADDR_WIDTH         (32-`CLOG2(BANK_LINE_SIZE))

`define LINE_ADDR_WIDTH         (`DRAM_ADDR_WIDTH-`BANK_SELECT_BITS)

`define BANK_SELECT_ADDR_RNG    (`BANK_SELECT_BITS+`WORD_SELECT_BITS-1):`WORD_SELECT_BITS

`define LINE_SELECT_ADDR_RNG    `WORD_ADDR_WIDTH-1:(`BANK_SELECT_BITS + `WORD_SELECT_BITS)

`define TAG_LINE_ADDR_RNG       `LINE_ADDR_WIDTH-1:`LINE_SELECT_BITS

`define BASE_ADDR_BITS          (`WORD_SELECT_BITS+`OFFSET_ADDR_BITS)

///////////////////////////////////////////////////////////////////////////////

`define CORE_REQ_TAG_COUNT      ((CORE_TAG_ID_BITS != 0) ? 1 : NUM_REQS)

`define DRAM_ADDR_BANK(x)       x[`BANK_SELECT_BITS-1:0]

`define DRAM_TO_LINE_ADDR(x)    x[`DRAM_ADDR_WIDTH-1:`BANK_SELECT_BITS]

`define LINE_TO_DRAM_ADDR(x, i) {x, `BANK_SELECT_BITS'(i)}

`define LINE_TO_BYTE_ADDR(x, i) {x, (32-$bits(x))'(i << (32-$bits(x)-`BANK_SELECT_BITS))}

`define TO_FULL_ADDR(x)         {x, (32-$bits(x))'(0)}

`endif
