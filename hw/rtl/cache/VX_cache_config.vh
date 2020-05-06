`ifndef VX_CACHE_CONFIG
`define VX_CACHE_CONFIG

`include "VX_define.vh"

`define BYTE_EN_NO      3'h7 
`define BYTE_EN_LB      3'h0 
`define BYTE_EN_LH      3'h1
`define BYTE_EN_LW      3'h2
`define BYTE_EN_HB      3'h4
`define BYTE_EN_HH      3'h5
`define BYTE_EN_BITS    3

//                               data           tid                    tag              read             write             base addr 
`define MRVQ_METADATA_WIDTH     (`WORD_WIDTH + `REQS_BITS + CORE_TAG_WIDTH + `BYTE_EN_BITS + `BYTE_EN_BITS + `BASE_ADDR_BITS)

//                               tag               read             write            reqs
`define REQ_INST_META_WIDTH     (CORE_TAG_WIDTH + `BYTE_EN_BITS + `BYTE_EN_BITS + `REQS_BITS)

`define REQS_BITS               `LOG2UP(NUM_REQUESTS)

`define BANK_BITS               `LOG2UP(NUM_BANKS)

`define WORD_WIDTH              (8 * WORD_SIZE)
`define BYTE_WIDTH              (`WORD_WIDTH / 4)

`define BANK_LINE_WIDTH         (8 * BANK_LINE_SIZE)

`define BANK_SIZE               (CACHE_SIZE / NUM_BANKS)      
`define BANK_LINE_COUNT         (`BANK_SIZE / BANK_LINE_SIZE)
`define BANK_LINE_WORDS         (BANK_LINE_SIZE / WORD_SIZE)

// Offset select
`define OFFSET_ADDR_BITS        `CLOG2(WORD_SIZE)
`define OFFSET_ADDR_START       0
`define OFFSET_ADDR_END         (`OFFSET_ADDR_START+`OFFSET_ADDR_BITS-1)
`define OFFSET_ADDR_RNG         `OFFSET_ADDR_END:`OFFSET_ADDR_START

// Word select
`define WORD_SELECT_BITS        `CLOG2(`BANK_LINE_WORDS)
`define WORD_SELECT_ADDR_START  (1+`OFFSET_ADDR_END)
`define WORD_SELECT_ADDR_END    (`WORD_SELECT_ADDR_START+`WORD_SELECT_BITS-1)
`define WORD_SELECT_ADDR_RNG    `WORD_SELECT_ADDR_END:`WORD_SELECT_ADDR_START

// Bank select
`define BANK_SELECT_BITS        `CLOG2(NUM_BANKS)
`define BANK_SELECT_ADDR_START  (1+`WORD_SELECT_ADDR_END)
`define BANK_SELECT_ADDR_END    (`BANK_SELECT_ADDR_START+`BANK_SELECT_BITS-1)
`define BANK_SELECT_ADDR_RNG    `BANK_SELECT_ADDR_END:`BANK_SELECT_ADDR_START

// Line select
`define LINE_SELECT_BITS        `CLOG2(`BANK_LINE_COUNT)
`define LINE_SELECT_ADDR_START  (1+`BANK_SELECT_ADDR_END)
`define LINE_SELECT_ADDR_END    (`LINE_SELECT_ADDR_START+`LINE_SELECT_BITS-1)
`define LINE_SELECT_ADDR_RNG    `LINE_SELECT_ADDR_END:`LINE_SELECT_ADDR_START

// Tag select
`define TAG_SELECT_BITS         (31-`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_START   (1+`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_END     31
`define TAG_SELECT_ADDR_RNG     `TAG_SELECT_ADDR_END:`TAG_SELECT_ADDR_START

`define DRAM_ADDR_WIDTH         (32-`CLOG2(BANK_LINE_SIZE))

`define LINE_ADDR_WIDTH         (`DRAM_ADDR_WIDTH-`BANK_SELECT_BITS)

`define TAG_LINE_ADDR_RNG       `LINE_ADDR_WIDTH-1:`LINE_SELECT_BITS

`define BASE_ADDR_BITS          (`WORD_SELECT_BITS+`OFFSET_ADDR_BITS)

///////////////////////////////////////////////////////////////////////////////

`define CORE_REQ_TAG_COUNT      ((CORE_TAG_ID_BITS != 0) ? 1 : NUM_REQUESTS)

`define DRAM_ADDR_BANK(x)       x[`BANK_SELECT_BITS-1:0]

`define DRAM_TO_LINE_ADDR(x)    x[`DRAM_ADDR_WIDTH-1:`BANK_SELECT_BITS]

`define LINE_TO_DRAM_ADDR(x, i) {x, (`BANK_SELECT_BITS)'(i)};

`endif
