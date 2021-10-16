`ifndef VX_CACHE_DEFINE
`define VX_CACHE_DEFINE

`include "VX_platform.vh"

`ifdef DBG_CACHE_REQ_INFO
`include "VX_define.vh"
`endif

`define REQS_BITS               `LOG2UP(NUM_REQS)

`define PORTS_BITS              `LOG2UP(NUM_PORTS)

//                                tag              valid  tid          word_sel              
`define MSHR_DATA_WIDTH         ((CORE_TAG_WIDTH + 1 +    `REQS_BITS + `UP(`WORD_SELECT_BITS)) * NUM_PORTS)

`define WORD_WIDTH              (8 * WORD_SIZE)

`define CACHE_LINE_WIDTH        (8 * CACHE_LINE_SIZE)

`define BANK_SIZE               (CACHE_SIZE / NUM_BANKS)
`define LINES_PER_BANK          (`BANK_SIZE / CACHE_LINE_SIZE)
`define WORDS_PER_LINE          (CACHE_LINE_SIZE / WORD_SIZE)

`define WORD_ADDR_WIDTH         (32-`CLOG2(WORD_SIZE))
`define MEM_ADDR_WIDTH          (32-`CLOG2(CACHE_LINE_SIZE))
`define LINE_ADDR_WIDTH         (`MEM_ADDR_WIDTH-`BANK_SELECT_BITS)

// Word select
`define WORD_SELECT_BITS        `CLOG2(`WORDS_PER_LINE)
`define WORD_SELECT_ADDR_START  0
`define WORD_SELECT_ADDR_END    (`WORD_SELECT_ADDR_START+`WORD_SELECT_BITS-1)

// Bank select
`define BANK_SELECT_BITS        `CLOG2(NUM_BANKS)
`define BANK_SELECT_ADDR_START  (1+`WORD_SELECT_ADDR_END+BANK_ADDR_OFFSET)
`define BANK_SELECT_ADDR_END    (`BANK_SELECT_ADDR_START+`BANK_SELECT_BITS-1)

// Line select
`define LINE_SELECT_BITS        `CLOG2(`LINES_PER_BANK)
`define LINE_SELECT_ADDR_START  (1+`BANK_SELECT_ADDR_END)
`define LINE_SELECT_ADDR_END    (`LINE_SELECT_ADDR_START-BANK_ADDR_OFFSET+`LINE_SELECT_BITS-1)

// Tag select
`define TAG_SELECT_BITS         (`WORD_ADDR_WIDTH-1-`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_START   (1+`LINE_SELECT_ADDR_END)
`define TAG_SELECT_ADDR_END     (`WORD_ADDR_WIDTH-1)

`define BANK_SELECT_ADDR(x)     x[`BANK_SELECT_ADDR_END : `BANK_SELECT_ADDR_START]

`define LINE_SELECT_ADDR0(x)    x[`WORD_ADDR_WIDTH-1 : `LINE_SELECT_ADDR_START]
`define LINE_SELECT_ADDRX(x)    {x[`WORD_ADDR_WIDTH-1 : `LINE_SELECT_ADDR_START], x[`BANK_SELECT_ADDR_START-1 : 1+`WORD_SELECT_ADDR_END]}

`define LINE_TAG_ADDR(x)        x[`LINE_ADDR_WIDTH-1 : `LINE_SELECT_BITS]

`define CACHE_REQ_INFO_RNG      CORE_TAG_WIDTH-1 : (CORE_TAG_WIDTH-`DBG_CACHE_REQ_MDATAW)

///////////////////////////////////////////////////////////////////////////////

`define CORE_RSP_TAGS           ((CORE_TAG_ID_BITS != 0) ? 1 : NUM_REQS)

`define LINE_TO_MEM_ADDR(x, i)  {x, `BANK_SELECT_BITS'(i)}

`define MEM_ADDR_TO_BANK_ID(x)  x[0 +: `BANK_SELECT_BITS]

`define MEM_TAG_TO_REQ_ID(x)    x[MSHR_ADDR_WIDTH-1:0]

`define MEM_TAG_TO_BANK_ID(x)   x[MSHR_ADDR_WIDTH +: `BANK_SELECT_BITS]

`define LINE_TO_BYTE_ADDR(x, i) {x, (32-$bits(x))'(i << (32-$bits(x)-`BANK_SELECT_BITS))}

`define TO_FULL_ADDR(x)         {x, (32-$bits(x))'(0)}

`endif