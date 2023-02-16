`ifndef VX_CACHE_DEFINE_VH
`define VX_CACHE_DEFINE_VH

`include "VX_define.vh"   

`define REQ_SEL_BITS            `CLOG2(NUM_REQS)

//                                tag         valid req_idx              word_idx
`define MSHR_DATA_WIDTH         ((TAG_WIDTH + 1 +   `UP(`REQ_SEL_BITS) + `UP(`WORD_SEL_BITS)) * NUM_PORTS)

`define WORD_WIDTH              (8 * WORD_SIZE)

`define LINE_WIDTH              (8 * LINE_SIZE)

`define BANK_SIZE               (CACHE_SIZE / NUM_BANKS)

`define WAY_SEL_BITS            `CLOG2(NUM_WAYS)

`define LINES_PER_BANK          (`BANK_SIZE / (LINE_SIZE * NUM_WAYS))
`define WORDS_PER_LINE          (LINE_SIZE / WORD_SIZE)

`define WORD_ADDR_WIDTH         (`XLEN-`CLOG2(WORD_SIZE))
`define MEM_ADDR_WIDTH          (`XLEN-`CLOG2(LINE_SIZE))
`define LINE_ADDR_WIDTH         (`MEM_ADDR_WIDTH-`CLOG2(NUM_BANKS))

// Word select
`define WORD_SEL_BITS           `CLOG2(`WORDS_PER_LINE)
`define WORD_SEL_ADDR_START     0
`define WORD_SEL_ADDR_END       (`WORD_SEL_ADDR_START+`WORD_SEL_BITS-1)

// Bank select
`define BANK_SEL_BITS           `CLOG2(NUM_BANKS)
`define BANK_SEL_ADDR_START     (1+`WORD_SEL_ADDR_END)
`define BANK_SEL_ADDR_END       (`BANK_SEL_ADDR_START+`BANK_SEL_BITS-1)

// Line select
`define LINE_SEL_BITS           `CLOG2(`LINES_PER_BANK)
`define LINE_SEL_ADDR_START     (1+`BANK_SEL_ADDR_END)
`define LINE_SEL_ADDR_END       (`LINE_SEL_ADDR_START+`LINE_SEL_BITS-1)

// Tag select
`define TAG_SEL_BITS            (`WORD_ADDR_WIDTH-1-`LINE_SEL_ADDR_END)
`define TAG_SEL_ADDR_START      (1+`LINE_SEL_ADDR_END)
`define TAG_SEL_ADDR_END        (`WORD_ADDR_WIDTH-1)

`define LINE_TAG_ADDR(x)        x[`LINE_ADDR_WIDTH-1 : `LINE_SEL_BITS]

///////////////////////////////////////////////////////////////////////////////

`define LINE_TO_MEM_ADDR(x, i)  {x, `BANK_SEL_BITS'(i)}

`define MEM_ADDR_TO_BANK_ID(x)  x[0 +: `BANK_SEL_BITS]

`define MEM_TAG_TO_REQ_ID(x)    x[MSHR_ADDR_WIDTH-1:0]

`define MEM_TAG_TO_BANK_ID(x)   x[MSHR_ADDR_WIDTH +: `BANK_SEL_BITS]

`define LINE_TO_BYTE_ADDR(x, i) {x, (32-$bits(x))'(i << (32-$bits(x)-`BANK_SEL_BITS))}

`define TO_FULL_ADDR(x)         {x, (32-$bits(x))'(0)}

`endif // VX_CACHE_DEFINE_VH
