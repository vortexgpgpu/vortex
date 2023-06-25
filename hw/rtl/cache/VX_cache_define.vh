`ifndef VX_CACHE_DEFINE_VH
`define VX_CACHE_DEFINE_VH

`include "VX_define.vh"   

`define CS_REQ_SEL_BITS         `CLOG2(NUM_REQS)

//                              tag         valid req_idx                 word_idx
`define CS_MSHR_DATA_WIDTH      ((TAG_WIDTH + 1 + `UP(`CS_REQ_SEL_BITS) + `UP(`CS_WORD_SEL_BITS)) * NUM_PORTS)

`define CS_WORD_WIDTH           (8 * WORD_SIZE)
`define CS_LINE_WIDTH           (8 * LINE_SIZE)
`define CS_BANK_SIZE            (CACHE_SIZE / NUM_BANKS)
`define CS_WAY_SEL_BITS         `CLOG2(NUM_WAYS)

`define CS_LINES_PER_BANK       (`CS_BANK_SIZE / (LINE_SIZE * NUM_WAYS))
`define CS_WORDS_PER_LINE       (LINE_SIZE / WORD_SIZE)

`define CS_WORD_ADDR_WIDTH      (`XLEN-`CLOG2(WORD_SIZE))
`define CS_MEM_ADDR_WIDTH       (`XLEN-`CLOG2(LINE_SIZE))
`define CS_LINE_ADDR_WIDTH      (`CS_MEM_ADDR_WIDTH-`CLOG2(NUM_BANKS))

// Word select
`define CS_WORD_SEL_BITS        `CLOG2(`CS_WORDS_PER_LINE)
`define CS_WORD_SEL_ADDR_START  0
`define CS_WORD_SEL_ADDR_END    (`CS_WORD_SEL_ADDR_START+`CS_WORD_SEL_BITS-1)

// Bank select
`define CS_BANK_SEL_BITS        `CLOG2(NUM_BANKS)
`define CS_BANK_SEL_ADDR_START  (1+`CS_WORD_SEL_ADDR_END)
`define CS_BANK_SEL_ADDR_END    (`CS_BANK_SEL_ADDR_START+`CS_BANK_SEL_BITS-1)

// Line select
`define CS_LINE_SEL_BITS        `CLOG2(`CS_LINES_PER_BANK)
`define CS_LINE_SEL_ADDR_START  (1+`CS_BANK_SEL_ADDR_END)
`define CS_LINE_SEL_ADDR_END    (`CS_LINE_SEL_ADDR_START+`CS_LINE_SEL_BITS-1)

// Tag select
`define CS_TAG_SEL_BITS         (`CS_WORD_ADDR_WIDTH-1-`CS_LINE_SEL_ADDR_END)
`define CS_TAG_SEL_ADDR_START   (1+`CS_LINE_SEL_ADDR_END)
`define CS_TAG_SEL_ADDR_END     (`CS_WORD_ADDR_WIDTH-1)

`define CS_LINE_TAG_ADDR(x)     x[`CS_LINE_ADDR_WIDTH-1 : `CS_LINE_SEL_BITS]

///////////////////////////////////////////////////////////////////////////////

`define CS_LINE_TO_MEM_ADDR(x, i)  {x, `CS_BANK_SEL_BITS'(i)}
`define CS_MEM_ADDR_TO_BANK_ID(x)  x[0 +: `CS_BANK_SEL_BITS]
`define CS_MEM_TAG_TO_REQ_ID(x)    x[MSHR_ADDR_WIDTH-1:0]
`define CS_MEM_TAG_TO_BANK_ID(x)   x[MSHR_ADDR_WIDTH +: `CS_BANK_SEL_BITS]

`define CS_LINE_TO_BYTE_ADDR(x, i) {x, (`XLEN-$bits(x))'(i << (`XLEN-$bits(x)-`CS_BANK_SEL_BITS))}
`define CS_TO_FULL_ADDR(x)         {x, (`XLEN-$bits(x))'(0)}

`endif // VX_CACHE_DEFINE_VH
