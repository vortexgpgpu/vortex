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

`define WORD_ADDR_WIDTH         (32-`CLOG2(WORD_SIZE))
`define MEM_ADDR_WIDTH          (32-`CLOG2(LINE_SIZE))
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

`define ASSIGN_REQ_UUID(dst, tag) \
    if (UUID_BITS != 0) begin \
        assign dst = tag[TAG_WIDTH-1 -: UUID_BITS]; \
    end else begin \
        assign dst = 0; \
    end

///////////////////////////////////////////////////////////////////////////////

`define LINE_TO_MEM_ADDR(x, i)  {x, `BANK_SEL_BITS'(i)}

`define MEM_ADDR_TO_BANK_ID(x)  x[0 +: `BANK_SEL_BITS]

`define MEM_TAG_TO_REQ_ID(x)    x[MSHR_ADDR_WIDTH-1:0]

`define MEM_TAG_TO_BANK_ID(x)   x[MSHR_ADDR_WIDTH +: `BANK_SEL_BITS]

`define LINE_TO_BYTE_ADDR(x, i) {x, (32-$bits(x))'(i << (32-$bits(x)-`BANK_SEL_BITS))}

`define TO_FULL_ADDR(x)         {x, (32-$bits(x))'(0)}

///////////////////////////////////////////////////////////////////////////////

`define CACHE_MEM_TAG_WIDTH(mshr_size, num_banks) \
        (`CLOG2(mshr_size) + `CLOG2(num_banks) + `NC_TAG_BITS)
        
`define CACHE_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width) \
        (`CLOG2(num_reqs) + `CLOG2(line_size / word_size) + tag_width)

`define CACHE_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width) \
        (`CACHE_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width) + `NC_TAG_BITS)

`define CACHE_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, line_size, word_size, tag_width) \
        `MAX(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks), `CACHE_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width))

///////////////////////////////////////////////////////////////////////////////

`define CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches) \
        (tag_width + `ARB_SEL_BITS(num_inputs, `UP(num_caches)))  

`define CACHE_CLUSTER_MEM_ARB_TAG(tag_width, num_caches) \
        (tag_width + `ARB_SEL_BITS(`UP(num_caches), 1))

`define CACHE_CLUSTER_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width, num_inputs, num_caches) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CLOG2(num_reqs) + `CLOG2(line_size / word_size) + `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches)), num_caches)

`define CACHE_CLUSTER_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, tag_width, num_inputs, num_caches) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`CACHE_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches)) + `NC_TAG_BITS), num_caches)

`define CACHE_CLUSTER_NC_MEM_TAG_WIDTH(mshr_size, num_banks, num_reqs, line_size, word_size, tag_width, num_inputs, num_caches) \
        `CACHE_CLUSTER_MEM_ARB_TAG(`MAX(`CACHE_MEM_TAG_WIDTH(mshr_size, num_banks), `CACHE_NC_BYPASS_TAG_WIDTH(num_reqs, line_size, word_size, `CACHE_CLUSTER_CORE_ARB_TAG(tag_width, num_inputs, num_caches))), num_caches)

`endif
