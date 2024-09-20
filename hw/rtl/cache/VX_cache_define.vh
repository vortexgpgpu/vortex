// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`ifndef VX_CACHE_DEFINE_VH
`define VX_CACHE_DEFINE_VH

`include "VX_define.vh"

`define CS_REQ_SEL_BITS         `CLOG2(NUM_REQS)

`define CS_WORD_WIDTH           (8 * WORD_SIZE)
`define CS_LINE_WIDTH           (8 * LINE_SIZE)
`define CS_BANK_SIZE            (CACHE_SIZE / NUM_BANKS)
`define CS_WAY_SEL_BITS         `CLOG2(NUM_WAYS)

`define CS_LINES_PER_BANK       (`CS_BANK_SIZE / (LINE_SIZE * NUM_WAYS))
`define CS_WORDS_PER_LINE       (LINE_SIZE / WORD_SIZE)

`define CS_WORD_ADDR_WIDTH      (`MEM_ADDR_WIDTH-`CLOG2(WORD_SIZE))
`define CS_MEM_ADDR_WIDTH       (`MEM_ADDR_WIDTH-`CLOG2(LINE_SIZE))
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

`define CS_LINE_ADDR_TAG(x)     x[`CS_LINE_ADDR_WIDTH-1 : `CS_LINE_SEL_BITS]

///////////////////////////////////////////////////////////////////////////////

`define CS_LINE_TO_MEM_ADDR(x, i)  {x, `CS_BANK_SEL_BITS'(i)}
`define CS_MEM_ADDR_TO_BANK_ID(x)  x[0 +: `CS_BANK_SEL_BITS]
`define CS_MEM_TAG_TO_REQ_ID(x)    x[MSHR_ADDR_WIDTH-1:0]
`define CS_MEM_TAG_TO_BANK_ID(x)   x[MSHR_ADDR_WIDTH +: `CS_BANK_SEL_BITS]

`define CS_LINE_TO_FULL_ADDR(x, i) {x, (`XLEN-$bits(x))'(i << (`XLEN-$bits(x)-`CS_BANK_SEL_BITS))}
`define CS_MEM_TO_FULL_ADDR(x)     {x, (`XLEN-$bits(x))'(0)}

///////////////////////////////////////////////////////////////////////////////

`define PERF_CACHE_ADD(dst, src, count) \
    `PERF_COUNTER_ADD (dst, src, reads, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, writes, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, read_misses, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, write_misses, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, bank_stalls, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, mshr_stalls, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, mem_stalls, `PERF_CTR_BITS, count, (count > 1)) \
    `PERF_COUNTER_ADD (dst, src, crsp_stalls, `PERF_CTR_BITS, count, (count > 1))

`endif // VX_CACHE_DEFINE_VH
