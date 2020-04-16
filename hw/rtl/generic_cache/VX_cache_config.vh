`ifndef VX_CACHE_CONFIG
`define VX_CACHE_CONFIG

`include "../VX_define.vh"


//                         data           tid                    rd  wb     warp_num   read  write


`define vx_clog2(value) ((value == 1) ? 1 : $clog2(value))


`define MRVQ_METADATA_SIZE (`WORD_SIZE + `vx_clog2(NUMBER_REQUESTS) + 5 + 2 + (`NW_BITS-1 + 1) + 3 + 3)

//                          5 + 2 + 4          + 3 + 3 + 1
`define REQ_INST_META_SIZE (5 + 2 + (`NW_BITS-1+1) + 3 + 3 + `vx_clog2(NUMBER_REQUESTS))

// `define vx_clog2_h(value, x) (value == (1 << x)) ? (x)

// `define vx_clog2(value)   (value == 0 ) ? 0 : \
//                           (value == 1 ) ? 1 : \
//                           `vx_clog2_h(value, 2 ) : \
//                           `vx_clog2_h(value, 3 ) : \
//                           `vx_clog2_h(value, 4 ) : \
//                           `vx_clog2_h(value, 5 ) : \
//                           `vx_clog2_h(value, 6 ) : \
//                           `vx_clog2_h(value, 7 ) : \
//                           `vx_clog2_h(value, 8 ) : \
//                           `vx_clog2_h(value, 9 ) : \
//                           `vx_clog2_h(value, 10) : \
//                           `vx_clog2_h(value, 11) : \
//                           `vx_clog2_h(value, 12) : \
//                           `vx_clog2_h(value, 13) : \
//                           `vx_clog2_h(value, 14) : \
//                           `vx_clog2_h(value, 15) : \
//                           `vx_clog2_h(value, 16) : \
//                           `vx_clog2_h(value, 17) : \
//                           `vx_clog2_h(value, 18) : \
//                           `vx_clog2_h(value, 19) : \
//                           `vx_clog2_h(value, 20) : \
//                           `vx_clog2_h(value, 21) : \
//                           `vx_clog2_h(value, 22) : \
//                           `vx_clog2_h(value, 23) : \
//                           `vx_clog2_h(value, 24) : \
//                           `vx_clog2_h(value, 25) : \
//                           `vx_clog2_h(value, 26) : \
//                           `vx_clog2_h(value, 27) : \
//                           `vx_clog2_h(value, 28) : \
//                           `vx_clog2_h(value, 29) : \
//                           `vx_clog2_h(value, 30) : \
//                           `vx_clog2_h(value, 31) : \
//                           0

`define WORD_SIZE (8*WORD_SIZE_BYTES)
`define WORD_SIZE_RNG (`WORD_SIZE)-1:0

// 128
`define BANK_SIZE_BYTES CACHE_SIZE_BYTES/NUMBER_BANKS

// 8
`define BANK_LINE_COUNT (`BANK_SIZE_BYTES/BANK_LINE_SIZE_BYTES)
// 4
`define BANK_LINE_WORDS (BANK_LINE_SIZE_BYTES / WORD_SIZE_BYTES)

// Offset is fixed
`define OFFSET_ADDR_NUM_BITS 2
`define OFFSET_SIZE_END 1
`define OFFSET_ADDR_START 0
`define OFFSET_ADDR_END 1
`define OFFSET_ADDR_RNG `OFFSET_ADDR_END:`OFFSET_ADDR_START
`define OFFSET_SIZE_RNG `OFFSET_SIZE_END:0

// 2
`define WORD_SELECT_NUM_BITS (`vx_clog2(`BANK_LINE_WORDS))
// 2
`define WORD_SELECT_SIZE_END (`WORD_SELECT_NUM_BITS)
// 2
`define WORD_SELECT_ADDR_START (1+`OFFSET_ADDR_END)
// 3
`define WORD_SELECT_ADDR_END (`WORD_SELECT_SIZE_END+`OFFSET_ADDR_END)
// 3:2
`define WORD_SELECT_ADDR_RNG `WORD_SELECT_ADDR_END:`WORD_SELECT_ADDR_START
`define WORD_SELECT_SIZE_RNG `WORD_SELECT_SIZE_END-1:0

// 3
`define BANK_SELECT_NUM_BITS (`vx_clog2(NUMBER_BANKS))
// 3
`define BANK_SELECT_SIZE_END (`BANK_SELECT_NUM_BITS)
// 4
`define BANK_SELECT_ADDR_START (1+`WORD_SELECT_ADDR_END)
// 6
`define BANK_SELECT_ADDR_END (`BANK_SELECT_SIZE_END+`BANK_SELECT_ADDR_START-1)
// 6:4
`define BANK_SELECT_ADDR_RNG `BANK_SELECT_ADDR_END:`BANK_SELECT_ADDR_START
// 2:0
`define BANK_SELECT_SIZE_RNG `BANK_SELECT_SIZE_END-1:0

// 3
`define LINE_SELECT_NUM_BITS (`vx_clog2(`BANK_LINE_COUNT))
// 3
`define LINE_SELECT_SIZE_END (`LINE_SELECT_NUM_BITS)
// 7
`define LINE_SELECT_ADDR_START (1+`BANK_SELECT_ADDR_END)
// 9
`define LINE_SELECT_ADDR_END (`LINE_SELECT_SIZE_END+`LINE_SELECT_ADDR_START-1)
// 9:7
`define LINE_SELECT_ADDR_RNG `LINE_SELECT_ADDR_END:`LINE_SELECT_ADDR_START
// 2:0
`define LINE_SELECT_SIZE_RNG `LINE_SELECT_SIZE_END-1:0


// 10
`define TAG_SELECT_ADDR_START (1+`LINE_SELECT_ADDR_END)
// 31:10
`define TAG_SELECT_ADDR_RNG  31:`TAG_SELECT_ADDR_START
// 22
`define TAG_SELECT_NUM_BITS  (32-`TAG_SELECT_ADDR_START)
// 22
`define TAG_SELECT_SIZE_END   (`TAG_SELECT_NUM_BITS)
// 21:0
`define TAG_SELECT_SIZE_RNG `TAG_SELECT_SIZE_END-1:0


`define BASE_ADDR_MASK (~((1<<(`WORD_SELECT_ADDR_END+1))-1))


`endif

