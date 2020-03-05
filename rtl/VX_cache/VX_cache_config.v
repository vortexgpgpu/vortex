`ifndef VX_CACHE_CONFIG
`define VX_CACHE_CONFIG

`include "../VX_define.v"

// ========================================= Configurable Knobs =========================================

// General Cache Knobs
	// Size of cache in bytes
	`define CACHE_SIZE_BYTES 1024
	// Size of line inside a bank in bytes
	`define BANK_LINE_SIZE_BYTES 16
	// Number of banks {1, 2, 4, 8,...}
	`define NUMBER_BANKS 8
	// Size of a word in bytes
	`define WORD_SIZE_BYTES 4
	// Number of Word requests per cycle {1, 2, 4, 8, ...}
	`define NUMBER_REQUESTS `NT
	// Number of cycles to complete stage 1 (read from memory)
	`define STAGE_1_CYCLES 2

// Queues feeding into banks Knobs {1, 2, 4, 8, ...}

	// Core Request Queue Size
	`define REQQ_SIZE `NT*`NW
	// Miss Reserv Queue Knob
	`define MRVQ_SIZE `REQQ_SIZE
	// Dram Fill Rsp Queue Size
	`define DFPQ_SIZE 2

// Queues for writebacks Knobs {1, 2, 4, 8, ...}
	// Core Writeback Queue Size
	`define CWBQ_SIZE `REQQ_SIZE
	// Dram Writeback Queue Size
	`define DWBQ_SIZE 4
	// Dram Fill Req Queue Size
	`define DFQQ_SIZE `REQQ_SIZE

// Dram knobs
	`define SIMULATED_DRAM_LATENCY_CYCLES 10

// ========================================= Configurable Knobs =========================================

//                         data       tid                    rd  wb     warp_num   read  write
`define MRVQ_METADATA_SIZE (32 + $clog2(`NUMBER_REQUESTS) + 5 + 2 + (`NW_M1 + 1) + 3 + 3)

`define REQ_INST_META_SIZE (5 + 2 + (`NW_M1+1) + 3 + 3 + $clog2(`NUMBER_REQUESTS))

`define vx_clog2(value) $clog2(value)
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


`define BANK_SIZE_BYTES `CACHE_SIZE_BYTES/`NUMBER_BANKS


`define BANK_LINE_COUNT (`BANK_SIZE_BYTES/`BANK_LINE_SIZE_BYTES)
`define BANK_LINE_SIZE_WORDS (`BANK_LINE_SIZE_BYTES / `WORD_SIZE_BYTES)
`define BANK_LINE_SIZE_RNG `BANK_LINE_SIZE_WORDS-1:0

// Offset is fixed
`define OFFSET_ADDR_NUM_BITS 2
`define OFFSET_SIZE_END 1
`define OFFSET_ADDR_START 0
`define OFFSET_ADDR_END 1
`define OFFSET_ADDR_RNG `OFFSET_ADDR_END:`OFFSET_ADDR_START
`define OFFSET_SIZE_RNG `OFFSET_SIZE_END:0

`define WORD_SELECT_NUM_BITS $clog2(`BANK_LINE_SIZE_WORDS)
`define WORD_SELECT_SIZE_END `WORD_SELECT_NUM_BITS
`define WORD_SELECT_ADDR_START 1+`OFFSET_ADDR_END
`define WORD_SELECT_ADDR_END `WORD_SELECT_SIZE_END+`OFFSET_ADDR_END
`define WORD_SELECT_ADDR_RNG `WORD_SELECT_ADDR_END:`WORD_SELECT_ADDR_START
`define WORD_SELECT_SIZE_RNG `WORD_SELECT_SIZE_END-1:0

`define BANK_SELECT_NUM_BITS $clog2(`NUMBER_BANKS)
`define BANK_SELECT_SIZE_END `BANK_SELECT_NUM_BITS
`define BANK_SELECT_ADDR_START 1+`WORD_SELECT_ADDR_END
`define BANK_SELECT_ADDR_END `BANK_SELECT_SIZE_END+`BANK_SELECT_ADDR_START
`define BANK_SELECT_ADDR_RNG `BANK_SELECT_ADDR_END:`BANK_SELECT_ADDR_START
`define BANK_SELECT_SIZE_RNG `BANK_SELECT_SIZE_END-1:0

`define LINE_SELECT_NUM_BITS $clog2(`BANK_LINE_COUNT)
`define LINE_SELECT_SIZE_END `LINE_SELECT_NUM_BITS
`define LINE_SELECT_ADDR_START 1+`BANK_SELECT_ADDR_END
`define LINE_SELECT_ADDR_END `LINE_SELECT_SIZE_END+`LINE_SELECT_ADDR_START
`define LINE_SELECT_ADDR_RNG `LINE_SELECT_ADDR_END:`LINE_SELECT_ADDR_START
`define LINE_SELECT_SIZE_RNG `LINE_SELECT_SIZE_END-1:0

`define TAG_SELECT_NUM_BITS   32-(`OFFSET_ADDR_NUM_BITS + `WORD_SELECT_NUM_BITS + `BANK_SELECT_NUM_BITS + `LINE_SELECT_NUM_BITS)
`define TAG_SELECT_SIZE_END   `TAG_SELECT_NUM_BITS
`define TAG_SELECT_ADDR_START 1+`LINE_SELECT_ADDR_END
`define TAG_SELECT_ADDR_RNG  31:`TAG_SELECT_ADDR_START
`define TAG_SELECT_SIZE_RNG `TAG_SELECT_SIZE_END-1:0


`define BASE_ADDR_MASK (~((1<<`WORD_SELECT_ADDR_END)-1))


`endif

