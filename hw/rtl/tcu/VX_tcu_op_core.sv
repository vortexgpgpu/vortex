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

`include "VX_define.vh"

`ifdef TCU_OP

module VX_tcu_op_core import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    // Width of tcu_lsu_mem_if's tag. With several engines sharing the TCU
    // memory port, VX_tcu_unit narrows it by the arbiter's select bits, which
    // VX_lsu_mem_arb then inserts to route responses back to this engine.
    parameter TAG_WIDTH = LSU_TAG_WIDTH
) (
    `SCOPE_IO_DECL

    input wire clk,
    input wire reset,

    // Inputs
    VX_execute_if.slave execute_if,

    VX_lsu_mem_if.master tcu_lsu_mem_if,
    VX_txbar_bus_if.master txbar_bus_if,

    // Outputs
    VX_result_if.master result_if
);
    `UNUSED_SPARAM (INSTANCE_ID);

    localparam MDATA_WIDTH = UUID_WIDTH + NW_WIDTH + PC_BITS + NUM_REGS_BITS;

    // Operand pipeline ahead of the multipliers. On the exact-product datapath
    // this is a word select then a sub-word align (D2); the legacy path builds
    // whole operand sets and needs a third stage for S1 compaction.
`ifdef VX_CFG_TCU_TYPE_TFR
    localparam OPND_LATENCY = 2;
`else
    localparam OPND_LATENCY = 3;
`endif
`ifndef VX_CFG_TCU_TYPE_DPI
`ifndef VX_CFG_TCU_TYPE_BHF
`ifndef VX_CFG_TCU_TYPE_TFR
    `error "VX_tcu_op_core: VX_CFG_TCU_TYPE_DPI, VX_CFG_TCU_TYPE_BHF or VX_CFG_TCU_TYPE_TFR must be defined"
`endif
`endif
`endif

`ifdef VX_CFG_TCU_TYPE_TFR
    // Exact-product datapath (VX_tcu_op_mul + VX_tcu_op_accu). Products are
    // never rounded; the accumulator adds them as fixed point in one cycle and
    // rounds once per output element at flush.
    localparam PROD_REG      = 1;  // significand-product register (DSP48 PREG)
    localparam FEOP_LATENCY  = OPND_LATENCY + PROD_REG;
    // Alignment is a coarse and a fine shift; the accumulate itself is a single
    // carry-save compression, so the bank update costs one more cycle.
    localparam ALIGN_LATENCY = 2;  // crossbar output -> aligned addend
    localparam ACCU_ADD_LATENCY = 1; // addend -> value in the bank
    // Flush: entry, |sum+carry|, lzc, normalize, fuse-align, fuse-add, round.
    localparam FLUSH_LATENCY = 7;
`else
    localparam FREC_LATENCY = 1;   // multiplier input recode stage
    localparam FMUL_LATENCY = 2;
    localparam FRND_LATENCY = 1;
    // Accumulate adder: input recode, add and round stages.
    localparam FACC_REC_LATENCY = 1;
    localparam FACC_ADD_LATENCY = 1;
    localparam FACC_LATENCY = FACC_REC_LATENCY + FACC_ADD_LATENCY + FRND_LATENCY;
    localparam FEOP_LATENCY = OPND_LATENCY + FREC_LATENCY + FMUL_LATENCY + FRND_LATENCY;
    // Accumulator read and execute stages between the crossbar and the adder.
    localparam ACCU_READ_LATENCY = 2;
`endif
    localparam MDATA_QUEUE_DEPTH = 1; // At maximum we have another intruction pending when the current one is finishing
    localparam XBAR_LATENCY      = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / 32;
`ifdef VX_CFG_TCU_TYPE_TFR
    // Cycles from the accumulator write port to the value being in a bank.
    localparam ACCU_WRITE_DEPTH = XBAR_LATENCY + ALIGN_LATENCY + ACCU_ADD_LATENCY;
`endif

    // K (8-bit instruction field) bounds the k-set counter.
    localparam K_W      = 8;
    // A/B operand window offsets, in elements: < i_ratio_max * NUM_LSU_LANES.
    localparam OFFSET_W = $clog2(8 * `VX_CFG_NUM_LSU_LANES);

    `UNUSED_VAR (execute_if.data.rs3_data)
    `UNUSED_VAR (execute_if.data.op_args)

    // Request side of the memory port. The engine owns tcu_lsu_mem_if's
    // request signals, so it exports them registered (req_buf, below).
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) lsu_req_if ();

    function automatic [1:0] calc_lg_i_ratio (input [3:0] fmt);
        case (32'(fmt))
            TCU_FP32_ID, TCU_I32_ID,
            TCU_TF32_ID:              calc_lg_i_ratio = 2'd0;
            TCU_FP16_ID, TCU_BF16_ID: calc_lg_i_ratio = 2'd1;
            TCU_FP8_ID,  TCU_BF8_ID,
            TCU_MXFP8_ID,
            TCU_I8_ID,   TCU_U8_ID:   calc_lg_i_ratio = 2'd2;
            TCU_I4_ID,   TCU_U4_ID:   calc_lg_i_ratio = 2'd3;
            default:                  calc_lg_i_ratio = 2'd0;
        endcase
    endfunction

    initial begin
`ifdef VX_CFG_TCU_TYPE_BHF
        `TRACE(1, ("[tcu_op_core]: VX_CFG_TCU_TYPE_BHF defined!\n\n"));
`elsif VX_CFG_TCU_TYPE_DPI
        `TRACE(1, ("[tcu_op_core]: VX_CFG_TCU_TYPE_DPI defined!\n\n"));
`endif
    end


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// INITIALIZATIONS & FSM

    /* Memory Requests handling */
    localparam BYTES_PER_MEM_REQUEST = LSU_WORD_SIZE * `VX_CFG_NUM_LSU_LANES;
`ifndef TCU_DISABLE_S1
    localparam SETS_PER_S1_BITMAP_BLOCK = BYTES_PER_MEM_REQUEST * 8 / TCU_TC_N_OP;
    localparam LG_SETS_PER_S1_BITMAP_BLOCK = $clog2(SETS_PER_S1_BITMAP_BLOCK);
    `STATIC_ASSERT(`IS_POW2(SETS_PER_S1_BITMAP_BLOCK), ("SETS_PER_S1_BITMAP_BLOCK must be power-of-two for shift/& indexing"))
`endif

    localparam SETS_PER_S2_BITMAP_BLOCK = BYTES_PER_MEM_REQUEST * 8 / (TCU_TC_M_OP + TCU_TC_N_OP);
    localparam LG_SETS_PER_S2_BITMAP_BLOCK = $clog2(SETS_PER_S2_BITMAP_BLOCK);
    `STATIC_ASSERT(`IS_POW2(SETS_PER_S2_BITMAP_BLOCK), ("SETS_PER_S2_BITMAP_BLOCK must be power-of-two for shift/& indexing"))
    
    // Amount of responses we can store before processing
    localparam A_BUF_SLOTS      = 2; // Current code only works for '2'
    localparam B_BUF_SLOTS      = 2; // Current code only works for '2'
    localparam BITMAP_BUF_SLOTS = 2; // Current code only works for '2'


    // Registers_per_FEOP_block / registers_per_LSU_load - 1
    localparam C_BUF_SLOTS = 6'(TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / `VX_CFG_NUM_LSU_LANES - 1); // Amount of responses we can store before accumulating
    
    `STATIC_ASSERT ((TCU_FEOP_STEPS != 32) || C_BUF_SLOTS == '0, ("for 32 steps, we dont accumulate C"));

    /* Used to calculate the total A, B blocks */
    localparam LG_REGS_PER_BLOCK = $clog2(`VX_CFG_NUM_LSU_LANES);

    localparam MATRIX_ID_BITS  = 4;   // one-hot: bitmap, A, B, C
    localparam MATRIX_IDX_BITS = 2;   // binary index of the same, on the wire
    reg a_req_pending_r, b_req_pending_r, c_req_pending_r, bm_req_pending_r;
    wire rsp_block_done;
    wire [MATRIX_ID_BITS-1:0] reqs;
    wire [MATRIX_ID_BITS-1:0] grant_onehot;
    wire rd_req_fire;
    wire rd_req_valid;
    wire [`VX_CFG_XLEN-1:0] req_rd_addr;

    localparam int C_BLOCKS_PER_FEOP_BLOCK = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / `VX_CFG_NUM_LSU_LANES;
    localparam LG_C_BLOCKS_PER_FEOP_BLOCK = $clog2(C_BLOCKS_PER_FEOP_BLOCK);
    wire rd_rsp_fire;
    wire [MATRIX_ID_BITS-1:0] rsp_matrix_id;
    wire accumulate_c;
    reg  c_write_r;       // registered C-preload write into the accumulator
    wire [$clog2(TCU_FEOP_STEPS):0] c_blk_idx;
    wire [C_BUF_SLOTS:0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] C_feop_block;

    // Tracked for the issue FSM's step accounting; only horizontal_steps and
    // set_steps are read back.
    reg  [LG_TCU_FEOP_M_STEPS:0] vertical_steps;
    `UNUSED_VAR (vertical_steps)
    reg  [LG_TCU_FEOP_N_STEPS:0] horizontal_steps;
    reg  [LG_TCU_FEOP_STEPS:0]   set_steps;

    reg [LG_TCU_FEOP_STEPS-1:0] step; // step increments from 0 -> set_steps
    reg [K_W-1:0] set;
    reg issuing_done;
    wire [LG_TCU_FEOP_N_STEPS:0] horizontal_steps_safe;
    wire [LG_TCU_TC_M_OP-1:0] m;
    wire [LG_TCU_TC_N_OP-1:0] n;
    wire last_step_in_set;
    wire last_step_in_block_a;
    wire last_step_in_block_b;
    wire last_step_in_execution;
    wire last_step_in_bitmap_block;
    
    wire bitmap_block_ready;
    wire [LG_TCU_TC_M_OP:0] a_set_elems;
    wire a_curr_loaded;
    wire a_next_loaded;
    wire [`VX_CFG_XLEN-1:0] a_window_need;
    wire a_window_ready;
    wire b_curr_loaded;
    wire b_next_loaded;
    wire b_window_ready;
    wire issue_busy;

    // Current set's bitmaps and zero counts (registered, see g_bm_cand)
    reg  [TCU_TC_M_OP-1:0] a_bitmap_in;
    reg  [TCU_TC_N_OP-1:0] b_bitmap_in;
    reg  [LG_TCU_TC_M_OP:0] a_zeros;
    reg  [LG_TCU_TC_N_OP:0] b_zeros;
    reg                     bitmap_fresh;

    wire [TCU_TC_M_OP-1:0] a_bitmap_out;
    wire [TCU_TC_N_OP-1:0] b_bitmap_out;

    wire [LG_TCU_TC_M_OP:0] a_non_zeros;
    wire [LG_TCU_TC_N_OP:0] b_non_zeros;
    wire [TCU_TC_M_OP-1:0][LG_TCU_TC_M_OP-1:0] a_addresses;
    wire [TCU_TC_N_OP-1:0][LG_TCU_TC_N_OP-1:0] b_addresses;

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] a_step_addresses;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] b_step_addresses;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] a_step_valids;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] b_step_valids;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] a_step_addresses_delayed;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] b_step_addresses_delayed;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] a_step_valids_delayed;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] b_step_valids_delayed;

    // Elements per LSU block at the current format: i_ratio * NUM_LSU_LANES.
    wire [31:0] window_elems;
    reg [OFFSET_W-1:0] a_offset;
    reg [OFFSET_W-1:0] b_offset;
    wire last_set_in_block_a;
    wire last_set_in_block_b;

`ifndef VX_CFG_TCU_TYPE_TFR
    // Operand-window state for the legacy datapath, which shifts the double
    // buffers rather than indexing them (D2). Declared here because nothing on
    // the TFR path drives or reads it.
    localparam int A_BUF_W = `VX_CFG_NUM_LSU_LANES * `VX_CFG_XLEN;
    localparam int B_BUF_W = `VX_CFG_NUM_LSU_LANES * `VX_CFG_XLEN;
    localparam int A_SET_W = TCU_TC_M_OP * `VX_CFG_XLEN;
    localparam int B_SET_W = TCU_TC_N_OP * `VX_CFG_XLEN;
    // Bit offset of an operand window into its double buffer.
    localparam WIN_SHIFT_W = $clog2(A_BUF_W * A_BUF_SLOTS) + 1;
    reg [A_SET_W-1:0] a_set_flat_compressed; // Only used in S1 case
    wire [A_SET_W-1:0] a_set_flat_processed;

    wire [A_BUF_W * A_BUF_SLOTS-1:0] A_window;
    wire [A_SET_W-1:0]               a_set_flat;
    wire [B_BUF_W * B_BUF_SLOTS-1:0] B_window;
    wire [B_SET_W-1:0]               b_set_flat;

    localparam int STEP_ELEM_CNT = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] write_data;
    wire [STEP_ELEM_CNT-1:0][`VX_CFG_XLEN-1:0] read_data;
`else
    // The accumulator returns the flushed row already rounded to fp32.
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] read_row_data;
`endif
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] read_row_valid;
    wire [LG_TCU_FEOP_STEPS-1:0] read_block_idx;
    wire [LG_TCU_FEOP_BLOCK_M_SIZE-1:0] read_row_in_block;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] c_blk_rows;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] c_blk_cols;
    wire accu_read_en;
    wire accu_write_valid;
    wire accu_enable;
    wire accu_queues_ready;
    wire accu_ready_to_flush;

    localparam int PAD_LANES = `VX_CFG_NUM_LSU_LANES - TCU_FEOP_BLOCK_N_SIZE;
    wire [`VX_CFG_XLEN-1:0] txbar_bar_id;
    wire [BAR_ADDR_W-1:0] txbar_addr;
    wire [BAR_ADDR_W-1:0] op_ctx_bar_addr;
    wire op_ctx_empty;
    wire result_pending;
    wire result_txbar_req;
    wire result_fire;

    wire [MDATA_WIDTH-1:0] mdata_queue_din, mdata_queue_dout;
    wire [UUID_WIDTH-1:0]    op_uuid;
    wire [NW_WIDTH-1:0]      op_wid;
    wire [PC_BITS-1:0]       op_PC;
    wire [NUM_REGS_BITS-1:0] op_rd;
    wire mqueue_full;
`ifdef VX_CFG_TCU_TYPE_TFR
    // Exact products from VX_tcu_op_mul, straight into the accumulator.
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][TCU_OP_EXP_W-1:0] prod_exp;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][TCU_OP_MAG_W-1:0] prod_mag;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0]                   prod_sign;
    fedp_excep_t [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0]           prod_exc;
`else
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] d_block;
`endif
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] d_line;
    wire wr_req_fire;
    wire ready_to_flush_delayed;
    wire [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE-1:0] d_line_to_flush_delayed;
    wire valid_out;
`ifdef VX_CFG_TCU_TYPE_TFR
    wire valid_out_w;
    wire [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE-1:0] d_line_to_flush_wr;
    wire flush_enable;
`endif

    wire [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE:0] d_lines_ready;
    wire ready_to_flush_raw;
    wire ready_to_flush;
    reg  [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE:0] d_line_to_flush;
    wire no_flush_complete;
    wire result_pulse;
    reg  result_pending_r;

    // preserve full byte address for correct LMEM detection
    localparam MEM_ASHIFT = `CLOG2(`VX_CFG_MEM_BLOCK_SIZE);      // bytes -> block
    localparam MEM_ADDRW  = `VX_CFG_MEM_ADDR_WIDTH - MEM_ASHIFT; // block address width
    localparam REQ_ASHIFT = `CLOG2(LSU_WORD_SIZE);        // bytes -> LSU word
    // 64-bit arithmetic: base + size overflows XLEN when LMEM tops out at
    // 2^32 (e.g. base 0xffff0000 with LMEM_LOG_SIZE=16), wrapping END to 0.
    // When LMEM reaches the very top of the block-address space, the
    // exclusive END does not fit in MEM_ADDRW bits either, so the upper
    // bound check must be dropped entirely (see LMEM_AT_ADDR_TOP uses).
    localparam [63:0] LMEM_ADDR_END64 = (64'(`VX_MEM_LMEM_BASE_ADDR) + 64'(1 << `VX_CFG_LMEM_LOG_SIZE)) >> MEM_ASHIFT;
    localparam LMEM_AT_ADDR_TOP = (LMEM_ADDR_END64 >= (64'd1 << MEM_ADDRW));
    localparam [MEM_ADDRW-1:0] LMEM_ADDR_START = MEM_ADDRW'(64'(`VX_MEM_LMEM_BASE_ADDR) >> MEM_ASHIFT);
    localparam [MEM_ADDRW-1:0] LMEM_ADDR_END   = MEM_ADDRW'(LMEM_ADDR_END64);

    wire [`VX_CFG_NUM_LSU_LANES-1:0] wr_mask;
    wire [15:0] bitmap_half_mask;
    wire [`VX_CFG_NUM_LSU_LANES-1:0] bitmap_small_k_mask;
    wire [31:0] bitmap_s1_mask;

    reg [K_W-1:0]   K;
    reg [3:0]       fmt_s;
    reg [3:0]       fmt_d;
`ifdef VX_CFG_TCU_TYPE_TFR
    // The descriptor's format field is 4 bits (see the kernel's static_asserts);
    // every format this datapath supports fits, so zero-extension is exact.
    wire [TCU_FMT_WIDTH-1:0] fmt_s5 = TCU_FMT_WIDTH'(fmt_s);
    // The destination format is not consulted on this path: the accumulator
    // always flushes fp32. Only the legacy FEOP and its accumulator read it.
    `UNUSED_VAR (fmt_d)
`endif
    reg [1:0]       sparsity; // 0: Dense x Dense, 1: Dense x Sparse, 2: Sparse x Sparse

    reg [`VX_CFG_XLEN-1:0] a_tile_addr;
    reg             a_tile_addr_valid;                             // Is set to false when all A blocks have been requested
    reg [`VX_CFG_XLEN-1:0] a_req_blocks_remaining;                        // Requested to be fetched
    reg [A_BUF_SLOTS-1:0] a_blk_rq_bits;
    reg [A_BUF_SLOTS-1:0] a_blk_ld_bits;
    reg [A_BUF_SLOTS-1:0] a_active_block;
    reg [A_BUF_SLOTS-1:0] a_load_block;
    reg [A_BUF_SLOTS-1:0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] A_buffered; // Holds loaded data to be processed
`ifndef TCU_DISABLE_S1
    // Per-nibble non-zero flags of A_buffered, set as A arrives (S1 A-row bitmap).
    reg [A_BUF_SLOTS-1:0][`VX_CFG_NUM_LSU_LANES-1:0][7:0] A_nz;
`endif

    reg [`VX_CFG_XLEN-1:0] b_tile_addr;
    reg             b_tile_addr_valid;                             // Is set to false when all B blocks have been requested
    reg [`VX_CFG_XLEN-1:0] b_req_blocks_remaining;                        // Requested to be fetched
    reg [B_BUF_SLOTS-1:0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] B_buffered; // Holds loaded data to be processed
    reg [B_BUF_SLOTS-1:0] b_blk_rq_bits;
    reg [B_BUF_SLOTS-1:0] b_blk_ld_bits;
    reg [B_BUF_SLOTS-1:0] b_active_block;
    reg [B_BUF_SLOTS-1:0] b_load_block;

    reg [`VX_CFG_XLEN-1:0] a_bitmap_addr;
    reg [`VX_CFG_XLEN-1:0] b_bitmap_addr;
    reg             bitmap_addr_valid;
    reg [`VX_CFG_XLEN-1:0] bitmap_req_blocks_remaining;   // Requested to be fetched
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_blk_rq_bits;
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_blk_ld_bits;
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_active_block;
    reg [`VX_CFG_XLEN-1:0] bitmap_blocks_loaded;                                    // Loaded but not processed
    reg [BITMAP_BUF_SLOTS-1:0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] Bitmap_buffered; // Holds loaded data to be processed

    reg [`VX_CFG_XLEN-1:0]                      c_tile_addr;
    reg                                  c_tile_addr_valid;    // Is set to false when all C blocks have been requested
    // Null-C flag, latched at accept. c_tile_addr cannot stand in for it: it
    // clears when the last C request issues, before its responses return.
    reg                                  c_is_null_r;
    // Zero-init: a dense init op with a null C skips the C preload. Every
    // accumulator entry then receives exactly one product in k-set 0, which
    // is written with overwrite=1 (pipe_fmul_ctrl). Sparse modes may skip an
    // entry in set 0, so they keep the preload.
    reg                                  zero_init_r;
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_requested;     // Requested to be fetched
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_loaded;        // Loaded but not accumulated
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_accumulated;   // Accumulated / loaded (once loaded they are directly accumulated)

    reg [`MAX(0, C_BUF_SLOTS-1):0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] C_buffered; // Holds loaded data to be accumulated
    `UNUSED_VAR (C_buffered); // Only used when C_BUF_SLOTS > 0


    reg [`VX_CFG_XLEN-1:0] d_tile_addr;
    reg                    busy_r;

    wire op_ctx_full;
    wire execute_ready_no_txbar = (execute_if.data.op_type == INST_TCU_MMA_OP) && (~busy_r) && (~mqueue_full) && (~op_ctx_full);
    wire execute_txbar_req = execute_if.valid && execute_ready_no_txbar;
    assign execute_if.ready = execute_ready_no_txbar && txbar_bus_if.ready;
    wire execute_fire = execute_txbar_req && txbar_bus_if.ready;

    // Format decode, registered at accept (fmt_s is constant for the op).
    reg  [1:0] lg_i_ratio;
    reg  [3:0] i_ratio;
    reg  [OFFSET_W:0] window_elems_r;
    wire [1:0] lg_i_ratio_imm = calc_lg_i_ratio(fmt_s_imm);
    assign window_elems = 32'(window_elems_r);

    // Delayed FEOP-valid (from FMUL-latency pipe), forward-declared for feop_enable.
    wire valid_in_delayed;
    wire first_set_overwrite;   // valid_in_delayed's set-0 zero-init write (pipe_fmul_ctrl)
`ifdef VX_CFG_TCU_TYPE_TFR
    // Credit-based back-pressure (D1). The FEOP pipeline never stalls -- it has
    // no clock enable at all and its valid bits simply flow. Issue stops instead,
    // while any accumulator queue lacks room for the steps already in flight,
    // which makes a refusal at the write port impossible.
    //
    // Margin of 2: one because credit_ok is registered (a queue may have grown
    // since it was sampled), one for the step issuing this cycle.
    localparam CREDIT_MARGIN = 2;
    localparam CREDIT_LIMIT  = TCU_FEOP_XBAR_QUEUE_DEPTH - FEOP_LATENCY - CREDIT_MARGIN;
    wire credit_ok;
    wire feop_enable = 1'b1;
    `UNUSED_VAR (accu_queues_ready)
`else
    // feop_enable handles back-pressure: the FEOP pipe stalls while the
    // accumulator cannot accept a write.
    wire feop_enable = accu_queues_ready;
`endif

    wire mem_stall = valid_out && ~lsu_req_if.req_ready;

    /* Processing of execute_if data */
    wire [`VX_CFG_XLEN-1:0] a_tile_addr_imm = execute_if.data.rs1_data[0];
    wire [`VX_CFG_XLEN-1:0] b_tile_addr_imm = execute_if.data.rs1_data[1];
    wire [`VX_CFG_XLEN-1:0] c_tile_addr_imm = execute_if.data.rs1_data[2];
    wire [`VX_CFG_XLEN-1:0] d_tile_addr_imm = execute_if.data.rs1_data[3];

    wire [`VX_CFG_XLEN-1:0] a_bitmap_addr_imm = execute_if.data.rs2_data[0];
    wire [`VX_CFG_XLEN-1:0] b_bitmap_addr_imm = execute_if.data.rs2_data[1];
    wire [`VX_CFG_XLEN-1:0] txbar_bar_id_imm  = execute_if.data.rs2_data[2];
    wire flush_flag_imm                = execute_if.data.rs2_data[3][0];
    wire init_flag_imm                 = execute_if.data.rs2_data[3][1];
    wire [1:0] sparsity_imm            = execute_if.data.rs2_data[3][3:2];
    wire [3:0] fmt_s_imm               = execute_if.data.rs2_data[3][7:4];
    wire [3:0] fmt_d_imm               = execute_if.data.rs2_data[3][11:8];
    wire [5:0] b_blocks_imm            = execute_if.data.rs2_data[3][17:12];
    wire [5:0] a_blocks_imm            = execute_if.data.rs2_data[3][23:18];
    wire [7:0] K_imm                   = execute_if.data.rs2_data[3][31:24];

    reg  init_r;
    reg  flush_r;
    wire init_flag  = init_r  | (execute_fire ? init_flag_imm : 1'b0);
    wire flush_flag = flush_r | (execute_fire ? flush_flag_imm : 1'b0);
    // Evaluated at execute_fire only (see zero_init_r).
    wire zero_init  = init_flag && (c_tile_addr_imm == '0) && (sparsity_imm == 2'b00);

    // A request stream is ready while it has blocks left and a free buffer
    // slot; A/B/bitmap wait for the C preload to finish.
    // One request in flight per stream (~*_req_pending_r): the per-stream
    // lane-coverage merge cannot separate partial beats of two same-stream
    // requests. LMEM latency is well below the 32-cycle block consumption
    // period, so a prefetch depth of 1 does not throttle A/B streaming.
    wire a_req_ready = a_tile_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (a_blk_rq_bits != '1) && ~a_req_pending_r; // A requests start only after the ACCU is initialized with the values of C
    wire b_req_ready = b_tile_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (b_blk_rq_bits != '1) && ~b_req_pending_r;
    wire bitmap_req_ready = bitmap_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (bitmap_blk_rq_bits != '1) && ~bm_req_pending_r; // Bitmap requests start only after the ACCU is initialized with the values of C
    wire c_req_ready = init_flag && c_tile_addr_valid && ~c_req_pending_r;

    /* Set when execute_fire and Cleared when result_fire */
    wire busy = busy_r || execute_fire;
    // Cycles the TCU stalls on full accumulator queues (reported in traces).
    reg [`VX_CFG_XLEN-1:0] full_queue_stall_cycles;

    always @ (posedge clk) begin
        if (reset) begin

            a_tile_addr        <= '0;
            a_tile_addr_valid  <= 1'b0;
            a_req_blocks_remaining <= '0;
            a_blk_rq_bits      <= '0;
            a_blk_ld_bits      <= '0;
            a_active_block     <= '0;
            a_load_block       <= '0;
            A_buffered         <= '0;
        `ifndef TCU_DISABLE_S1
            A_nz               <= '0;
        `endif

            b_tile_addr        <= '0;
            b_tile_addr_valid  <= 1'b0;
            b_req_blocks_remaining <= '0;
            b_blk_rq_bits      <= '0;
            b_blk_ld_bits      <= '0;
            b_active_block     <= '0;
            b_load_block       <= '0;
            B_buffered         <= '0;

            c_tile_addr          <= '0;
            c_tile_addr_valid    <= 1'b0;
            c_is_null_r          <= 1'b1;
            zero_init_r          <= 1'b0;
            c_blocks_requested   <= '0;
            c_blocks_loaded      <= '0;
            c_blocks_accumulated <= '0;
        
            d_tile_addr <= '0;

            a_bitmap_addr           <= '0;
            b_bitmap_addr           <= '0;
            bitmap_addr_valid       <= '0;
            bitmap_req_blocks_remaining <= '0;
            bitmap_blk_rq_bits      <= '0;
            bitmap_blk_ld_bits      <= '0;
            bitmap_active_block     <= '0;
            bitmap_blocks_loaded    <= '0;
            Bitmap_buffered         <= '0;

            full_queue_stall_cycles <= '0;

            set  <= '0;
            step <= '0;
            issuing_done <= 0;

            a_offset <= '0;
            b_offset <= '0;

            result_pending_r <= 1'b0;

            d_line_to_flush <= '0;
            busy_r <= 1'b0;

            init_r  <= 1'b0;
            flush_r <= 1'b0;

            lg_i_ratio     <= '0;
            i_ratio        <= 4'd1;
            window_elems_r <= (OFFSET_W+1)'(`VX_CFG_NUM_LSU_LANES);
        end else begin
            // Initialization
            if (execute_fire) begin
                a_tile_addr_valid <= 1'b1;
                b_tile_addr_valid <= 1'b1;
                // A zero-init op preloads nothing, exactly like a continuation
                // op: no C requests, and the C counters start full so A/B
                // requests are not held behind the preload.
                c_tile_addr_valid <= init_flag && ~zero_init;
                zero_init_r       <= zero_init;
                /*                    sparsity mode                     */
                bitmap_addr_valid <= (sparsity_imm >= 2'd1) ? 1'b1 : 1'b0;

                b_req_blocks_remaining <= '0;
                c_blocks_requested     <= (init_flag && ~zero_init) ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);
                a_blk_rq_bits        <= '0;
                a_blk_ld_bits        <= '0;
                a_active_block       <= 2'b01;
                a_load_block         <= 2'b01;
                b_blk_rq_bits        <= '0;
                b_blk_ld_bits        <= '0;
                b_active_block       <= 2'b01;
                b_load_block         <= 2'b01;
                bitmap_blk_rq_bits   <= '0;
                bitmap_blk_ld_bits   <= '0;
                bitmap_active_block  <= 2'b01;
                c_blocks_loaded      <= (init_flag && ~zero_init) ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);
                bitmap_blocks_loaded <= '0;

                c_blocks_accumulated <= (init_flag && ~zero_init) ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);

                // Compute total blocks using the incoming instruction fields to avoid stale values
                if (sparsity_imm == 2'b00) begin
                    /* a_req_blocks_remaining = K * TCU_TC_M_OP / i_ratio */
                    a_req_blocks_remaining  <= 32'((`VX_CFG_XLEN'(K_imm) << LG_TCU_TC_M_OP) >> (32'(calc_lg_i_ratio(fmt_s_imm)) + LG_REGS_PER_BLOCK));
                    /* b_req_blocks_remaining = K * TCU_TC_N_OP / i_ratio */
                    b_req_blocks_remaining  <= 32'((`VX_CFG_XLEN'(K_imm) << LG_TCU_TC_N_OP) >> (32'(calc_lg_i_ratio(fmt_s_imm)) + LG_REGS_PER_BLOCK));
                    /* No bitmap in dense case */
                    bitmap_req_blocks_remaining <= '0;
                end
            `ifndef TCU_DISABLE_S1
                else if (sparsity_imm == 2'b01) begin
                    /* a_req_blocks_remaining = K * TCU_TC_M_OP / i_ratio */
                    a_req_blocks_remaining  <= 32'((`VX_CFG_XLEN'(K_imm) << LG_TCU_TC_M_OP) >> (32'(calc_lg_i_ratio(fmt_s_imm)) + LG_REGS_PER_BLOCK));
                    /* b_req_blocks_remaining = B_compressed_blocks */
                    b_req_blocks_remaining  <= (`VX_CFG_XLEN)'(b_blocks_imm);
                    /* bitmap_req_blocks_remaining = ceil(K / SETS_PER_S1_BITMAP_BLOCK) */
                    bitmap_req_blocks_remaining <= 32'((`VX_CFG_XLEN'(K_imm) + SETS_PER_S1_BITMAP_BLOCK - 1) >> LG_SETS_PER_S1_BITMAP_BLOCK);
                end
            `endif
                else begin /* s2 case */
                    /* a_req_blocks_remaining = A_compressed_blocks */
                    a_req_blocks_remaining  <= (`VX_CFG_XLEN)'(a_blocks_imm);
                    /* b_req_blocks_remaining = B_compressed_blocks */
                    b_req_blocks_remaining  <= (`VX_CFG_XLEN)'(b_blocks_imm);
                    /* bitmap_req_blocks_remaining = ceil(K * 2 / 32) */
                    bitmap_req_blocks_remaining <= 32'((`VX_CFG_XLEN'(K_imm) + 15) >> 4);
                end

                /* Get configuration from the instruction */
                a_tile_addr <= (`VX_CFG_XLEN)'(a_tile_addr_imm);
                b_tile_addr <= (`VX_CFG_XLEN)'(b_tile_addr_imm);
                c_tile_addr <= (`VX_CFG_XLEN)'(c_tile_addr_imm);
                c_is_null_r <= (c_tile_addr_imm == '0);
                d_tile_addr <= (`VX_CFG_XLEN)'(d_tile_addr_imm);

                a_bitmap_addr <= (`VX_CFG_XLEN)'(a_bitmap_addr_imm);
                b_bitmap_addr <= (`VX_CFG_XLEN)'(b_bitmap_addr_imm);

                K        <= K_imm;
                fmt_s    <= 4'(fmt_s_imm);
                lg_i_ratio     <= lg_i_ratio_imm;
                i_ratio        <= 4'(1 << lg_i_ratio_imm);
                window_elems_r <= (OFFSET_W+1)'(1 << lg_i_ratio_imm) << $clog2(`VX_CFG_NUM_LSU_LANES);
                fmt_d    <= 4'(fmt_d_imm);
                sparsity <= 2'(sparsity_imm);

                full_queue_stall_cycles <= '0;

                result_pending_r <= 1'b0;
            end else if (rd_req_fire) begin
                // Memory Request Handling
                case (grant_onehot)
                    MATRIX_ID_BITS'(1): begin  // Bitmap
                        if (bitmap_req_blocks_remaining == 1) begin
                            a_bitmap_addr     <= '0;
                            b_bitmap_addr     <= '0;
                            bitmap_addr_valid <= 1'b0; // Completed all requests for this tile
                        end else begin
                        `ifndef TCU_DISABLE_S1
                            if (sparsity == 2'd1) begin
                                b_bitmap_addr <= b_bitmap_addr + BYTES_PER_MEM_REQUEST;
                            end else begin
                        `endif
                                a_bitmap_addr <= a_bitmap_addr + (BYTES_PER_MEM_REQUEST >> 1);
                                b_bitmap_addr <= b_bitmap_addr + (BYTES_PER_MEM_REQUEST >> 1);
                        `ifndef TCU_DISABLE_S1
                            end
                        `endif
                        end
                        bitmap_req_blocks_remaining <= bitmap_req_blocks_remaining - 1'b1;
                        if (~last_step_in_bitmap_block) begin
                            bitmap_blk_rq_bits <= {|bitmap_blk_rq_bits, 1'b1};
                        end
                    end
                    MATRIX_ID_BITS'(2): begin  // A
                        if (a_req_blocks_remaining == 1) begin
                            a_tile_addr       <= '0;
                            a_tile_addr_valid <= 1'b0; // Completed all requests for this tile
                        end else begin
                            a_tile_addr <= a_tile_addr + BYTES_PER_MEM_REQUEST;
                        end
                        a_req_blocks_remaining <= a_req_blocks_remaining - 1'b1;
                        if (~last_step_in_block_a) begin
                            a_blk_rq_bits <= {|a_blk_rq_bits, 1'b1};
                        end
                    end
                    MATRIX_ID_BITS'(4): begin  // B
                        if (b_req_blocks_remaining == 1) begin
                            b_tile_addr       <= '0;
                            b_tile_addr_valid <= 1'b0; // Completed all requests for this tile
                        end else begin
                            b_tile_addr <= b_tile_addr + BYTES_PER_MEM_REQUEST;
                        end
                        b_req_blocks_remaining <= b_req_blocks_remaining - 1'b1;
                        if (~last_step_in_block_b) begin
                            b_blk_rq_bits <= {|b_blk_rq_bits, 1'b1};
                        end
                    end
                    MATRIX_ID_BITS'(8): begin  // C
                        if (32'(c_blocks_requested) == (TCU_C_BLOCKS_IN_ACCU - 1)) begin
                            c_tile_addr       <= '0;
                            c_tile_addr_valid <= 1'b0; // Completed all requests for this tile
                        end else begin
                            if (c_tile_addr != '0) begin
                                c_tile_addr <= c_tile_addr + BYTES_PER_MEM_REQUEST;
                            end
                        end
                        c_blocks_requested <= c_blocks_requested + 1'b1;
                    end
                    default: begin
                    end
                endcase
            end else if (result_fire) begin
                full_queue_stall_cycles <= '0;
            end

            if (busy && ~accu_queues_ready) begin
                full_queue_stall_cycles <= full_queue_stall_cycles + 1'b1;
            end
            if (issue_busy) begin
                if (last_step_in_block_a) begin
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(2))) begin
                        a_blk_rq_bits <= (a_blk_rq_bits >> 1);
                    end
                    if (~(rsp_block_done && rsp_matrix_id == MATRIX_ID_BITS'(2))) begin
                        a_blk_ld_bits <= a_blk_ld_bits & ~a_active_block;
                    end
                    a_active_block <= ~a_active_block; // 01->10, 10->01
                end
                if (last_step_in_block_b) begin
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(4))) begin
                        b_blk_rq_bits <= (b_blk_rq_bits >> 1);
                    end
                    if (~(rsp_block_done && rsp_matrix_id == MATRIX_ID_BITS'(4))) begin
                        b_blk_ld_bits <= b_blk_ld_bits & ~b_active_block;
                    end
                    b_active_block <= ~b_active_block; // 01->10, 10->01
                end
                if (last_step_in_bitmap_block) begin
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(1))) begin
                        bitmap_blk_rq_bits <= (bitmap_blk_rq_bits >> 1);
                    end
                    if (~(rsp_block_done && rsp_matrix_id == MATRIX_ID_BITS'(1))) begin
                        bitmap_blk_ld_bits <= bitmap_blk_ld_bits & ~bitmap_active_block;
                    end
                    bitmap_active_block <= ~bitmap_active_block; // 01->10, 10->01
                end
            end

            // Load data mechanism. Data merges per-lane under the response
            // mask (the packer may split a request into partial beats); all
            // block-level bookkeeping advances only on rsp_block_done.
            if (rd_rsp_fire) begin
                case (rsp_matrix_id)
                    MATRIX_ID_BITS'(1): begin  // Bitmap
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                Bitmap_buffered[bitmap_blk_ld_bits[0]][l] <= tcu_lsu_mem_if.rsp_data.data[l]; // Double buffering
                            end
                        end
                        if (rsp_block_done) begin
                            bitmap_blocks_loaded <= bitmap_blocks_loaded + 1'b1; // Necessary for the bitmap_block_ready
                            if (~last_step_in_bitmap_block) begin
                                bitmap_blk_ld_bits <= {|bitmap_blk_ld_bits, 1'b1}; // 00->01, 01->11, 10->11
                            end else begin
                                bitmap_blk_ld_bits <= ~bitmap_blk_ld_bits; // 01->10, 10->01
                            end
                        end
                    end
                    MATRIX_ID_BITS'(2): begin  // A
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                A_buffered[~a_load_block[0]][l] <= tcu_lsu_mem_if.rsp_data.data[l]; // Double buffering
                            `ifndef TCU_DISABLE_S1
                                for (integer j = 0; j < 8; ++j) begin
                                    A_nz[~a_load_block[0]][l][j] <= |tcu_lsu_mem_if.rsp_data.data[l][4*j +: 4];
                                end
                            `endif
                            end
                        end
                        if (rsp_block_done) begin
                            if (~last_step_in_block_a) begin
                                a_blk_ld_bits <= ((a_blk_ld_bits == 2'b00) ? a_load_block : 2'b11); // 00->a_load_block, 01->11, 10->11
                            end else begin
                                a_blk_ld_bits <= ~a_blk_ld_bits; // 01->10, 10->01
                            end
                            a_load_block <= ~a_load_block; // 01->10, 10->01
                        end
                    end
                    MATRIX_ID_BITS'(4): begin  // B
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                B_buffered[~b_load_block[0]][l] <= tcu_lsu_mem_if.rsp_data.data[l]; // Double buffering
                            end
                        end
                        if (rsp_block_done) begin
                            if (~last_step_in_block_b) begin
                                b_blk_ld_bits <= ((b_blk_ld_bits == 2'b00) ? b_load_block : 2'b11); // 00->b_load_block, 01->11, 10->11
                            end else begin
                                b_blk_ld_bits <= ~b_blk_ld_bits; // 01->10, 10->01
                            end
                            b_load_block <= ~b_load_block; // 01->10, 10->01
                        end
                    end
                    MATRIX_ID_BITS'(8): begin  // C
                        if (~accumulate_c && C_BUF_SLOTS > 0) begin
                            for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin
                                if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                    C_buffered[c_blocks_loaded % (C_BUF_SLOTS+1)][l] <= tcu_lsu_mem_if.rsp_data.data[l]; // Buffer until you can accumulate them all in 1 cycle
                                end
                            end
                        end
                        if (rsp_block_done) begin
                            c_blocks_loaded <= c_blocks_loaded + 1'b1;
                        end
                    end
                    default: begin
                    end
                endcase

                if (accumulate_c) begin
                    c_blocks_accumulated <= c_blocks_accumulated + (C_BUF_SLOTS + 1);
                end
            end

            if (execute_fire || result_fire) begin
                set  <= '0;
                step <= '0;
                issuing_done <= 1'b0;
            end
            if (last_step_in_execution && feop_enable) begin
                issuing_done <= 1'b1;
            end
            if (issue_busy) begin
                if (last_step_in_set) begin
                    step <= '0;
                    set  <= set + K_W'(1);
                end else begin
                    step <= step + LG_TCU_FEOP_STEPS'(1);
                end
            end

            if (execute_fire) begin
                a_offset <= '0;
                b_offset <= '0;
            end
            if (last_step_in_set) begin
                a_offset <= OFFSET_W'((32'(a_offset) + 32'(a_set_elems)) & (window_elems - 1));
                b_offset <= OFFSET_W'((32'(b_offset) + 32'(b_non_zeros)) & (window_elems - 1));
            end

            if (~execute_fire && result_fire) begin
                result_pending_r <= 1'b0;
            end
            if (~execute_fire && result_pulse) begin
                result_pending_r <= 1'b1;
            end

            if (execute_fire) begin
                d_line_to_flush <= '0;
                busy_r <= 1'b1;
            end
            if (ready_to_flush && ~mem_stall) begin
                d_line_to_flush <= d_line_to_flush + (LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE + 1)'(1);
            end
            if (wr_req_fire) begin
                d_tile_addr <= d_tile_addr + (LSU_WORD_SIZE << LG_TCU_FEOP_BLOCK_N_SIZE);
            end
            if (result_fire) begin
                busy_r <= 1'b0;
                init_r  <= 1'b0;
                flush_r <= 1'b0;
            end

            if (execute_fire) begin
                init_r  <= init_flag_imm;
                flush_r <= flush_flag_imm;
                `TRACE(2, ("init_flag=%b, flush_flag=%b\n", init_flag, flush_flag));
            end
        end
    end

// INITIALIZATIONS & FSM
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY REQUEST HANDLING



    // Read-request arbitration:  [3] C,  [2] B,  [1] A,  [0] bitmap
    assign reqs = {c_req_ready, b_req_ready, a_req_ready, bitmap_req_ready};

    assign rd_req_fire = rd_req_valid && lsu_req_if.req_ready;

    VX_rr_rot_arbiter #(
        .NUM_REQS (MATRIX_ID_BITS)
    ) cyclic_loader (
        .clk          (clk),
        .reset        (reset),
        .requests     (reqs),
        .grant_onehot (grant_onehot),
        .grant_valid  (rd_req_valid),
        .grant_ready  (lsu_req_if.req_ready)
    );

    assign req_rd_addr = grant_onehot[0] ? b_bitmap_addr : // Convenient for s1 case, not used in s2 case
                         grant_onehot[1] ? a_tile_addr   :
                         grant_onehot[2] ? b_tile_addr   :
                         c_is_null_r     ? a_tile_addr   : c_tile_addr; // If C is NULL, redirect the (discarded) read at A

// MEMORY REQUEST HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY RESPONSE HANDLING

    assign rd_rsp_fire = tcu_lsu_mem_if.rsp_valid && tcu_lsu_mem_if.rsp_ready;

    // The matrix travels in tag.value as a binary index (see the request
    // side); inside this module it is one-hot.
    assign rsp_matrix_id = MATRIX_ID_BITS'(1) << tcu_lsu_mem_if.rsp_data.tag.value[MATRIX_IDX_BITS-1:0];

    // ---- Partial-response merging ----
    // A request's lanes may return over several beats. Data merges per lane
    // on every beat; block bookkeeping advances only on rsp_block_done, once
    // the request's full lane mask has been covered.
    reg  [`VX_CFG_NUM_LSU_LANES-1:0] a_rsp_seen, b_rsp_seen, c_rsp_seen, bm_rsp_seen;
    reg  [`VX_CFG_NUM_LSU_LANES-1:0] a_req_mask_r, b_req_mask_r, c_req_mask_r, bm_req_mask_r;
    wire [`VX_CFG_NUM_LSU_LANES-1:0] rsp_seen_cur =
        (rsp_matrix_id == MATRIX_ID_BITS'(1)) ? bm_rsp_seen :
        (rsp_matrix_id == MATRIX_ID_BITS'(2)) ? a_rsp_seen  :
        (rsp_matrix_id == MATRIX_ID_BITS'(4)) ? b_rsp_seen  :
                                                c_rsp_seen;
    wire [`VX_CFG_NUM_LSU_LANES-1:0] rsp_expected_mask =
        (rsp_matrix_id == MATRIX_ID_BITS'(1)) ? bm_req_mask_r :
        (rsp_matrix_id == MATRIX_ID_BITS'(2)) ? a_req_mask_r  :
        (rsp_matrix_id == MATRIX_ID_BITS'(4)) ? b_req_mask_r  :
                                                c_req_mask_r;
    wire [`VX_CFG_NUM_LSU_LANES-1:0] rsp_seen_next = rsp_seen_cur | tcu_lsu_mem_if.rsp_data.mask;
    assign rsp_block_done = rd_rsp_fire && ((rsp_seen_next & rsp_expected_mask) == rsp_expected_mask);

    // Merge buffer for the direct (C_BUF_SLOTS == 0) C path, which feeds the
    // accumulator straight from the response bus and has no backing buffer
    // to accumulate partial beats into.
    reg  [`VX_CFG_NUM_LSU_LANES-1:0][`VX_CFG_XLEN-1:0] c_partial_r;
    wire [`VX_CFG_NUM_LSU_LANES-1:0][`VX_CFG_XLEN-1:0] c_block_merged;
    for (genvar l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin : g_c_block_merged
        assign c_block_merged[l] = tcu_lsu_mem_if.rsp_data.mask[l] ? tcu_lsu_mem_if.rsp_data.data[l] : c_partial_r[l];
    end


    always @(posedge clk) begin
        if (reset) begin
            a_rsp_seen       <= '0;
            b_rsp_seen       <= '0;
            c_rsp_seen       <= '0;
            bm_rsp_seen      <= '0;
            a_req_mask_r     <= '1;
            b_req_mask_r     <= '1;
            c_req_mask_r     <= '1;
            bm_req_mask_r    <= '1;
            a_req_pending_r  <= 1'b0;
            b_req_pending_r  <= 1'b0;
            c_req_pending_r  <= 1'b0;
            bm_req_pending_r <= 1'b0;
            c_partial_r      <= '0;
        end else begin
            if (rd_req_fire) begin
                case (grant_onehot)
                    MATRIX_ID_BITS'(1): begin
                        bm_req_mask_r    <= lsu_req_if.req_data.mask;
                        bm_req_pending_r <= 1'b1;
                    end
                    MATRIX_ID_BITS'(2): begin
                        a_req_mask_r     <= lsu_req_if.req_data.mask;
                        a_req_pending_r  <= 1'b1;
                    end
                    MATRIX_ID_BITS'(4): begin
                        b_req_mask_r     <= lsu_req_if.req_data.mask;
                        b_req_pending_r  <= 1'b1;
                    end
                    default: begin
                        c_req_mask_r     <= lsu_req_if.req_data.mask;
                        c_req_pending_r  <= 1'b1;
                    end
                endcase
            end
            if (rd_rsp_fire) begin
                case (rsp_matrix_id)
                    MATRIX_ID_BITS'(1): begin
                        bm_rsp_seen <= rsp_block_done ? '0 : rsp_seen_next;
                        if (rsp_block_done) begin
                            bm_req_pending_r <= 1'b0;
                        end
                    end
                    MATRIX_ID_BITS'(2): begin
                        a_rsp_seen  <= rsp_block_done ? '0 : rsp_seen_next;
                        if (rsp_block_done) begin
                            a_req_pending_r <= 1'b0;
                        end
                    end
                    MATRIX_ID_BITS'(4): begin
                        b_rsp_seen  <= rsp_block_done ? '0 : rsp_seen_next;
                        if (rsp_block_done) begin
                            b_req_pending_r <= 1'b0;
                        end
                    end
                    default: begin
                        c_rsp_seen  <= rsp_block_done ? '0 : rsp_seen_next;
                        if (rsp_block_done) begin
                            c_req_pending_r <= 1'b0;
                        end
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; ++l) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                c_partial_r[l] <= tcu_lsu_mem_if.rsp_data.data[l];
                            end
                        end
                    end
                endcase
            end
        end
    end

    // Accept a response when its stream's buffer has a free slot.
    assign tcu_lsu_mem_if.rsp_ready = (rsp_matrix_id == MATRIX_ID_BITS'(1)) ? (bitmap_blk_ld_bits != '1) :
                                      (rsp_matrix_id == MATRIX_ID_BITS'(2)) ? (a_blk_ld_bits != '1) :
                                      (rsp_matrix_id == MATRIX_ID_BITS'(4)) ? (b_blk_ld_bits != '1) :
                                      (rsp_matrix_id == MATRIX_ID_BITS'(8)) ? ((c_blocks_loaded % (C_BUF_SLOTS+1) == C_BUF_SLOTS) ? accu_enable : 1'b1) :
                                      1'b0;

    assign accumulate_c = (rsp_matrix_id == MATRIX_ID_BITS'(8)) && rsp_block_done && (c_blocks_loaded - c_blocks_accumulated == C_BUF_SLOTS) && accu_enable;

    assign c_blk_idx = ($clog2(TCU_FEOP_STEPS+1))'(c_blocks_accumulated >> LG_C_BLOCKS_PER_FEOP_BLOCK);

    assign C_feop_block = c_is_null_r ? '0 : c_block_merged;

// MEMORY RESPONSE HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// WORK ASSIGNMENT TO FEOPS

// *************************************************************************************************************
// FSM VARIABLES

    // Steps in this k-set: only FEOP blocks that hold non-zero A rows / B columns.
    // vertical_steps, horizontal_steps and set_steps are registered with the
    // set's bitmaps (g_bm_cand).

    /* horizontal_steps_safe avoids division with 0 */
    assign horizontal_steps_safe = (horizontal_steps == '0) ? (LG_TCU_FEOP_N_STEPS+1)'(1) : horizontal_steps;
    // m = (step / horizontal_steps) * BLOCK_M, n = (step % horizontal_steps) * BLOCK_N.
    // horizontal_steps <= TCU_FEOP_N_STEPS, so select among constant divisions.
    wire [TCU_FEOP_N_STEPS:1][LG_TCU_FEOP_STEPS-1:0] step_div;
    wire [TCU_FEOP_N_STEPS:1][LG_TCU_FEOP_STEPS-1:0] step_mod;
    for (genvar d = 1; d <= TCU_FEOP_N_STEPS; ++d) begin : g_step_div
        assign step_div[d] = step / LG_TCU_FEOP_STEPS'(d);
        assign step_mod[d] = step % LG_TCU_FEOP_STEPS'(d);
    end
    assign m = LG_TCU_TC_M_OP'(step_div[horizontal_steps_safe] << LG_TCU_FEOP_BLOCK_M_SIZE);
    assign n = LG_TCU_TC_N_OP'(step_mod[horizontal_steps_safe] << LG_TCU_FEOP_BLOCK_N_SIZE);

    assign last_step_in_set       = (step == LG_TCU_FEOP_STEPS'(set_steps - (LG_TCU_FEOP_STEPS+1)'(1))) && issue_busy;
    assign last_step_in_block_a   = last_step_in_set && last_set_in_block_a;
    assign last_step_in_block_b   = last_step_in_set && last_set_in_block_b;
    assign last_step_in_execution = last_step_in_set && (set == K - K_W'(1));

`ifndef TCU_DISABLE_S1
    assign last_step_in_bitmap_block = last_step_in_set
                                  && (((sparsity == 2'd1) && (((32'(set) + 1) & (SETS_PER_S1_BITMAP_BLOCK-1)) == 0))
                                   || ((sparsity == 2'd2) && (((32'(set) + 1) & (SETS_PER_S2_BITMAP_BLOCK-1)) == 0)));
`else
    assign last_step_in_bitmap_block = last_step_in_set && (sparsity == 2'd2) && (((32'(set) + 1) & (SETS_PER_S2_BITMAP_BLOCK-1)) == 0);
`endif

    always @ (posedge clk) begin
        if (~reset && last_step_in_execution) begin
            `TRACE(2, ("%t: last_step_in_execution: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (~reset && issuing_done) begin
            `TRACE(2, ("%t: issuing_done: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
    end

    /* Stalls when no new data have arrived  */
    // TODO: Make B available not only when it is written to B_BUFF but also the moment it arrives from LMEM
    assign bitmap_block_ready = (sparsity == 2'd2) ? (bitmap_blocks_loaded > (32'(set) >> LG_SETS_PER_S2_BITMAP_BLOCK)) :
                            `ifndef TCU_DISABLE_S1
                              (sparsity == 2'd1) ? (bitmap_blocks_loaded > (32'(set) >> LG_SETS_PER_S1_BITMAP_BLOCK)) :
                            `endif
                              1'b1; // Always ready in dense case

    // A storage format depends on sparsity mode:
    // - s2: A is compressed, so set span is number of non-zeros.
    // - s1/s0: A is dense in memory, so each set always spans full M dimension.
    assign a_set_elems = (sparsity == 2'd2) ? a_non_zeros : (LG_TCU_TC_M_OP+1)'(TCU_TC_M_OP);

    assign a_curr_loaded = |(a_blk_ld_bits & a_active_block);
    assign a_next_loaded = |(a_blk_ld_bits & ~a_active_block);
    // In s1/s0 modes, A is dense in memory and bitmap extraction scans the whole set.
    // In s2 mode, A is compressed and only non-zero payload is needed.
    assign a_window_need = (sparsity == 2'd2) ? `MIN((32'(a_non_zeros)), (32'(m) + TCU_FEOP_BLOCK_M_SIZE)) : 32'(a_set_elems);
    assign a_window_ready = a_curr_loaded && (((32'(a_offset) + a_window_need) <= window_elems) || a_next_loaded);

    assign b_curr_loaded = |(b_blk_ld_bits & b_active_block);
    assign b_next_loaded = |(b_blk_ld_bits & ~b_active_block);
    assign b_window_ready = b_curr_loaded && (((32'(b_offset) + `MIN((32'(b_non_zeros)), (32'(n) + TCU_FEOP_BLOCK_N_SIZE))) <= window_elems) || b_next_loaded);

    // Products start only once the C preload is fully accumulated: the
    // accumulator write port takes a C block or a product per cycle, never both.
    wire c_preload_done = (c_blocks_accumulated == ($clog2(TCU_C_BLOCKS_IN_ACCU)+1)'(TCU_C_BLOCKS_IN_ACCU));

`ifdef VX_CFG_TCU_TYPE_TFR
    assign issue_busy = busy_r && credit_ok && ~issuing_done && ~accumulate_c && ~c_write_r && c_preload_done
`else
    assign issue_busy = busy_r && feop_enable && ~issuing_done && ~accumulate_c && ~c_write_r && c_preload_done
`endif
                      && a_window_ready
                      && b_window_ready
                      && bitmap_block_ready
                      && bitmap_fresh;

// FSM VARIABLES
// *************************************************************************************************************
// BITMAP PROCESSING

    // Set Extraction
    assign a_non_zeros = TCU_TC_M_OP - a_zeros;
    assign b_non_zeros = TCU_TC_N_OP - b_zeros;

`ifndef TCU_DISABLE_S1
    // S1 A-row bitmap (elements down to 4 bits) from a block's nibble flags.
    // An S1 set is TCU_TC_M_OP dense elements aligned within the block: set k
    // holds elements [32k, 32k+32), element e sits in lane e / i_ratio and
    // covers 8 / i_ratio nibbles of it.
    `STATIC_ASSERT ((`VX_CFG_XLEN == 32) && (`VX_CFG_NUM_LSU_LANES == TCU_TC_M_OP),
        ("tcu_op_core: S1 bitmap assumes 32-bit lanes and one lane per A row"))
    localparam LG_LANES = $clog2(`VX_CFG_NUM_LSU_LANES);

    function automatic [TCU_TC_M_OP-1:0] s1_a_bitmap (
        input [`VX_CFG_NUM_LSU_LANES-1:0][7:0] nz,
        input [2:0]                            kset,
        input [1:0]                            lg
    );
        for (int i = 0; i < TCU_TC_M_OP; ++i) begin
            case (lg)
                2'd0:    s1_a_bitmap[i] = |nz[i];
                2'd1:    s1_a_bitmap[i] = |nz[LG_LANES'((32'(kset) << (LG_LANES - 1)) + (i >> 1))][(i & 1) * 4 +: 4];
                2'd2:    s1_a_bitmap[i] = |nz[LG_LANES'((32'(kset) << (LG_LANES - 2)) + (i >> 2))][(i & 3) * 2 +: 2];
                default: s1_a_bitmap[i] =  nz[LG_LANES'((32'(kset) << (LG_LANES - 3)) + (i >> 3))][i & 7];
            endcase
        end
    endfunction
`endif

    function automatic [LG_TCU_TC_M_OP:0] count_zeros (input [TCU_TC_M_OP-1:0] bm);
        count_zeros = '0;
        for (int i = 0; i < TCU_TC_M_OP; ++i) begin
            count_zeros = count_zeros + (LG_TCU_TC_M_OP+1)'(~bm[i]);
        end
    endfunction
    `STATIC_ASSERT (TCU_TC_M_OP == TCU_TC_N_OP, ("tcu_op_core: count_zeros assumes square sets"))

    // The issue loop reads the current set's bitmaps and zero counts from
    // registers. Every cycle they load the set that is current next cycle:
    // this set (k=0), or the next one on its last step (k=1). A capture that
    // may predate an op accept or a bitmap/A write is not fresh; issue then
    // waits a cycle for the recapture.
    wire [1:0][TCU_TC_M_OP-1:0]      a_bm_cand;
    wire [1:0][TCU_TC_N_OP-1:0]      b_bm_cand;
    wire [1:0][LG_TCU_TC_M_OP:0]     a_zeros_cand;
    wire [1:0][LG_TCU_TC_N_OP:0]     b_zeros_cand;
    wire [1:0][LG_TCU_FEOP_M_STEPS:0] vsteps_cand;
    wire [1:0][LG_TCU_FEOP_N_STEPS:0] hsteps_cand;
    wire [1:0][LG_TCU_FEOP_STEPS:0]   ssteps_cand;

    for (genvar k = 0; k < 2; ++k) begin : g_bm_cand
        wire [K_W-1:0] cset = set + K_W'(k);

        wire [TCU_TC_M_OP-1:0] a_s2 = Bitmap_buffered[(cset >> LG_SETS_PER_S2_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][cset & (SETS_PER_S2_BITMAP_BLOCK-1)];
        wire [TCU_TC_N_OP-1:0] b_s2 = Bitmap_buffered[(cset >> LG_SETS_PER_S2_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][(`VX_CFG_NUM_LSU_LANES >> 1) + (cset & (SETS_PER_S2_BITMAP_BLOCK-1))];
    `ifndef TCU_DISABLE_S1
        // The next S1 set advances a_offset by one set and, past the block
        // end, moves to the other A block (as the issue FSM does).
        wire [OFFSET_W-1:0] coff = (k == 0) ? a_offset : OFFSET_W'((32'(a_offset) + TCU_TC_M_OP) & (window_elems - 1));
        // An S1 set is always TCU_TC_M_OP elements, so its block crossing
        // needs no set count.
        wire                s1_cross = (32'(a_offset) + TCU_TC_M_OP) >= window_elems;
        wire                cblk = (k == 0) ? ~a_active_block[0] : (s1_cross ? a_active_block[0] : ~a_active_block[0]);
        wire [TCU_TC_M_OP-1:0] a_s1 = s1_a_bitmap(A_nz[cblk], 3'(coff >> LG_TCU_TC_M_OP), lg_i_ratio);
        wire [TCU_TC_N_OP-1:0] b_s1 = Bitmap_buffered[(cset >> LG_SETS_PER_S1_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][cset & (SETS_PER_S1_BITMAP_BLOCK-1)];
    `endif

        assign a_bm_cand[k] = (sparsity == 2'd2) ? a_s2 :
                            `ifndef TCU_DISABLE_S1
                              (sparsity == 2'd1) ? a_s1 :
                            `endif
                                                   '1;
        assign b_bm_cand[k] = (sparsity == 2'd2) ? b_s2 :
                            `ifndef TCU_DISABLE_S1
                              (sparsity == 2'd1) ? b_s1 :
                            `endif
                                                   '1;
        assign a_zeros_cand[k] = count_zeros(a_bm_cand[k]);
        assign b_zeros_cand[k] = count_zeros(b_bm_cand[k]);

        // Steps in the set: only FEOP blocks that hold non-zero A rows / B columns.
        wire [LG_TCU_TC_M_OP:0] a_nz = (LG_TCU_TC_M_OP+1)'(TCU_TC_M_OP) - a_zeros_cand[k];
        wire [LG_TCU_TC_N_OP:0] b_nz = (LG_TCU_TC_N_OP+1)'(TCU_TC_N_OP) - b_zeros_cand[k];
        assign vsteps_cand[k] = (LG_TCU_FEOP_M_STEPS+1)'((32'(a_nz) + (TCU_FEOP_BLOCK_M_SIZE - 1)) >> LG_TCU_FEOP_BLOCK_M_SIZE);
        assign hsteps_cand[k] = (LG_TCU_FEOP_N_STEPS+1)'((32'(b_nz) + (TCU_FEOP_BLOCK_N_SIZE - 1)) >> LG_TCU_FEOP_BLOCK_N_SIZE);
        wire [LG_TCU_FEOP_STEPS:0] ssteps_raw = (LG_TCU_FEOP_STEPS+1)'(vsteps_cand[k] * hsteps_cand[k]);
        assign ssteps_cand[k] = (ssteps_raw == '0) ? (LG_TCU_FEOP_STEPS+1)'(1) : ssteps_raw;
    end

    wire bitmap_stale = execute_fire
                     || ((sparsity != 2'd0) && rd_rsp_fire
                         && ((rsp_matrix_id == MATRIX_ID_BITS'(1)) || ((sparsity == 2'd1) && (rsp_matrix_id == MATRIX_ID_BITS'(2)))));

    always @(posedge clk) begin
        if (reset) begin
            a_bitmap_in  <= '1;
            b_bitmap_in  <= '1;
            a_zeros      <= '0;
            b_zeros      <= '0;
            vertical_steps   <= '0;
            horizontal_steps <= '0;
            set_steps        <= (LG_TCU_FEOP_STEPS+1)'(1);
            bitmap_fresh <= 1'b0;
        end else begin
            a_bitmap_in  <= last_step_in_set ? a_bm_cand[1]    : a_bm_cand[0];
            b_bitmap_in  <= last_step_in_set ? b_bm_cand[1]    : b_bm_cand[0];
            a_zeros      <= last_step_in_set ? a_zeros_cand[1] : a_zeros_cand[0];
            b_zeros      <= last_step_in_set ? b_zeros_cand[1] : b_zeros_cand[0];
            vertical_steps   <= last_step_in_set ? vsteps_cand[1] : vsteps_cand[0];
            horizontal_steps <= last_step_in_set ? hsteps_cand[1] : hsteps_cand[0];
            set_steps        <= last_step_in_set ? ssteps_cand[1] : ssteps_cand[0];
            bitmap_fresh <= ~bitmap_stale;
        end
    end

    VX_tcu_32_way_sorter a_sorter (
        .in_bitmap      (a_bitmap_in),
        .out_bitmap     (a_bitmap_out),
        .out_address    (a_addresses),
        `UNUSED_PIN     (zero_pop_count)
    );

    VX_tcu_32_way_sorter b_sorter (
        .in_bitmap      (b_bitmap_in),
        .out_bitmap     (b_bitmap_out),
        .out_address    (b_addresses),
        `UNUSED_PIN     (zero_pop_count)
    );

// BITMAP PROCESSING
// *************************************************************************************************************
// ADDRESS EXTRACTION

    assign a_step_addresses = a_addresses[m +: TCU_FEOP_BLOCK_M_SIZE];
    assign b_step_addresses = b_addresses[n +: TCU_FEOP_BLOCK_N_SIZE];

    assign a_step_valids = a_bitmap_out[m +: TCU_FEOP_BLOCK_M_SIZE];
    assign b_step_valids = b_bitmap_out[n +: TCU_FEOP_BLOCK_N_SIZE];

    // Models FEOP latency for address information
    VX_pipe_register #(
        .DATAW  ($bits(a_step_addresses) + $bits(b_step_addresses) + $bits(a_step_valids) + $bits(b_step_valids)),
        .RESETW ($bits(a_step_addresses) + $bits(b_step_addresses) + $bits(a_step_valids) + $bits(b_step_valids)),
        .DEPTH  (FEOP_LATENCY)
    ) pipe_addr (
        .clk     (clk),
        .reset   (reset),
        .enable  (feop_enable),
        .data_in ({a_step_addresses,         b_step_addresses,         a_step_valids,         b_step_valids}),
        .data_out({a_step_addresses_delayed, b_step_addresses_delayed, a_step_valids_delayed, b_step_valids_delayed})
    );

// ADDRESS EXTRACTION
// *************************************************************************************************************
// FEOPs


    // Operand-window rotation. Needed by both datapaths, so it sits outside the
    // datapath split -- leaving it inside stalled the window advance entirely.
    assign last_set_in_block_a = (32'(a_offset) + 32'(a_set_elems)) >= window_elems;
    assign last_set_in_block_b = (32'(b_offset) + 32'(b_non_zeros)) >= window_elems;

`ifdef VX_CFG_TCU_TYPE_TFR
    // ---- D2: index the double buffers, do not shift them -------------------
    // The step reads TCU_FEOP_BLOCK_M_SIZE A elements and one run of B
    // elements. Element e of a window sits in word e >> lg_i_ratio, in the
    // sub-word field e & (i_ratio-1); the words are already in place, so this is
    // a word select plus a small field align.
    localparam int WIN_WORDS = 2 * `VX_CFG_NUM_LSU_LANES;      // double buffer, in words
    localparam int LG_WIN    = $clog2(WIN_WORDS);
    localparam int B_RUN     = TCU_FEOP_BLOCK_N_SIZE;          // B words a step reads
    localparam int ELEM_IDXW = OFFSET_W + LG_TCU_TC_M_OP + 1;

    wire [WIN_WORDS-1:0][`VX_CFG_XLEN-1:0] a_words;
    wire [WIN_WORDS-1:0][`VX_CFG_XLEN-1:0] b_words;
    assign a_words = {A_buffered[a_active_block[0]], A_buffered[~a_active_block[0]]};
    assign b_words = {B_buffered[b_active_block[0]], B_buffered[~b_active_block[0]]};

    // A: one element per FEOP unit. Under S1 the compacted position's source
    // index comes from the sorter; A is dense in memory there. Under S2 A is
    // compressed, so the compacted position indexes the payload directly.
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][`VX_CFG_XLEN-1:0] a_sel_word;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][1:0]              a_sel_sub;
    for (genvar id = 0; id < TCU_FEOP_BLOCK_M_SIZE; ++id) begin : g_a_pick
        wire [LG_TCU_TC_M_OP-1:0] a_pos = LG_TCU_TC_M_OP'(32'(m) + id);
    `ifndef TCU_DISABLE_S1
        wire [LG_TCU_TC_M_OP-1:0] a_src = (sparsity == 2'd1) ? a_addresses[a_pos] : a_pos;
    `else
        wire [LG_TCU_TC_M_OP-1:0] a_src = a_pos;
    `endif
        wire [ELEM_IDXW-1:0] a_eidx = ELEM_IDXW'(32'(a_offset) + 32'(a_src));
        wire [LG_WIN-1:0]    a_widx = LG_WIN'(32'(a_eidx) >> lg_i_ratio);
        assign a_sel_word[id] = a_words[a_widx];
        assign a_sel_sub[id]  = 2'(32'(a_eidx) & ((32'd1 << lg_i_ratio) - 1));
    end

    // B: a contiguous run of elements starting at b_offset + n. The run may
    // straddle a word, so B_RUN+1 words are selected and then rotated by the
    // sub-word offset.
    wire [ELEM_IDXW-1:0] b_eidx  = ELEM_IDXW'(32'(b_offset) + 32'(n));
    wire [LG_WIN-1:0]    b_word0 = LG_WIN'(32'(b_eidx) >> lg_i_ratio);
    wire [1:0]           b_sub   = 2'(32'(b_eidx) & ((32'd1 << lg_i_ratio) - 1));
    wire [B_RUN:0][`VX_CFG_XLEN-1:0] b_sel_words;
    for (genvar k = 0; k <= B_RUN; ++k) begin : g_b_pick
        assign b_sel_words[k] = b_words[LG_WIN'(32'(b_word0) + k)];
    end

    // ---- operand stage 1: the selected words --------------------------------
    wire                                               opnd_valid_w;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][`VX_CFG_XLEN-1:0] a_word_w;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][1:0]              a_sub_w;
    wire [B_RUN:0][`VX_CFG_XLEN-1:0]                   b_word_w;
    wire [1:0]                                         b_sub_w;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0]                   a_valids_w;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0]                   b_valids_w;

    VX_pipe_register #(
        .DATAW  (1 + TCU_FEOP_BLOCK_M_SIZE * (`VX_CFG_XLEN + 2) + (B_RUN+1) * `VX_CFG_XLEN + 2
                 + TCU_FEOP_BLOCK_M_SIZE + TCU_FEOP_BLOCK_N_SIZE),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_opnd_word (
        .clk      (clk),
        .reset    (reset),
        .enable   (feop_enable),
        .data_in  ({issue_busy,   a_sel_word, a_sel_sub, b_sel_words, b_sub,   a_step_valids, b_step_valids}),
        .data_out ({opnd_valid_w, a_word_w,   a_sub_w,   b_word_w,    b_sub_w, a_valids_w,    b_valids_w})
    );

    // ---- operand stage 2: sub-word align ------------------------------------
    // Element width is 32 >> lg_i_ratio bits, so the align shift is at most
    // (i_ratio-1) fields -- a handful of positions, not 2048.
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][`VX_CFG_XLEN-1:0] a_elem_a;
    for (genvar id = 0; id < TCU_FEOP_BLOCK_M_SIZE; ++id) begin : g_a_align
        assign a_elem_a[id] = a_word_w[id] >> (32'(a_sub_w[id]) << (5 - 32'(lg_i_ratio)));
    end

    wire [(B_RUN+1)*`VX_CFG_XLEN-1:0] b_flat = b_word_w;
    wire [B_RUN-1:0][`VX_CFG_XLEN-1:0] b_row_a =
        (B_RUN * `VX_CFG_XLEN)'(b_flat >> (32'(b_sub_w) << (5 - 32'(lg_i_ratio))));

    wire                                               opnd_valid_s1;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][`VX_CFG_XLEN-1:0] a_elem_s1v;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] b_row_s1v;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0]                   a_valids_s1;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0]                   b_valids_s1;

    VX_pipe_register #(
        .DATAW  (1 + TCU_FEOP_BLOCK_M_SIZE * `VX_CFG_XLEN + TCU_FEOP_BLOCK_N_SIZE * `VX_CFG_XLEN
                 + TCU_FEOP_BLOCK_M_SIZE + TCU_FEOP_BLOCK_N_SIZE),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_opnd_elem_shared (
        .clk      (clk),
        .reset    (reset),
        .enable   (feop_enable),
        .data_in  ({opnd_valid_w,  a_elem_a,   b_row_a,   a_valids_w,  b_valids_w}),
        .data_out ({opnd_valid_s1, a_elem_s1v, b_row_s1v, a_valids_s1, b_valids_s1})
    );
`else
    assign A_window = {A_buffered[a_active_block[0]], A_buffered[~a_active_block[0]]};
    assign a_set_flat = (A_SET_W)'(A_window >> (WIN_SHIFT_W'(a_offset) << ($clog2(`VX_CFG_XLEN) - 32'(lg_i_ratio))));

    assign B_window = {B_buffered[b_active_block[0]], B_buffered[~b_active_block[0]]};
    assign b_set_flat = (B_SET_W)'(B_window >> (WIN_SHIFT_W'(b_offset) << ($clog2(`VX_CFG_XLEN) - 32'(lg_i_ratio))));

    // Operand stage 0: the step's operand windows, with the A-row order for
    // S1 compaction.
    wire                                         opnd_valid_w;
    wire [A_SET_W-1:0]                           a_set_w;
    wire [B_SET_W-1:0]                           b_set_w;
    wire [TCU_TC_M_OP-1:0]                       a_bitmap_out_w;
    wire [TCU_TC_M_OP-1:0][LG_TCU_TC_M_OP-1:0]   a_addresses_w;
    wire [LG_TCU_TC_M_OP-1:0]                    m_w;
    wire [LG_TCU_TC_N_OP-1:0]                    n_w;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0]             a_step_valids_w;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0]             b_step_valids_w;

    VX_pipe_register #(
        .DATAW  (1 + A_SET_W + B_SET_W + TCU_TC_M_OP + TCU_TC_M_OP * LG_TCU_TC_M_OP + LG_TCU_TC_M_OP + LG_TCU_TC_N_OP + TCU_FEOP_BLOCK_M_SIZE + TCU_FEOP_BLOCK_N_SIZE),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_opnd_win (
        .clk      (clk),
        .reset    (reset),
        .enable   (feop_enable),
        .data_in  ({issue_busy,   a_set_flat, b_set_flat, a_bitmap_out,   a_addresses,   m,   n,   a_step_valids,   b_step_valids}),
        .data_out ({opnd_valid_w, a_set_w,    b_set_w,    a_bitmap_out_w, a_addresses_w, m_w, n_w, a_step_valids_w, b_step_valids_w})
    );

`ifdef TCU_DISABLE_S1
    `UNUSED_VAR ({a_bitmap_out_w, a_addresses_w})
    assign a_set_flat_processed = a_set_w;
`else
    // S1: compress the 32 logical A elements (width = 32 / i_ratio) into
    // contiguous positions; a_addresses_w[i] is the source of position i.
    always @(*) begin
        a_set_flat_compressed = '0;
        for (int i = 0; i < TCU_TC_M_OP; ++i) begin
            if (a_bitmap_out_w[i]) begin
                case (i_ratio)
                    4'd1: a_set_flat_compressed[(i << 5) +: 32] = a_set_w[(int'(a_addresses_w[i]) << 5) +: 32];
                    4'd2: a_set_flat_compressed[(i << 4) +: 16] = a_set_w[(int'(a_addresses_w[i]) << 4) +: 16];
                    4'd4: a_set_flat_compressed[(i << 3) +: 8]  = a_set_w[(int'(a_addresses_w[i]) << 3) +: 8];
                    4'd8: a_set_flat_compressed[(i << 2) +: 4]  = a_set_w[(int'(a_addresses_w[i]) << 2) +: 4];
                    default: begin
                        // Unsupported ratio: leave compressed data zeroed.
                    end
                endcase
            end
        end
    end

    assign a_set_flat_processed = (sparsity == 2'd1) ? a_set_flat_compressed : a_set_w;
`endif

    // Operand stage 1: the step's operand sets and select coordinates.
    wire                             opnd_valid_s1;
    wire [A_SET_W-1:0]               a_set_s1;
    wire [B_SET_W-1:0]               b_set_s1;
    wire [LG_TCU_TC_M_OP-1:0]        m_s1;
    wire [LG_TCU_TC_N_OP-1:0]        n_s1;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] a_step_valids_s1;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] b_step_valids_s1;

    VX_pipe_register #(
        .DATAW  (1 + A_SET_W + B_SET_W + LG_TCU_TC_M_OP + LG_TCU_TC_N_OP + TCU_FEOP_BLOCK_M_SIZE + TCU_FEOP_BLOCK_N_SIZE),
        .RESETW (1),
        .DEPTH  (1)
    ) pipe_opnd_set (
        .clk      (clk),
        .reset    (reset),
        .enable   (feop_enable),
        .data_in  ({opnd_valid_w,  a_set_flat_processed, b_set_w,  m_w,  n_w,  a_step_valids_w,  b_step_valids_w}),
        .data_out ({opnd_valid_s1, a_set_s1,             b_set_s1, m_s1, n_s1, a_step_valids_s1, b_step_valids_s1})
    );

`endif

    // Operand stage 2 (per FEOP unit): the selected elements, then the multipliers.
    for (genvar id = 0; id < TCU_FEOP_BLOCK_M_SIZE; id++) begin : g_feop_units

    `ifdef VX_CFG_TCU_TYPE_TFR
        // Already selected and aligned upstream (D2).
        wire [`VX_CFG_XLEN-1:0] a_elem_s1 = a_elem_s1v[id];
        wire [TCU_FEOP_BLOCK_N_SIZE*`VX_CFG_XLEN-1:0] b_row_s1 = b_row_s1v;
        wire [TCU_FEOP_BLOCK_N_SIZE-1:0] feop_bitmap_s1 = a_valids_s1[id] ? b_valids_s1 : '0;
    `else
        wire [`VX_CFG_XLEN-1:0] a_elem_s1 = `VX_CFG_XLEN'(a_set_s1 >> (((32'(m_s1) + 32'(id))) << ($clog2(`VX_CFG_XLEN) - 32'(lg_i_ratio))));
        wire [TCU_FEOP_BLOCK_N_SIZE*`VX_CFG_XLEN-1:0] b_row_s1 = (TCU_FEOP_BLOCK_N_SIZE*`VX_CFG_XLEN)'(b_set_s1 >> ((32'(n_s1) >> lg_i_ratio) << $clog2(`VX_CFG_XLEN)));
        wire [TCU_FEOP_BLOCK_N_SIZE-1:0] feop_bitmap_s1 = a_step_valids_s1[id] ? b_step_valids_s1 : '0;
    `endif

        wire                                               opnd_valid_s2;
        wire [`VX_CFG_XLEN-1:0]                            a_elem;
        wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`VX_CFG_XLEN-1:0] b_row;
        wire [TCU_FEOP_BLOCK_N_SIZE-1:0]                   feop_bitmap;

    `ifdef VX_CFG_TCU_TYPE_TFR
        assign opnd_valid_s2 = opnd_valid_s1;
        assign a_elem        = a_elem_s1;
        assign b_row         = b_row_s1;
        assign feop_bitmap   = feop_bitmap_s1;
    `else
        VX_pipe_register #(
            .DATAW  (1 + `VX_CFG_XLEN + TCU_FEOP_BLOCK_N_SIZE * `VX_CFG_XLEN + TCU_FEOP_BLOCK_N_SIZE),
            .RESETW (1),
            .DEPTH  (1)
        ) pipe_opnd_elem (
            .clk      (clk),
            .reset    (reset),
            .enable   (feop_enable),
            .data_in  ({opnd_valid_s1, a_elem_s1, b_row_s1, feop_bitmap_s1}),
            .data_out ({opnd_valid_s2, a_elem,    b_row,    feop_bitmap})
        );
    `endif

    `ifdef VX_CFG_TCU_TYPE_TFR
        `UNUSED_VAR (opnd_valid_s2)
        // Exact, unrounded products: {sign, exponent, magnitude}.
        VX_tcu_op_mul #(
            .N        (TCU_FEOP_BLOCK_N_SIZE),
            .USE_DSP  (`VX_CFG_TCU_USE_DSP),
            .PROD_REG (PROD_REG)
        ) feop (
            .clk             (clk),
            .enable          (feop_enable),
            .fmt_s           (fmt_s5),
            .lg_i_ratio      (lg_i_ratio),
            .a_elem          (a_elem),
            .b_row           (b_row),
            .valid_in_bitmap (feop_bitmap),
            .prod_exp        (prod_exp[id]),
            .prod_mag        (prod_mag[id]),
            .prod_sign       (prod_sign[id]),
            .prod_exc        (prod_exc[id])
        );
    `else
        VX_tcu_feop #(
            .N            (TCU_FEOP_BLOCK_N_SIZE),
            .FREC_LATENCY (FREC_LATENCY),
            .FMUL_LATENCY (FMUL_LATENCY),
            .FRND_LATENCY (FRND_LATENCY),
            .ID           (id)
        ) feop (
            .clk             (clk),
            .reset           (reset),
            .enable          (feop_enable), // Enables FEOP processing
            .valid_in        (opnd_valid_s2),
            .valid_in_bitmap (feop_bitmap), // Valid bits for the current step
            .fmt_s           (fmt_s),
            .fmt_d           (fmt_d),
            .a_elem          (a_elem),
            .b_row           (b_row),
            .d_block         (d_block[id])
        );
    `endif

    `ifdef DBG_TRACE_TCU
        always @(posedge clk) begin
            if (issue_busy) begin
                `TRACE(1, ("%t: FEOP-enq(%0d): wid=%0d, set=%0d, m=%0d, n=%0d, id=%0d, step=%0d\n", $time, id, execute_if.data.header.wid, set, m, n, id, step));
            end
        end
    `endif // DBG_TRACE_TCU
    end

// FEOPs
// *************************************************************************************************************

    // Models FEOP latency for valid_in (issue_busy) and for the zero-init
    // overwrite of k-set 0 (see zero_init_r), which must reach the accumulator
    // with the write it belongs to.
    wire first_set_overwrite_in = issue_busy && zero_init_r && (set == '0);
    VX_pipe_register #(
        .DATAW  (2),
        .RESETW (2),
        .DEPTH  (FEOP_LATENCY)
    ) pipe_fmul_ctrl (
        .clk     (clk),
        .reset   (reset),
        .enable  (feop_enable),
        .data_in ({issue_busy,       first_set_overwrite_in}),
        .data_out({valid_in_delayed, first_set_overwrite})
    );


    // C-preload writes enter the accumulator one cycle after their response.
    reg  [C_BUF_SLOTS:0][`VX_CFG_NUM_THREADS-1:0][`VX_CFG_XLEN-1:0] c_block_r;
    reg  [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] c_rows_r;
    reg  [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] c_cols_r;
    always @(posedge clk) begin
        if (reset) begin
            c_write_r <= 1'b0;
        end else begin
            c_write_r <= accumulate_c;
        end
        if (accumulate_c) begin
            c_block_r <= C_feop_block;
            c_rows_r  <= c_blk_rows;
            c_cols_r  <= c_blk_cols;
        end
    end

    // The C preload never contends with products, so its write is never refused.
    `RUNTIME_ASSERT(~(c_write_r && ~accu_queues_ready),
        ("%t: *** %s: C-preload write refused by the accumulator", $time, INSTANCE_ID))

`ifndef VX_CFG_TCU_TYPE_TFR
    assign write_data = c_write_r ? c_block_r : d_block;
`endif

    // Accumulator addressing: flush reads and C-preload writes.
    assign read_block_idx = {
        d_line_to_flush_delayed[LG_TCU_FEOP_BLOCK_M_SIZE + LG_TCU_FEOP_N_STEPS +: (LG_TCU_FEOP_STEPS - LG_TCU_FEOP_N_STEPS)],
        d_line_to_flush_delayed[LG_TCU_FEOP_N_STEPS-1:0]
    };
    assign read_row_in_block = LG_TCU_FEOP_BLOCK_M_SIZE'((32'(d_line_to_flush_delayed) >> LG_TCU_FEOP_N_STEPS) & (TCU_FEOP_BLOCK_M_SIZE-1));

    for (genvar i = 0; i < TCU_FEOP_BLOCK_M_SIZE; i++) begin : g_row_addresses
        // While flushing, read only the row that feeds the current d_line.
        assign read_row_valid[i] = (read_row_in_block == LG_TCU_FEOP_BLOCK_M_SIZE'(i));
        assign c_blk_rows[i] = LG_TCU_TC_M_OP'(((32'(c_blk_idx[LG_TCU_FEOP_STEPS-1:0]) >> LG_TCU_FEOP_N_STEPS) << LG_TCU_FEOP_BLOCK_M_SIZE) + i);
    end
    for (genvar i = 0; i < TCU_FEOP_BLOCK_N_SIZE; i++) begin : g_col_addresses
        assign c_blk_cols[i] = LG_TCU_TC_N_OP'(((32'(c_blk_idx[LG_TCU_FEOP_STEPS-1:0]) & (TCU_FEOP_N_STEPS-1)) << LG_TCU_FEOP_BLOCK_N_SIZE) + i);
    end

    assign accu_read_en = ready_to_flush_delayed;
`ifdef VX_CFG_TCU_TYPE_TFR
    // The accumulator's flush stages freeze with the core's flush control.
    assign flush_enable = feop_enable && ~mem_stall;
`endif
    // Registered busy: nothing of a new op reaches the accumulator in its accept cycle.
    assign accu_write_valid = busy_r && (valid_in_delayed || c_write_r);
`ifdef VX_CFG_TCU_TYPE_TFR
    // Every accumulator write is already qualified by a valid bit, so no global
    // enable is needed; dropping it removes ~8k more clock-enable loads.
    assign accu_enable = 1'b1;
`else
    assign accu_enable = busy_r;
`endif

`ifdef VX_CFG_TCU_TYPE_TFR
    // The C preload enters the same fixed-point domain as the products.
    localparam int ACCU_BANKS = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE;
    wire [ACCU_BANKS-1:0][TCU_OP_EXP_W-1:0] accu_write_exp;
    wire [ACCU_BANKS-1:0][TCU_OP_MAG_W-1:0] accu_write_mag;
    wire [ACCU_BANKS-1:0]                   accu_write_sign;
    fedp_excep_t [ACCU_BANKS-1:0]           accu_write_exc;

    // A C beat carries a zero payload with overwrite asserted, which clears the
    // product banks for the tile; the fp32 C word goes to the C store and is
    // fused back in at flush (its exponent range does not fit the banks' window).
    wire [ACCU_BANKS-1:0][`VX_CFG_XLEN-1:0] accu_c_data;

    for (genvar i = 0; i < ACCU_BANKS; ++i) begin : g_accu_write
        localparam int ROW = i / TCU_FEOP_BLOCK_N_SIZE;
        localparam int COL = i % TCU_FEOP_BLOCK_N_SIZE;

        assign accu_write_exp[i]  = c_write_r ? '0    : prod_exp[ROW][COL];
        assign accu_write_mag[i]  = c_write_r ? '0    : prod_mag[ROW][COL];
        assign accu_write_sign[i] = c_write_r ? 1'b0  : prod_sign[ROW][COL];
        assign accu_write_exc[i].is_nan = c_write_r ? 1'b0 : prod_exc[ROW][COL].is_nan;
        assign accu_write_exc[i].is_inf = c_write_r ? 1'b0 : prod_exc[ROW][COL].is_inf;
        assign accu_write_exc[i].sign   = c_write_r ? 1'b0 : prod_exc[ROW][COL].sign;

        assign accu_c_data[i] = c_block_r[0][i];
    end

    VX_tcu_op_accu #(
        .BLOCK_M          (TCU_FEOP_BLOCK_M_SIZE),
        .BLOCK_N          (TCU_FEOP_BLOCK_N_SIZE),
        .XBAR_LATENCY     (XBAR_LATENCY),
        .XBAR_QUEUE_DEPTH (TCU_FEOP_XBAR_QUEUE_DEPTH),
        .CREDIT_LIMIT     (CREDIT_LIMIT)
    ) feop_accu (
        .clk                  (clk),
        .reset                (reset),
        .enable               (accu_enable),
        .fmt_s                (fmt_s5),
        .flush_enable         (flush_enable),
        .read_en              (accu_read_en),
        .read_row_valid       (read_row_valid),
        .read_block_idx       (read_block_idx),
        .read_row_data        (read_row_data),
        .write_valid          (accu_write_valid),
        .write_ready          (accu_queues_ready),
        .write_addr_row       (c_write_r ? c_rows_r : a_step_addresses_delayed),
        .write_addr_row_valid (c_write_r ? '1 : a_step_valids_delayed),
        .write_addr_col       (c_write_r ? c_cols_r : b_step_addresses_delayed),
        .write_addr_col_valid (c_write_r ? '1 : b_step_valids_delayed),
        .write_exp            (accu_write_exp),
        .write_mag            (accu_write_mag),
        .write_sign           (accu_write_sign),
        .write_exc            (accu_write_exc),
        .overwrite            (c_write_r || first_set_overwrite),
        .write_is_c           (c_write_r),
        .write_c_data         (accu_c_data),
        .credit_ok            (credit_ok),
        .accu_ready_to_flush  (accu_ready_to_flush)
    );
`else
    VX_tcu_feop_accu #(
        .BLOCK_M          (TCU_FEOP_BLOCK_M_SIZE),
        .BLOCK_N          (TCU_FEOP_BLOCK_N_SIZE),
        .FREC_LATENCY     (FACC_REC_LATENCY),
        .FADD_LATENCY     (FACC_ADD_LATENCY),
        .FRND_LATENCY     (FRND_LATENCY),
        .FACC_LATENCY     (FACC_LATENCY),
        .XBAR_LATENCY     (XBAR_LATENCY),
        .XBAR_QUEUE_DEPTH (TCU_FEOP_XBAR_QUEUE_DEPTH)
    ) feop_accu (
        .clk                  (clk),
        .reset                (reset),
        .enable               (accu_enable),
        .fmt_d                (fmt_d),
        .read_en              (accu_read_en),
        .read_row_valid       (read_row_valid),
        .read_block_idx       (read_block_idx),
        .read_data            (read_data),
        // All write inputs from FEOPs are delayed FMUL cycles
        .write_valid          (accu_write_valid),
        .write_ready          (accu_queues_ready),  // Not ready when even one of them is full
        .write_addr_row       (c_write_r ? c_rows_r : a_step_addresses_delayed),
        .write_addr_row_valid (c_write_r ? '1 : a_step_valids_delayed),
        .write_addr_col       (c_write_r ? c_cols_r : b_step_addresses_delayed),
        .write_addr_col_valid (c_write_r ? '1 : b_step_valids_delayed),
        .write_data           (write_data),
        // C preload blocks, or the first k-set of a zero-init op (never both:
        // a zero-init op issues no C requests).
        .overwrite            (c_write_r || first_set_overwrite),
        .accu_ready_to_flush  (accu_ready_to_flush)
    );
`endif

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

// WORK ASSIGNMENT TO FEOPS
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// FLUSHING AND RESULT HANDLING

    assign d_lines_ready = (issuing_done && (c_blk_idx == TCU_FEOP_STEPS)) ? (LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE + 1)'((TCU_TC_M_OP * TCU_FEOP_N_STEPS)) : '0;

    // Steps issued whose products have not yet reached the accumulator: the
    // accumulator cannot see products still in the FEOP pipeline, and
    // accumulation stalls can hold them there for any length of time.
    localparam FEOP_INFLIGHT_W = $clog2(FEOP_LATENCY + 1);
    reg [FEOP_INFLIGHT_W-1:0] feop_inflight;
    always @(posedge clk) begin
        if (reset) begin
            feop_inflight <= '0;
        end else if (feop_enable) begin
            feop_inflight <= feop_inflight + FEOP_INFLIGHT_W'(issue_busy) - FEOP_INFLIGHT_W'(valid_in_delayed);
        end
    end

    // Flush once every product has been accumulated, and only while the port
    // can take the write each accumulator read produces.
    assign ready_to_flush_raw = (d_lines_ready > d_line_to_flush) && (feop_inflight == '0) && accu_ready_to_flush && lsu_req_if.req_ready;
    assign ready_to_flush = flush_flag && ready_to_flush_raw;

    // Completion. A flush op completes on its last D write. A no-flush op
    // completes once the drained state has held for the full FEOP->xbar->accu
    // latency: issuing_done only means the last product entered the pipe.
`ifdef VX_CFG_TCU_TYPE_TFR
    // accu_ready_to_flush covers the queues, the crossbar outputs and the align
    // stage; this settle covers the FEOP pipe and the crossbar's internal skid,
    // whose occupancy the stream crossbar does not expose.
    localparam NOFLUSH_SETTLE = FEOP_LATENCY + ACCU_WRITE_DEPTH + 2;
`else
    localparam NOFLUSH_SETTLE = FEOP_LATENCY + XBAR_LATENCY + ACCU_READ_LATENCY + FACC_LATENCY + 1;
`endif
    localparam NOFLUSH_CTR_W  = $clog2(NOFLUSH_SETTLE + 1);
    wire no_flush_base = busy && ~flush_flag && ready_to_flush_raw && (d_line_to_flush == '0) && ~result_pending_r;
    reg [NOFLUSH_CTR_W-1:0] noflush_settle_ctr;
    always @(posedge clk) begin
        if (reset || ~no_flush_base) begin
            noflush_settle_ctr <= '0;
        end else if (noflush_settle_ctr != NOFLUSH_CTR_W'(NOFLUSH_SETTLE)) begin
            noflush_settle_ctr <= noflush_settle_ctr + NOFLUSH_CTR_W'(1);
        end
    end
    assign no_flush_complete = no_flush_base && (noflush_settle_ctr == NOFLUSH_CTR_W'(NOFLUSH_SETTLE));
`ifdef VX_CFG_TCU_TYPE_TFR
    assign result_pulse = no_flush_complete || (busy && flush_flag && (32'(d_line_to_flush_wr) == TCU_TC_M_OP * TCU_FEOP_N_STEPS - 1) && wr_req_fire);
`else
    assign result_pulse = no_flush_complete || (busy && flush_flag && (32'(d_line_to_flush_delayed) == TCU_TC_M_OP * TCU_FEOP_N_STEPS - 1) && wr_req_fire);
`endif

// ----------------------------------- tx_bar HANDLING ----------------------------------------------

    assign txbar_bar_id = txbar_bar_id_imm;
    `UNUSED_VAR (txbar_bar_id)

    if (`VX_CFG_NUM_WARPS > 1) begin : g_txbar_addr_w
        assign txbar_addr = {txbar_bar_id[NW_BITS-1:0], txbar_bar_id[BAR_ID_SHIFT +: NB_BITS]};
    end else begin : g_txbar_addr_wo
        assign txbar_addr = BAR_ADDR_W'(txbar_bar_id[BAR_ID_SHIFT +: NB_BITS]);
    end

    // Completion releases the op's transaction barrier and emits its result
    // in the same cycle; the result is then registered (result_buf).
    wire result_buf_ready;
    assign result_pending   = result_pending_r;
    assign result_txbar_req = result_pending && ~op_ctx_empty && result_buf_ready;
    assign result_fire      = result_txbar_req && txbar_bus_if.ready;

    assign txbar_bus_if.valid = execute_txbar_req || result_txbar_req;
    assign txbar_bus_if.data.addr = execute_txbar_req ? txbar_addr : op_ctx_bar_addr;

    /* is_done polarity is reversed: 
       1: Acquire the lock
       0: Release the lock            */
    assign txbar_bus_if.data.is_done = ~execute_txbar_req;

    VX_fifo_queue #(
        .DATAW (BAR_ADDR_W),
        .DEPTH (1)
    ) txbar_opctx_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (execute_fire),
        .pop      (result_fire),
        .data_in  (txbar_addr),
        .data_out (op_ctx_bar_addr),
        .empty    (op_ctx_empty),
        `UNUSED_PIN(alm_empty),
        .full     (op_ctx_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    always @(posedge clk) begin
        if (result_pulse && ~reset && ~execute_fire) begin
            `TRACE (2, ("%t: [tcu_op_core]: result_pulse=%b no_flush_complete=%b flush_flag=%b busy=%b busy_r=%b result_pending_r=%b d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d ready_to_flush_raw=%b ready_to_flush=%b accu_ready_to_flush=%b wr_req_fire=%b\n",
                $time, result_pulse, no_flush_complete, flush_flag, busy, busy_r, result_pending_r, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, ready_to_flush_raw, ready_to_flush, accu_ready_to_flush, wr_req_fire))
        end
        if (execute_if.valid && ~execute_if.ready) begin
            `TRACE(1, ("%t: [tcu_op_core]: execute stall op_type=0x%0h busy=%b busy_r=%b mqueue_full=%b op_ctx_full=%b txbar_ready=%b result_pending=%b result_pending_r=%b result_pulse=%b result_fire=%b\n",
                $time, execute_if.data.op_type, busy, busy_r, mqueue_full, op_ctx_full, txbar_bus_if.ready, result_pending, result_pending_r, result_pulse, result_fire));
            `TRACE(1, ("%t: [tcu_op_core]: stalled payload wid=%0d pc=0x%0h uuid=%0d rs1={0x%0h 0x%0h 0x%0h 0x%0h 0x%0h 0x%0h} rs2={%0d %0d %0d %0d %0d %0d 0x%0h 0x%0h}\n",
                $time,
                execute_if.data.header.wid,
                execute_if.data.header.PC,
                execute_if.data.header.uuid,
                execute_if.data.rs1_data[0],
                execute_if.data.rs1_data[1],
                execute_if.data.rs1_data[2],
                execute_if.data.rs1_data[3],
                execute_if.data.rs1_data[4],
                execute_if.data.rs1_data[5],
                execute_if.data.rs2_data[0],
                execute_if.data.rs2_data[1],
                execute_if.data.rs2_data[2],
                execute_if.data.rs2_data[3],
                execute_if.data.rs2_data[4],
                execute_if.data.rs2_data[5],
                execute_if.data.rs2_data[6],
                execute_if.data.rs2_data[7]));
        end
        if (ready_to_flush_raw || ready_to_flush || wr_req_fire || valid_out) begin
            `TRACE(2, ("%t: [tcu_op_core]: flush-state flush_flag=%b ready_to_flush_raw=%b ready_to_flush=%b valid_out=%b mem_stall=%b wr_req_fire=%b d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d accu_ready_to_flush=%b tcu_req_ready=%b\n",
                $time, flush_flag, ready_to_flush_raw, ready_to_flush, valid_out, mem_stall, wr_req_fire, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, accu_ready_to_flush, lsu_req_if.req_ready));
        end
        if (no_flush_complete) begin
            `TRACE(2, ("%t: [tcu_op_core]: no-flush completion path no_flush_complete=%b flush_flag=%b ready_to_flush_raw=%b d_line_to_flush=%0d result_pending_r=%b execute_valid=%b execute_ready=%b execute_fire=%b\n",
                $time, no_flush_complete, flush_flag, ready_to_flush_raw, d_line_to_flush, result_pending_r, execute_if.valid, execute_if.ready, execute_fire));
        end
    end

    always @(posedge clk) begin
        if (~reset) begin
            if (execute_fire) begin
                `TRACE(2, ("%t: [tcu_op_core-txbar] START fire wid=%0d bar_id=0x%0h addr=%0d\n",
                    $time, execute_if.data.header.wid, txbar_bar_id, txbar_addr))
                `TRACE(2, ("%t: [tcu_op_core]: accepted payload uuid=%0d pc=0x%0h rs1={0x%0h 0x%0h 0x%0h 0x%0h 0x%0h 0x%0h} rs2={%0d %0d %0d %0d %0d %0d 0x%0h 0x%0h}\n",
                    $time,
                    execute_if.data.header.uuid,
                    execute_if.data.header.PC,
                    execute_if.data.rs1_data[0],
                    execute_if.data.rs1_data[1],
                    execute_if.data.rs1_data[2],
                    execute_if.data.rs1_data[3],
                    execute_if.data.rs1_data[4],
                    execute_if.data.rs1_data[5],
                    execute_if.data.rs2_data[0],
                    execute_if.data.rs2_data[1],
                    execute_if.data.rs2_data[2],
                    execute_if.data.rs2_data[3],
                    execute_if.data.rs2_data[4],
                    execute_if.data.rs2_data[5],
                    execute_if.data.rs2_data[6],
                    execute_if.data.rs2_data[7]))
            end
            if (result_pulse) begin
                `TRACE(2, ("%t: [tcu_op_core-txbar] RESULT pulse pending=%0b opctx_empty=%0b txbar_ready=%0b\n",
                    $time, result_pending_r, op_ctx_empty, txbar_bus_if.ready))
            end
            if (result_fire) begin
                `TRACE(2, ("%t: [tcu_op_core-txbar] DONE fire wid=%0d addr=%0d\n",
                    $time, op_wid, op_ctx_bar_addr))
            end
            if (txbar_bus_if.valid && txbar_bus_if.ready) begin
                `TRACE(2, ("%t: [tcu_op_core-txbar] TXBAR xfer addr=%0d is_done=%0b\n",
                    $time, txbar_bus_if.data.addr, txbar_bus_if.data.is_done))
            end
        end
    end
// ----------------------------------- tx_bar HANDLING ----------------------------------------------


    // Stores u-ops until commit side accepts completion.
    VX_fifo_queue #(
        .DATAW (MDATA_WIDTH),
        .DEPTH (MDATA_QUEUE_DEPTH)
    ) mdata_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (execute_fire),
        .pop      (result_fire),
        .data_in  (mdata_queue_din),
        .data_out (mdata_queue_dout),
        `UNUSED_PIN(empty),
        `UNUSED_PIN(alm_empty),
        .full(mqueue_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    assign mdata_queue_din = {
        execute_if.data.header.uuid,
        execute_if.data.header.wid,
        execute_if.data.header.PC,
        execute_if.data.header.rd};

    // The in-flight op's header.
    assign {op_uuid, op_wid, op_PC, op_rd} = mdata_queue_dout;
    `UNUSED_VAR ({op_wid, op_PC, op_rd})

    wire [MDATA_WIDTH-1:0] result_mdata;

    VX_elastic_buffer #(
        .DATAW (MDATA_WIDTH),
        .SIZE  (2)
    ) result_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (result_fire),
        .ready_in  (result_buf_ready),
        .data_in   (mdata_queue_dout),
        .data_out  (result_mdata),
        .ready_out (result_if.ready),
        .valid_out (result_if.valid)
    );

    assign {result_if.data.header.uuid,
            result_if.data.header.wid,
            result_if.data.header.PC,
            result_if.data.header.rd} = result_mdata;

    assign result_if.data.header.wb       = 1'b0;
    assign result_if.data.header.wr_xregs = '0;
    assign result_if.data.header.tmask    = {`VX_CFG_NUM_THREADS{1'b1}};
    assign result_if.data.data            = '0;
    assign result_if.data.header.pid      = '0;
    assign result_if.data.header.sop      = 1'b1;
    assign result_if.data.header.eop      = 1'b1;

`ifdef VX_CFG_TCU_TYPE_TFR
    assign d_line = read_row_data;
`else
    assign d_line = read_data[(((32'(d_line_to_flush_delayed) >> LG_TCU_FEOP_N_STEPS) & (TCU_FEOP_BLOCK_M_SIZE-1)) << LG_TCU_FEOP_BLOCK_N_SIZE) +: TCU_FEOP_BLOCK_N_SIZE];
`endif


    /* Flush the D line to MEM - Pad with zeros to fill the 32 spots */
    assign lsu_req_if.req_data.data = rd_req_valid ? '0 : {{PAD_LANES{`VX_CFG_XLEN'(0)}}, d_line};

    assign wr_req_fire = (valid_out && feop_enable) && lsu_req_if.req_ready && (lsu_req_if.req_data.rw == 1'b1);

`ifdef VX_CFG_TCU_TYPE_TFR
    // Read side: one stage, then the accumulator's asynchronous bank read.
    VX_pipe_register #(
        .DATAW  (1 + $bits(d_line_to_flush_delayed)),
        .RESETW (1 + $bits(d_line_to_flush_delayed)),
        .DEPTH  (1)
    ) pipe_flush_rd (
        .clk     (clk),
        .reset   (reset),
        .enable  (flush_enable),
        .data_in ({ready_to_flush,         d_line_to_flush[LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE - 1:0]}),
        .data_out({ready_to_flush_delayed, d_line_to_flush_delayed})
    );

    // Write side: the rounded line appears FLUSH_LATENCY cycles after the read,
    // so the D-line request and its index trail by the same amount.
    VX_pipe_register #(
        .DATAW  (1 + $bits(d_line_to_flush_wr)),
        .RESETW (1 + $bits(d_line_to_flush_wr)),
        .DEPTH  (FLUSH_LATENCY)
    ) pipe_flush_wr (
        .clk     (clk),
        .reset   (reset),
        .enable  (flush_enable),
        .data_in ({ready_to_flush_delayed, d_line_to_flush_delayed}),
        .data_out({valid_out_w,            d_line_to_flush_wr})
    );

    assign valid_out = valid_out_w;
`else
    // Delay WB control signals to match FEOP latency
    VX_pipe_register #(
        .DATAW  (1 + $bits(d_line_to_flush_delayed)),
        .RESETW (1 + $bits(d_line_to_flush_delayed)),
        .DEPTH  (FEOP_LATENCY + XBAR_LATENCY + ACCU_READ_LATENCY + FACC_LATENCY)
    ) pipe_flush_dummy (
        .clk     (clk),
        .reset   (reset),
        .enable  (feop_enable && ~mem_stall),
        .data_in ({ready_to_flush,         d_line_to_flush[LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE - 1:0]}),
        .data_out({ready_to_flush_delayed, d_line_to_flush_delayed})
    );

    // Use the same stage as read_data (ready_to_flush_delayed) to avoid misalignment
    assign valid_out = ready_to_flush_delayed;
`endif

// FLUSHING AND RESULT HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY INTERFACE

    // Reads (operand fetch) and D-line writes share the request port; the two
    // never overlap (see the assertion below).
    assign lsu_req_if.req_valid = rd_req_valid || (valid_out && feop_enable); // feop_enable: never issue the same valid_out twice
    assign lsu_req_if.req_data.rw = ~rd_req_valid;

    // Read tags carry the matrix as a 2-bit index plus a per-matrix sequence
    // bit. tag.value is the LSU adapter's response reassembly key, so
    // concurrent requests never share it; the index (not the one-hot id)
    // leaves the tag's upper bits free for the engine index that
    // VX_tcu_unit's arbiter inserts. Writes get no response and carry 0.
    reg [MATRIX_ID_BITS-1:0] req_seq_r;
    always @(posedge clk) begin
        if (reset) begin
            req_seq_r <= '0;
        end else if (rd_req_fire) begin
            req_seq_r <= req_seq_r ^ grant_onehot;
        end
    end
    wire req_seq_bit = |(req_seq_r & grant_onehot);
    wire [MATRIX_IDX_BITS-1:0] grant_index;
    VX_onehot_encoder #(
        .N (MATRIX_ID_BITS)
    ) grant_enc (
        .data_in    (grant_onehot),
        .data_out   (grant_index),
        `UNUSED_PIN (valid_out)
    );
    `STATIC_ASSERT ((TAG_WIDTH - UUID_WIDTH) >= (MATRIX_IDX_BITS + 1),
        ("tag id field too narrow to encode TCU matrix index + sequence bit"))
    assign lsu_req_if.req_data.tag.uuid  = op_uuid;
    assign lsu_req_if.req_data.tag.value =
        (TAG_WIDTH - UUID_WIDTH)'(rd_req_valid ? {req_seq_bit, grant_index} : '0);

    assign wr_mask = {{(`VX_CFG_NUM_LSU_LANES - TCU_FEOP_BLOCK_N_SIZE){1'b0}}, {TCU_FEOP_BLOCK_N_SIZE{1'b1}}};
    // For bitmap reads with K<16, request identical 16-bit lane masks for A-half and B-half:
    // [31:16] = 0...01...1 (K ones in LSBs), [15:0] = same.
    assign bitmap_half_mask = (16'hFFFF     >> (5'd16 - K[4:0]));
    assign bitmap_small_k_mask = {bitmap_half_mask, bitmap_half_mask};

`ifndef TCU_DISABLE_S1
    assign bitmap_s1_mask = (32'hFFFFFFFF >> (6'd32 - K[5:0]));
`endif
    assign lsu_req_if.req_data.mask   = ~rd_req_valid ? wr_mask :
                                        (grant_onehot == MATRIX_ID_BITS'(1) && K < 16 && sparsity == 2'd2) ? bitmap_small_k_mask :
                                    `ifndef TCU_DISABLE_S1
                                        (grant_onehot == MATRIX_ID_BITS'(1) && K < 32 && sparsity == 2'd1) ? bitmap_s1_mask :
                                    `endif
                                        {`VX_CFG_NUM_LSU_LANES{1'b1}};
    assign lsu_req_if.req_data.byteen = {`VX_CFG_NUM_LSU_LANES{{LSU_WORD_SIZE{1'b1}}}};

    for (genvar l = 0; l < `VX_CFG_NUM_LSU_LANES; l++) begin : g_mem_addr
        wire [LSU_ADDR_WIDTH-1:0] word_addr;
        wire [MEM_ADDRW-1:0] block_addr;
        wire is_lmem;

        wire [`VX_CFG_XLEN-1:0] lane_byte_addr = rd_req_valid ?
                                            ((grant_onehot[0] && sparsity == 2'd2) ?
                                                (l < `VX_CFG_NUM_LSU_LANES/2 ?
                                                    a_bitmap_addr + (`VX_CFG_XLEN'(l) << $clog2(LSU_WORD_SIZE)) :
                                                    b_bitmap_addr + ((`VX_CFG_XLEN'(l) - (`VX_CFG_NUM_LSU_LANES >> 1)) << $clog2(LSU_WORD_SIZE))) :
                                                req_rd_addr + (`VX_CFG_XLEN'(l) << $clog2(LSU_WORD_SIZE))) :
                                            d_tile_addr + (`VX_CFG_XLEN'(l) << $clog2(LSU_WORD_SIZE));

        `UNUSED_VAR (lane_byte_addr)
        assign word_addr = lane_byte_addr[LSU_ADDR_WIDTH + REQ_ASHIFT - 1 : REQ_ASHIFT]; // LSU word address per lane
        assign block_addr = lane_byte_addr[`VX_CFG_MEM_ADDR_WIDTH-1:MEM_ASHIFT];                     // MEM block address for LMEM flagging
        assign is_lmem = (block_addr >= LMEM_ADDR_START) && (LMEM_AT_ADDR_TOP || (block_addr < LMEM_ADDR_END));

        // Per-lane sideband: only the LMEM route bit is set (flush/io stay 0).
        mem_bus_attr_t lane_attr_w;
        always_comb begin
            lane_attr_w = '0;
            lane_attr_w.is_addr_local = is_lmem;
        end
        assign lsu_req_if.req_data.user[l] = lane_attr_w;

        assign lsu_req_if.req_data.addr[l] = word_addr;

        // Read operands must be LMEM-resident (inactive lanes are don't-care).
        `RUNTIME_ASSERT(~(rd_req_valid && lsu_req_if.req_data.mask[l] && ~is_lmem),
            ("%t: *** %s: TCU_OP read operand is not LMEM-resident: byte_addr=0x%0h word_addr=0x%0h block_addr=0x%0h lane=%0d",
             $time, INSTANCE_ID, lane_byte_addr, word_addr, block_addr, l))
    end

    // The engine registers its own request port. A skid buffer's ready means
    // "has space", independent of valid, which ready_to_flush_raw relies on.
    localparam REQ_DATAW = 1 + `VX_CFG_NUM_LSU_LANES * (1 + (`VX_CFG_MEM_ADDR_WIDTH - `CLOG2(LSU_WORD_SIZE))
                         + (8 * LSU_WORD_SIZE) + LSU_WORD_SIZE + `UP(MEM_ATTR_WIDTH)) + TAG_WIDTH;
    `STATIC_ASSERT ($bits(lsu_req_if.req_data) == REQ_DATAW, ("tcu_op_core: REQ_DATAW mismatch"))

    wire [REQ_DATAW-1:0] req_buf_din = lsu_req_if.req_data;
    wire [REQ_DATAW-1:0] req_buf_dout;

    VX_elastic_buffer #(
        .DATAW (REQ_DATAW),
        .SIZE  (2)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (lsu_req_if.req_valid),
        .ready_in  (lsu_req_if.req_ready),
        .data_in   (req_buf_din),
        .data_out  (req_buf_dout),
        .ready_out (tcu_lsu_mem_if.req_ready),
        .valid_out (tcu_lsu_mem_if.req_valid)
    );
    assign tcu_lsu_mem_if.req_data = req_buf_dout;

    // lsu_req_if carries requests only; responses arrive on tcu_lsu_mem_if.
    assign lsu_req_if.rsp_valid = 1'b0;
    assign lsu_req_if.rsp_data  = '0;
    `UNUSED_VAR (lsu_req_if.rsp_ready)

    // Reads and D-line writes never compete for the port.
    `RUNTIME_ASSERT(~(rd_req_valid && valid_out && feop_enable),
        ("%t: *** %s: operand read and D-line write requested in the same cycle", $time, INSTANCE_ID))
    // Only dense, S1 and S2 modes exist.
    `RUNTIME_ASSERT(~(execute_fire && (sparsity_imm == 2'd3)),
        ("%t: *** %s: invalid sparsity mode %0d", $time, INSTANCE_ID, sparsity_imm))

// MEMORY INTERFACE
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DEBUGGING TRACES

    always @ (posedge clk) begin
        if (~reset && ~result_fire && execute_fire) begin
            `TRACE(2, ("init_flag=%b, flush_flag=%b\n", init_flag, flush_flag));
        end
    end

    always_ff @(posedge clk) begin
        if (~reset) begin
            if (execute_fire) begin
                `TRACE(1, ("%t: [tcu_op_core] TCU execution fired, \nA_addr=0x%x, B_addr=0x%x, C_addr=0x%x, D_addr=0x%x, \nA_compressed_blocks=%0d, B_compressed_blocks=%0d, K=%0d, fmt_s=%0d, fmt_d=%0d, sparsity=%0d, A_bitmap_addr=0x%x, B_bitmap_addr=0x%x, barrier_ID=0x%x, flags=%b\n", 
                            $time, a_tile_addr_imm, b_tile_addr_imm, c_tile_addr_imm, d_tile_addr_imm,
                            a_blocks_imm, b_blocks_imm, K_imm, fmt_s_imm, fmt_d_imm, sparsity_imm,
                            a_bitmap_addr_imm, b_bitmap_addr_imm, txbar_bar_id_imm, {init_flag_imm, flush_flag_imm}));   
            
            end
            if (execute_if.valid && ~execute_if.ready) begin
                `TRACE(2, ("%t: [tcu_op_core] pending op snapshot uuid=%0d pc=0x%0h busy=%b busy_r=%b issue_busy=%b issuing_done=%b op_ctx_full=%b mqueue_full=%b txbar_ready=%b ready_to_flush=%b d_lines_ready=%0d\n",
                            $time,
                            execute_if.data.header.uuid,
                            execute_if.data.header.PC,
                            busy,
                            busy_r,
                            issue_busy,
                            issuing_done,
                            op_ctx_full,
                            mqueue_full,
                            txbar_bus_if.ready,
                            ready_to_flush,
                            d_lines_ready));
            end
            if (issue_busy) begin
                `TRACE(1, ("%t: [tcu_op_core] Issue Busy processing, i_ratio=%0d, lg_i_ratio=%0d, ready_to_flush=%0d\n", 
                            $time, i_ratio, lg_i_ratio, ready_to_flush));
                `TRACE(2, ("%t: c_blk_idx=%0d, step=%0d / %0d, set=%0d, m=%0d / %0d, n=%0d / %0d\n", 
                            $time, c_blk_idx, step, TCU_FEOP_STEPS, set, m, TCU_TC_M_OP, n, TCU_TC_N_OP));
            end
            if (feop_enable && last_step_in_execution) begin
                `TRACE(2, ("%t: [tcu_op_core] All iterations issued\n", $time));
            end
            if (result_fire) begin
                `TRACE(1, ("%t: [tcu_op_core] Result fired downstream\n", $time));
                `TRACE(1, ("%t: [tcu_op_core] Full-queue stall cycles=%0d\n", $time, full_queue_stall_cycles));
                if (full_queue_stall_cycles > 0 && sparsity == 2'b00) begin
                    `TRACE(1, ("%t: [tcu_op_core] WARNING: full-queue stalls on a dense workload\n", $time));
                end
            end
            if (execute_if.valid && mqueue_full) begin
                `TRACE(1, ("%t: [tcu_op_core]: issue back-pressure, mqueue_full=%b\n", $time, mqueue_full));
            end
            if (busy && ~accu_queues_ready) begin
                `TRACE(1, ("%t: [tcu_op_core] FULL QUEUES: accu_queues_ready=%b\n", $time, accu_queues_ready));
            end
            if (rd_req_fire) begin
                // Print requested info
                `TRACE(2, ("%t: LMEM: Issuing read request, req_addr[0]=0x%x mask=%b, tag=%b matrix=%0s, c_blocks_requested=%0d\n", 
                            $time, lsu_req_if.req_data.addr[0] * LSU_WORD_SIZE, lsu_req_if.req_data.mask, lsu_req_if.req_data.tag,
                            (grant_onehot == 4'b0001) ? "Bitmap" : (grant_onehot == 4'b0010) ? "A" : (grant_onehot == 4'b0100) ? "B" : "C",
                            c_blocks_requested));
                `TRACE(2, ("a_req_blocks_remaining=%0d, b_req_blocks_remaining=%0d, total_c_blocks=%0d, bitmap_req_blocks_remaining=%0d\n",
                            a_req_blocks_remaining, b_req_blocks_remaining, TCU_C_BLOCKS_IN_ACCU, bitmap_req_blocks_remaining));
            end
            if (rd_rsp_fire) begin
                `TRACE(2, ("%t: LMEM: read rsp: tag=%x mask=%b,   matrix=%0s accumulate_c=%b\nc_blocks_loaded=%0d, c_blocks_accumulated=%0d, C_buf_idx=%0d\n", $time,
                            tcu_lsu_mem_if.rsp_data.tag, tcu_lsu_mem_if.rsp_data.mask,
                            (rsp_matrix_id == 4'b0001) ? "Bitmap" : (rsp_matrix_id == 4'b0010) ? "A" : (rsp_matrix_id == 4'b0100) ? "B" : "C",
                            accumulate_c, c_blocks_loaded, c_blocks_accumulated, c_blocks_loaded % (C_BUF_SLOTS+1)));
                
                for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; l++) begin
                    if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                        if (rsp_matrix_id == 4'b0001) begin // Bitmap printing
                            `TRACE(2, ("    lane[%0d]: data=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                        end else begin
                            `TRACE(2, ("    lane[%0d]: data=%x\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                        end
                    end
                end
                if (rsp_matrix_id == 4'b0001) begin
                    `TRACE(2, ("\n"));
                    if (sparsity == 2'd2) begin
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; l++) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                if (l < `VX_CFG_NUM_LSU_LANES / 2) begin
                                    `TRACE(2, ("    A_bitmap[%0d]=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                                end
                                else if (l >= `VX_CFG_NUM_LSU_LANES / 2) begin
                                    `TRACE(2, ("    B_bitmap[%0d]=%b\n", l - `VX_CFG_NUM_LSU_LANES / 2, tcu_lsu_mem_if.rsp_data.data[l]));
                                end
                            end
                        end
                    end
                `ifndef TCU_DISABLE_S1
                    else if (sparsity == 2'd1) begin
                        for (integer l = 0; l < `VX_CFG_NUM_LSU_LANES; l++) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                `TRACE(2, ("    B_bitmap[%0d]=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                            end
                        end
                    end
                `endif
                end
            end
            if (issue_busy && (32'(d_line_to_flush) == TCU_TC_M_OP * TCU_FEOP_BLOCK_N_SIZE)) begin
                `TRACE(2, ("%t:[tcu_op_core]: d_line_to_flush=%0d\n", $time, d_line_to_flush));
            end
            if (wr_req_fire) begin
                `TRACE(2, ("%t: [tcu_op_core]: Flushing, d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d\n read_block=%0d\n", $time, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, read_block_idx));
            `ifndef VX_CFG_TCU_TYPE_TFR
                `TRACE(2, ("read_data=\n"));
                `TRACE_ARRAY1D(2, "0x%0h ", read_data, TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(2, ("\n"));
            `endif
                `TRACE(2, ("%t: LMEM: Issuing write request, req_addr[0]=0x%x mask=%b, tag=%b\n", 
                            $time, d_tile_addr, lsu_req_if.req_data.mask, lsu_req_if.req_data.tag));
                `TRACE(2, ("d_line being written to MEM: "));
                `TRACE_ARRAY1D(2, "0x%0h ", d_line, TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(2, ("\n"));
            end
        end
    end

// DEBUGGING TRACES
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

endmodule

`endif // TCU_OP
