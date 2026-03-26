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

module VX_tcu_op_core import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
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

`ifdef TCU_TYPE_DPI
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY = FADD_LATENCY;// + FRND_LATENCY;
    localparam FEOP_LATENCY = FMUL_LATENCY + FRND_LATENCY;
`elsif TCU_TYPE_BHF
    localparam FMUL_LATENCY = 2;
    localparam FADD_LATENCY = 1;
    localparam FRND_LATENCY = 1;
    localparam FACC_LATENCY = FADD_LATENCY;// + FRND_LATENCY; // $clog2(2 * TCU_TC_K + 1) * (FADD_LATENCY + FRND_LATENCY);
    localparam FEOP_LATENCY = FMUL_LATENCY + FRND_LATENCY; // (FMUL_LATENCY + FRND_LATENCY) + 1 + FACC_LATENCY;
`else
    `error "VX_tcu_op_core: TCU_TYPE_DPI or TCU_TYPE_BHF must be defined"
`endif
    localparam MDATA_QUEUE_DEPTH = 1; // At maximum we have another intruction pending when the current one is finishing
    localparam XBAR_LATENCY      = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / 32; // TODO: Not sure if condition is legit

    `UNUSED_VAR(execute_if.data.rs3_data)
    `UNUSED_VAR(execute_if.data.op_args)


    function automatic [1:0] calc_lg_i_ratio (input [3:0] fmt);
        case (fmt)
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
`ifdef TCU_TYPE_BHF
        `TRACE(1, ("[tcu_op_core]: TCU_TYPE_BHF defined!\n\n"));
`elsif TCU_TYPE_DPI
        `TRACE(1, ("[tcu_op_core]: TCU_TYPE_DPI defined!\n\n"));
`endif
    end


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// INITIALIZATIONS & FSM

    /* Memory Requests handling */
    localparam BYTES_PER_MEM_REQUEST = LSU_WORD_SIZE * `NUM_LSU_LANES;
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
    localparam C_BUF_SLOTS = 6'(TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / `NUM_LSU_LANES - 1); // Amount of responses we can store before accumulating
    
    `STATIC_ASSERT ((TCU_FEOP_STEPS != 32) || C_BUF_SLOTS == '0, ("for 32 steps, we dont accumulate C"));

    /* Used to calculate the total A, B blocks */
    localparam LG_REGS_PER_BLOCK = $clog2(`NUM_LSU_LANES);

    reg [`XLEN-1:0] K;
    reg [3:0]       fmt_s;
    reg [3:0]       fmt_d;
    reg [1:0]       sparsity; // 0: Dense x Dense, 1: Dense x Sparse, 2: Sparse x Sparse

    reg [`XLEN-1:0] a_tile_addr;
    reg             a_tile_addr_valid;                             // Is set to false when all A blocks have been requested
    reg [`XLEN-1:0] a_req_blocks_remaining;                        // Requested to be fetched
    reg [A_BUF_SLOTS-1:0] a_blk_rq_bits;
    reg [A_BUF_SLOTS-1:0] a_blk_ld_bits;
    reg [A_BUF_SLOTS-1:0] a_active_block;
    reg [A_BUF_SLOTS-1:0] a_load_block;
    reg [A_BUF_SLOTS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] A_buffered; // Holds loaded data to be processed
    wire a_req_ready = a_tile_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (a_blk_rq_bits != '1); // A requests start only after the ACCU is initialized with the values of C

    always @ (posedge clk) begin
        if (busy) begin
            if (last_step_in_block_a && (rd_req_fire && grant_onehot == MATRIX_ID_BITS'(2))) begin
                `TRACE(1, ("[tcu_op_core]: [NEW]: Requested & processed an A block, unchanged a_blk_rq_bits=%b\n", a_blk_rq_bits));
            end
            if (last_step_in_block_a && (rd_rsp_fire && rsp_matrix_id == MATRIX_ID_BITS'(4))) begin
                `TRACE(1, ("[tcu_op_core]: [NEW]: Loaded & processed an A block, a_blk_ld_bits=%b->%b\n", a_blk_rq_bits, ~a_blk_ld_bits));
            end
        end
    end

    reg [`XLEN-1:0] b_tile_addr;
    reg             b_tile_addr_valid;                             // Is set to false when all B blocks have been requested
    reg [`XLEN-1:0] b_req_blocks_remaining;                        // Requested to be fetched
    reg [B_BUF_SLOTS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] B_buffered; // Holds loaded data to be processed
    wire b_req_ready = b_tile_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (b_blk_rq_bits != '1);
    reg [B_BUF_SLOTS-1:0] b_blk_rq_bits;
    reg [B_BUF_SLOTS-1:0] b_blk_ld_bits;
    reg [B_BUF_SLOTS-1:0] b_active_block;
    reg [B_BUF_SLOTS-1:0] b_load_block;

    reg [`XLEN-1:0] a_bitmap_addr;
    reg [`XLEN-1:0] b_bitmap_addr;
    reg             bitmap_addr_valid;
    reg [`XLEN-1:0] bitmap_req_blocks_remaining;   // Requested to be fetched
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_blk_rq_bits;
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_blk_ld_bits;
    reg [BITMAP_BUF_SLOTS-1:0] bitmap_active_block;
    reg [`XLEN-1:0] bitmap_blocks_loaded;                                    // Loaded but not processed
    reg [BITMAP_BUF_SLOTS-1:0][`NUM_THREADS-1:0][`XLEN-1:0] Bitmap_buffered; // Holds loaded data to be processed
    wire bitmap_req_ready = bitmap_addr_valid && (c_blocks_requested == TCU_C_BLOCKS_IN_ACCU) && (bitmap_blk_rq_bits != '1); // Bitmap requests start only after the ACCU is initialized with the values of C

    reg [`XLEN-1:0]                      c_tile_addr;
    reg                                  c_tile_addr_valid;    // Is set to false when all C blocks have been requested
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_requested;     // Requested to be fetched
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_loaded;        // Loaded but not accumulated
    reg [$clog2(TCU_C_BLOCKS_IN_ACCU):0] c_blocks_accumulated;   // Accumulated / loaded (once loaded they are directly accumulated)

    reg [`MAX(0, C_BUF_SLOTS-1):0][`NUM_THREADS-1:0][`XLEN-1:0] C_buffered; // Holds loaded data to be accumulated
    `UNUSED_VAR (C_buffered); // Only used when C_BUF_SLOTS > 0

    wire c_req_ready = init_flag && c_tile_addr_valid;
     
    reg [`XLEN-1:0] d_tile_addr;

    wire op_ctx_full;
    wire execute_ready_no_txbar = (execute_if.data.op_type == INST_TCU_MMA_OP) && (~busy_r) && (~mqueue_full) && (~op_ctx_full);
    wire execute_txbar_req = execute_if.valid && execute_ready_no_txbar;
    assign execute_if.ready = execute_ready_no_txbar && txbar_bus_if.ready;
    wire execute_fire = execute_txbar_req && txbar_bus_if.ready;

    wire [1:0] lg_i_ratio = calc_lg_i_ratio(fmt_s);
    wire [3:0] i_ratio = 4'(1 << lg_i_ratio);

    // Delayed FEOP-valid (from FMUL-latency pipe), forward-declared for feop_enable.
    wire valid_in_delayed;
    /* feop_enable handles back-pressure.
       Stall only when a delayed FEOP write beat is pending and ACCU cannot accept it. */
       // TODO: Add valid_in_delayed...
    wire feop_enable = accu_queues_ready; // || ~valid_in_delayed;

    wire mem_stall = valid_out && ~tcu_lsu_mem_if.req_ready;

    reg  init_r;
    reg  flush_r;
    wire init_flag  = init_r  | (execute_fire ? execute_if.data.rs2_data[7][1] : 1'b0);
    wire flush_flag = flush_r | (execute_fire ? execute_if.data.rs2_data[7][0] : 1'b0);

    /* Set when execute_fire and Cleared when result_fire */
    wire busy = busy_r || execute_fire;
    reg  busy_r;
    // Debugging: counts cycles where TCU stalls because accumulator queues are full.
    reg [`XLEN-1:0] full_queue_stall_cycles;

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
        end else begin
            // Initialization
            if (execute_fire) begin
                a_tile_addr_valid <= 1'b1;
                b_tile_addr_valid <= 1'b1;
                c_tile_addr_valid <= init_flag;
                bitmap_addr_valid <= (execute_if.data.rs2_data[5][1:0] >= 2'd1) ? 1'b1 : 1'b0;

                b_req_blocks_remaining <= '0;
                c_blocks_requested     <= init_flag ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);

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
                
                c_blocks_loaded      <= init_flag ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);
                bitmap_blocks_loaded <= '0;

                c_blocks_accumulated <= init_flag ? '0 : $clog2(TCU_C_BLOCKS_IN_ACCU + 1)'(TCU_C_BLOCKS_IN_ACCU);

                // Compute total blocks using the incoming instruction fields to avoid stale values
                if (2'(execute_if.data.rs2_data[5]) == 2'b00) begin
                    /* a_req_blocks_remaining = K * TCU_TC_M_OP / i_ratio */
                    a_req_blocks_remaining  <= 32'((`XLEN'(execute_if.data.rs2_data[2]) << LG_TCU_TC_M_OP) >> (32'(calc_lg_i_ratio(4'(execute_if.data.rs2_data[3]))) + LG_REGS_PER_BLOCK));
                    /* b_req_blocks_remaining = K * TCU_TC_N_OP / i_ratio */
                    b_req_blocks_remaining  <= 32'((`XLEN'(execute_if.data.rs2_data[2]) << LG_TCU_TC_N_OP) >> (32'(calc_lg_i_ratio(4'(execute_if.data.rs2_data[3]))) + LG_REGS_PER_BLOCK));
                    /* No bitmap in dense case */
                    bitmap_req_blocks_remaining <= '0;
                end
            `ifndef TCU_DISABLE_S1
                else if (2'(execute_if.data.rs2_data[5]) == 2'b01) begin
                    /* a_req_blocks_remaining = A_compressed_blocks */
                    a_req_blocks_remaining  <= 32'((`XLEN'(execute_if.data.rs2_data[2]) << LG_TCU_TC_M_OP) >> (32'(calc_lg_i_ratio(4'(execute_if.data.rs2_data[3]))) + LG_REGS_PER_BLOCK));
                    /* b_req_blocks_remaining = B_compressed_blocks */
                    b_req_blocks_remaining  <= (`XLEN)'(execute_if.data.rs2_data[1]);
                    /* bitmap_req_blocks_remaining = ceil(K / SETS_PER_S1_BITMAP_BLOCK) */
                    bitmap_req_blocks_remaining <= 32'((`XLEN'(execute_if.data.rs2_data[2]) + SETS_PER_S1_BITMAP_BLOCK - 1) >> LG_SETS_PER_S1_BITMAP_BLOCK);
                end 
            `endif
                else begin /* s2 case */
                    /* a_req_blocks_remaining = A_compressed_blocks */
                    a_req_blocks_remaining  <= (`XLEN)'(execute_if.data.rs2_data[0]);
                    /* b_req_blocks_remaining = B_compressed_blocks */
                    b_req_blocks_remaining  <= (`XLEN)'(execute_if.data.rs2_data[1]);
                    /* bitmap_req_blocks_remaining = ceil(K * 2 / 32) */
                    bitmap_req_blocks_remaining <= 32'((`XLEN'(execute_if.data.rs2_data[2]) + 15) >> 4);
                end

                /* Get configuration from the instruction */
                a_tile_addr <= (`XLEN)'(execute_if.data.rs1_data[0]);
                b_tile_addr <= (`XLEN)'(execute_if.data.rs1_data[1]);
                c_tile_addr <= (`XLEN)'(execute_if.data.rs1_data[2]);
                d_tile_addr <= (`XLEN)'(execute_if.data.rs1_data[3]);

                a_bitmap_addr <= (`XLEN)'(execute_if.data.rs1_data[4]);
                b_bitmap_addr <= (`XLEN)'(execute_if.data.rs1_data[5]);
                
                K <= (`XLEN)'(execute_if.data.rs2_data[2]);
                fmt_s  <= 4'(execute_if.data.rs2_data[3]);
                fmt_d  <= 4'(execute_if.data.rs2_data[4]);
                sparsity <= 2'(execute_if.data.rs2_data[5]);

                full_queue_stall_cycles <= '0;
            end
            if (result_fire) begin
                full_queue_stall_cycles <= '0;
            end
            if (busy && ~accu_queues_ready) begin
                full_queue_stall_cycles <= full_queue_stall_cycles + 1'b1;
            end
            // TODO: Move it to WORK ASSIGNMENT? (or REQ-RSP) part of code
            if (issue_busy) begin
                if (last_step_in_block_a) begin
                    // a_blocks_processed <= a_blocks_processed + 1'b1;
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(2))) begin
                        a_blk_rq_bits <= (a_blk_rq_bits >> 1);
                        `TRACE(1, ("%t: [NEW] A_processed && ~A_requested: a_blk_rq_bits=%b->%b\n", $time, a_blk_rq_bits, a_blk_rq_bits >> 1));
                    end
                    if (~(rd_rsp_fire && rsp_matrix_id == MATRIX_ID_BITS'(2))) begin
                        a_blk_ld_bits <= a_blk_ld_bits & ~a_active_block;
                        `TRACE(1, ("%t: [NEW] A_processed && ~A_loaded: a_blk_ld_bits=%b->%b\n", $time, a_blk_ld_bits, (a_blk_ld_bits & ~a_active_block)));
                    end
                    a_active_block <= ~a_active_block; // 01->10, 10->01
                    `TRACE(1, ("%t: [NEW] A_processed: a_active_block=%b->%b a_blk_req_bits=%b a_blk_ld_bits=%b\n", $time, a_active_block, ~a_active_block, a_blk_rq_bits, a_blk_ld_bits));
                end
                if (last_step_in_block_b) begin
                    // b_blocks_processed <= b_blocks_processed + 1'b1;
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(4))) begin
                        b_blk_rq_bits <= (b_blk_rq_bits >> 1);
                        `TRACE(1, ("%t: [NEW] B_processed && ~B_requested: b_blk_rq_bits=%b->%b\n", $time, b_blk_rq_bits, b_blk_rq_bits >> 1));
                    end
                    if (~(rd_rsp_fire && rsp_matrix_id == MATRIX_ID_BITS'(4))) begin
                        b_blk_ld_bits <= b_blk_ld_bits & ~b_active_block;
                        `TRACE(1, ("%t: [NEW] B_processed && ~B_loaded: b_blk_ld_bits=%b->%b\n", $time, b_blk_ld_bits, (b_blk_ld_bits & ~b_active_block)));
                    end
                    b_active_block <= ~b_active_block; // 01->10, 10->01
                    `TRACE(1, ("%t: [NEW] B_processed: b_active_block=%b->%b b_blk_req_bits=%b b_blk_ld_bits=%b\n", $time, b_active_block, ~b_active_block, b_blk_rq_bits, b_blk_ld_bits));
                end
                if (last_step_in_bitmap_block) begin
                    // bitmap_blocks_processed <= bitmap_blocks_processed + 1'b1;
                    if (~(rd_req_fire && grant_onehot == MATRIX_ID_BITS'(1))) begin
                        bitmap_blk_rq_bits <= (bitmap_blk_rq_bits >> 1);
                        `TRACE(1, ("%t: [NEW] Bitmap_processed && ~Bitmap_requested: bitmap_blk_rq_bits=%b->%b\n", $time, bitmap_blk_rq_bits, bitmap_blk_rq_bits >> 1));
                    end
                    if (~(rd_rsp_fire && rsp_matrix_id == MATRIX_ID_BITS'(1))) begin
                        bitmap_blk_ld_bits <= bitmap_blk_ld_bits & ~bitmap_active_block;
                        `TRACE(1, ("%t: [NEW] Bitmap_processed && ~Bitmap_loaded: bitmap_blk_ld_bits=%b->%b\n", $time, bitmap_blk_ld_bits, (bitmap_blk_ld_bits & ~bitmap_active_block)));
                    end
                    bitmap_active_block <= ~bitmap_active_block; // 01->10, 10->01
                    `TRACE(1, ("%t: [NEW] Bitmap_processed: bitmap_active_block=%b->%b bitmap_blk_req_bits=%b bitmap_blk_ld_bits=%b\n", $time, bitmap_active_block, ~bitmap_active_block, bitmap_blk_rq_bits, bitmap_blk_ld_bits));
                end
            end
        end
    end

// INITIALIZATIONS & FSM
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY REQUEST HANDLING

    localparam MATRIX_ID_BITS = 4;
    /* First, decide whose turn it is to issue a read request */
    /*                                    [3]          [2]          [1]            [0]         */
    wire [MATRIX_ID_BITS-1:0] reqs = {c_req_ready, b_req_ready, a_req_ready, bitmap_req_ready};
    wire [MATRIX_ID_BITS-1:0] grant_onehot;
    wire rd_req_valid;
    wire rd_req_fire = rd_req_valid && tcu_lsu_mem_if.req_ready;

    // Pointer advances if (rd_req_valid && tcu_lsu_mem_if.req_ready)
    VX_rr_rot_arbiter #(
        .NUM_REQS (MATRIX_ID_BITS)
    ) cyclic_loader (
        .clk (clk),
        .reset (reset),
        .requests (reqs),
        .grant_onehot (grant_onehot),
        .grant_valid (rd_req_valid),
        .grant_ready (tcu_lsu_mem_if.req_ready)
    );
                          
    wire [`XLEN-1:0] req_rd_addr = grant_onehot[0] ? b_bitmap_addr : // Convenient for s1 case, not used in s2 case
                                   grant_onehot[1] ? a_tile_addr   :
                                   grant_onehot[2] ? b_tile_addr   :
                                                     c_tile_addr;

    always @ (posedge clk) begin
        if (~reset && rd_req_fire) begin
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
                        `TRACE(1, ("%t: [NEW] Bitmap_requested && ~Bitmap_processed: bitmap_blk_rq_bits=%b->%b\n", $time, bitmap_blk_rq_bits, {|bitmap_blk_rq_bits, 1'b1}));
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
                        `TRACE(1, ("%t: [NEW] A_requested && ~A_processed: a_blk_rq_bits=%b->%b\n", $time, a_blk_rq_bits, {|a_blk_rq_bits, 1'b1}));
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
                        `TRACE(1, ("%t: [NEW] B_requested && ~B_processed: b_blk_rq_bits=%b->%b\n", $time, b_blk_rq_bits, {|b_blk_rq_bits, 1'b1}));
                    end
                end
                MATRIX_ID_BITS'(8): begin  // C
                    if (32'(c_blocks_requested) == (TCU_C_BLOCKS_IN_ACCU - 1)) begin
                        c_tile_addr       <= '0;
                        c_tile_addr_valid <= 1'b0; // Completed all requests for this tile
                    end else begin
                        c_tile_addr <= c_tile_addr + BYTES_PER_MEM_REQUEST;
                    end
                    c_blocks_requested <= c_blocks_requested + 1'b1;
                end
                default: begin
                    `TRACE(1, ("[tcu_op_core]: ERROR: Unexpected request tag %d\n", rsp_matrix_id));
                end
            endcase
        end
    end

// MEMORY REQUEST HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY RESPONSE HANDLING

    wire rd_rsp_fire = tcu_lsu_mem_if.rsp_valid && tcu_lsu_mem_if.rsp_ready;

    wire [MATRIX_ID_BITS-1:0] rsp_matrix_id = tcu_lsu_mem_if.rsp_data.tag.uuid[MATRIX_ID_BITS-1:0];

                                      // If this is an bitmap block, accept it if we have a free slot in Bitmap_buffered
    assign tcu_lsu_mem_if.rsp_ready = (rsp_matrix_id == MATRIX_ID_BITS'(1)) ? (bitmap_blk_ld_bits != '1) :
                                      // If this is an A block, accept it if we have a free slot in A_buffered
                                      (rsp_matrix_id == MATRIX_ID_BITS'(2)) ? (a_blk_ld_bits != '1) : // A block is accepted if there is at least 1 free slot 
                                      // If this is a  B block, accept it if we have a free slot in B_buffered
                                      (rsp_matrix_id == MATRIX_ID_BITS'(4)) ? (b_blk_ld_bits != '1) :
                                      // If this is a  C block, accept it if we have a free slot in C_buffered
                                      (rsp_matrix_id == MATRIX_ID_BITS'(8)) ? ((c_blocks_loaded % (C_BUF_SLOTS+1) == C_BUF_SLOTS) ? accu_enable : 1'b1) :
                                      1'b0;
    
    wire accumulate_c = (rsp_matrix_id == MATRIX_ID_BITS'(8)) && rd_rsp_fire && (c_blocks_loaded - c_blocks_accumulated == C_BUF_SLOTS) && accu_enable;
    
    localparam int C_BLOCKS_PER_FEOP_BLOCK = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE / `NUM_LSU_LANES;
    localparam LG_C_BLOCKS_PER_FEOP_BLOCK = $clog2(C_BLOCKS_PER_FEOP_BLOCK);
    wire [$clog2(TCU_FEOP_STEPS):0] c_blk_idx = ($clog2(TCU_FEOP_STEPS+1))'(c_blocks_accumulated >> LG_C_BLOCKS_PER_FEOP_BLOCK);

    wire [C_BUF_SLOTS:0][`NUM_THREADS-1:0][`XLEN-1:0] C_feop_block;
    if (C_BUF_SLOTS > 0) begin : g_c_feop_block
        assign C_feop_block = {tcu_lsu_mem_if.rsp_data.data, C_buffered};
    end else begin : g_c_feop_block_no_buffer
        assign C_feop_block = tcu_lsu_mem_if.rsp_data.data;
    end

    always @(posedge clk) begin
        // Load data mechanism
        if (~reset && rd_rsp_fire) begin
            case (rsp_matrix_id)
                MATRIX_ID_BITS'(1): begin  // Bitmap
                    bitmap_blocks_loaded <= bitmap_blocks_loaded + 1'b1; // Necessary for the bitmap_block_ready
                    Bitmap_buffered[bitmap_blk_ld_bits[0]] <= tcu_lsu_mem_if.rsp_data.data; // Double buffering
                    if (~last_step_in_bitmap_block) begin
                        bitmap_blk_ld_bits <= {|bitmap_blk_ld_bits, 1'b1}; // 00->01, 01->11, 10->11
                        `TRACE(1, ("%t: [NEW] Bitmap_loaded && ~Bitmap_processed: bitmap_blk_ld_bits=%b->%b\n", $time, bitmap_blk_ld_bits, {|bitmap_blk_ld_bits, 1'b1}));
                    end else begin
                        bitmap_blk_ld_bits <= ~bitmap_blk_ld_bits; // 01->10, 10->01
                        `TRACE(1, ("%t: [NEW] Bitmap_loaded && Bitmap_processed: bitmap_blk_ld_bits=%b->%b\n", $time, bitmap_blk_ld_bits, ~bitmap_blk_ld_bits));
                    end 
                end
                MATRIX_ID_BITS'(2): begin  // A
                    A_buffered[~a_load_block[0]] <= tcu_lsu_mem_if.rsp_data.data; // Double buffering
                    if (~last_step_in_block_a) begin
                        a_blk_ld_bits <= ((a_blk_ld_bits == 2'b00) ? a_load_block : 2'b11); // 00->a_load_block, 01->11, 10->11        // {|a_blk_ld_bits, 1'b1}; 
                        `TRACE(1, ("%t: [NEW] A_loaded && ~A_processed: a_blk_ld_bits=%b->%b\n", $time, a_blk_ld_bits, ((a_blk_ld_bits == 2'b00) ? a_load_block : 2'b11)));
                    end else begin
                        a_blk_ld_bits <= ~a_blk_ld_bits; // 01->10, 10->01
                        `TRACE(1, ("%t: [NEW] A_loaded && A_processed: a_blk_ld_bits=%b->%b\n", $time, a_blk_ld_bits, ~a_blk_ld_bits));
                    end 
                    a_load_block <= ~a_load_block; // 01->10, 10->01
                end
                MATRIX_ID_BITS'(4): begin  // B
                    B_buffered[~b_load_block[0]] <= tcu_lsu_mem_if.rsp_data.data; // Double buffering
                    if (~last_step_in_block_b) begin
                        b_blk_ld_bits <= ((b_blk_ld_bits == 2'b00) ? b_load_block : 2'b11); // 00->b_load_block, 01->11, 10->11
                        `TRACE(1, ("%t: [NEW] B_loaded && ~B_processed: b_blk_ld_bits=%b->%b\n", $time, b_blk_ld_bits, ((b_blk_ld_bits == 2'b00) ? b_load_block : 2'b11)));
                    end else begin
                        b_blk_ld_bits <= ~b_blk_ld_bits; // 01->10, 10->01
                        `TRACE(1, ("%t: [NEW] B_loaded && B_processed: b_blk_ld_bits=%b->%b\n", $time, b_blk_ld_bits, ~b_blk_ld_bits));
                    end 
                    b_load_block <= ~b_load_block; // 01->10, 10->01
                end
                MATRIX_ID_BITS'(8): begin  // C
                    if (~accumulate_c && C_BUF_SLOTS > 0) begin
                        C_buffered[c_blocks_loaded % (C_BUF_SLOTS+1)] <= tcu_lsu_mem_if.rsp_data.data; // Buffer until you can accumulate them all in 1 cycle
                    end
                    c_blocks_loaded <= c_blocks_loaded + 1'b1;
                end
                default: begin
                    `TRACE(1, ("[tcu_op_core]: ERROR: Unexpected response tag %d\n", rsp_matrix_id));
                end
            endcase

            if (accumulate_c) begin
                c_blocks_accumulated <= c_blocks_accumulated + (C_BUF_SLOTS + 1);
            end
        end
    end

// MEMORY RESPONSE HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// WORK ASSIGNMENT TO FEOPS
    
// *************************************************************************************************************
// FSM VARIABLES
    // Step number reduction
    // TODO: Calculate steps as (total_steps - skips)
    wire [LG_TCU_FEOP_M_STEPS:0] vertical_steps   = (LG_TCU_FEOP_M_STEPS+1)'((32'(a_non_zeros) + (TCU_FEOP_BLOCK_M_SIZE - 1)) >> LG_TCU_FEOP_BLOCK_M_SIZE);
    wire [LG_TCU_FEOP_N_STEPS:0] horizontal_steps = (LG_TCU_FEOP_N_STEPS+1)'((32'(b_non_zeros) + (TCU_FEOP_BLOCK_N_SIZE - 1)) >> LG_TCU_FEOP_BLOCK_N_SIZE);
    wire [LG_TCU_FEOP_STEPS:0]   set_steps_raw    = (LG_TCU_FEOP_STEPS+1)'(vertical_steps * horizontal_steps);
    wire [LG_TCU_FEOP_STEPS:0]   set_steps        = (set_steps_raw == '0) ? (LG_TCU_FEOP_STEPS+1)'(1) : set_steps_raw;
    wire [LG_TCU_FEOP_M_STEPS:0] vertical_skips   = (LG_TCU_FEOP_M_STEPS+1)'(32'(a_zeros) >> LG_TCU_FEOP_BLOCK_M_SIZE);
    wire [LG_TCU_FEOP_N_STEPS:0] horizontal_skips = (LG_TCU_FEOP_N_STEPS+1)'(32'(b_zeros) >> LG_TCU_FEOP_BLOCK_N_SIZE);

    reg [LG_TCU_FEOP_STEPS-1:0] step; // step increments from 0 -> set_steps
    reg [`XLEN-1:0] set;

    // TODO: Fix all operations on 32'() to the smallest possible size

    /* horizontal_steps_safe avoids division with 0 */
    wire [LG_TCU_FEOP_N_STEPS:0] horizontal_steps_safe = (horizontal_steps == '0) ? (LG_TCU_FEOP_N_STEPS+1)'(1) : horizontal_steps;
    /*  m = (step / horizontal_steps) * BLOCK_M  */
    wire [LG_TCU_TC_M_OP-1:0] m = LG_TCU_TC_M_OP'((32'(step) / 32'(horizontal_steps_safe)) << LG_TCU_FEOP_BLOCK_M_SIZE);
    /*  n = (step % horizontal_steps) * BLOCK_N  */
    wire [LG_TCU_TC_N_OP-1:0] n = LG_TCU_TC_N_OP'((32'(step) % 32'(horizontal_steps_safe)) << LG_TCU_FEOP_BLOCK_N_SIZE);

    wire last_step_in_set       = (step == LG_TCU_FEOP_STEPS'(set_steps - (LG_TCU_FEOP_STEPS+1)'(1))) && issue_busy;
    wire last_step_in_block_a   = last_step_in_set && last_set_in_block_a;
    wire last_step_in_block_b   = last_step_in_set && last_set_in_block_b;
    wire last_step_in_execution = last_step_in_set && (set == K - 1);

`ifndef TCU_DISABLE_S1
    wire last_step_in_bitmap_block = last_step_in_set
                                  && (((sparsity == 2'd1) && (((set + 1) & (SETS_PER_S1_BITMAP_BLOCK-1)) == 0))
                                   || ((sparsity == 2'd2) && (((set + 1) & (SETS_PER_S2_BITMAP_BLOCK-1)) == 0)));
`else
    wire last_step_in_bitmap_block = last_step_in_set && (sparsity == 2'd2) && (((set + 1) & (SETS_PER_S2_BITMAP_BLOCK-1)) == 0);
`endif
    
    reg issuing_done;

    `UNUSED_VAR (vertical_skips);

    always @ (posedge clk) begin
        if (~reset && last_step_in_set) begin
            `TRACE(1, ("%t: last_step_in_set: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (~reset && last_step_in_block_a) begin
            `TRACE(1, ("%t: last_step_in_block_a: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (~reset && last_step_in_block_b) begin
            `TRACE(1, ("%t: last_step_in_block_b: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (~reset && last_step_in_execution) begin
            `TRACE(1, ("%t: last_step_in_execution: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (~reset && issuing_done) begin
            `TRACE(1, ("%t: issuing_done: set=%0d, m=%0d, n=%0d\n", $time, set, m, n));
        end
        if (reset) begin
            set  <= '0;
            step <= '0;
            issuing_done <= 0;
        end
        if (~reset && execute_fire) begin
            set  <= '0;
            step <= '0;
            issuing_done <= 0;
        end
        if (~reset && result_fire) begin
            set  <= '0;
            step <= '0;
            issuing_done <= 0;
        end
        if (last_step_in_execution && feop_enable) begin
            issuing_done <= 1;
        end
        if (~reset && issue_busy) begin
            if (last_step_in_set) begin
                step <= '0;
                set <= set + 1;
            end else begin
                step <= step + 1;
            end
        end
    end


    /* Stalls when no new data have arrived  */
    // TODO: Make B available not only when it is written to B_BUFF but also the moment it arrives from LMEM
    // TODO: wait for next block if the current STEP requires it: DONE - remove if it gives no speedup
    wire bitmap_block_ready = (sparsity == 2'd2) ? (bitmap_blocks_loaded > (set >> LG_SETS_PER_S2_BITMAP_BLOCK)) :
                            `ifndef TCU_DISABLE_S1
                              (sparsity == 2'd1) ? (bitmap_blocks_loaded > (set >> LG_SETS_PER_S1_BITMAP_BLOCK)) :
                            `endif
                              1'b1; // Always ready in dense case

    // A storage format depends on sparsity mode:
    // - s2: A is compressed, so set span is number of non-zeros.
    // - s1/s0: A is dense in memory, so each set always spans full M dimension.
    wire [LG_TCU_TC_M_OP:0] a_set_elems = (sparsity == 2'd2) ? a_non_zeros : (LG_TCU_TC_M_OP+1)'(TCU_TC_M_OP);

    wire a_curr_loaded = |(a_blk_ld_bits & a_active_block);
    wire a_next_loaded = |(a_blk_ld_bits & ~a_active_block);
    // In s1/s0 modes, A is dense in memory and bitmap extraction scans the whole set.
    // In s2 mode, A is compressed and only non-zero payload is needed.
    wire [`XLEN-1:0] a_window_need = (sparsity == 2'd2) ? `MIN((32'(a_non_zeros)), (32'(m) + TCU_FEOP_BLOCK_M_SIZE)) : 32'(a_set_elems);
    wire a_window_ready = a_curr_loaded && (((a_offset + a_window_need) <= (32'(i_ratio) << $clog2(`NUM_LSU_LANES))) || a_next_loaded);

    wire b_curr_loaded = |(b_blk_ld_bits & b_active_block);
    wire b_next_loaded = |(b_blk_ld_bits & ~b_active_block);
    wire b_window_ready = b_curr_loaded && (((b_offset + `MIN((32'(b_non_zeros)), (32'(n) + TCU_FEOP_BLOCK_N_SIZE))) <= (32'(i_ratio) << $clog2(`NUM_LSU_LANES))) || b_next_loaded);
    
    wire issue_busy = busy_r && feop_enable && ~issuing_done && ~accumulate_c
                      && a_window_ready
                      && b_window_ready
                      && bitmap_block_ready;

// FSM VARIABLES
// *************************************************************************************************************
// BITMAP PROCESSING

    // Set Extraction
    // TODO: Fix the flattening here
    wire [TCU_TC_M_OP-1:0][`XLEN-1:0] a_set = a_set_flat_processed;
    wire [TCU_TC_N_OP-1:0][`XLEN-1:0] b_set = b_set_flat;

    // Bitmap extraction for sparse case
    wire [TCU_TC_M_OP-1:0] a_bitmap_in;
    wire [TCU_TC_N_OP-1:0] b_bitmap_in;

    wire [TCU_TC_M_OP-1:0] a_bitmap_out;
    wire [TCU_TC_N_OP-1:0] b_bitmap_out;

    wire [TCU_TC_M_OP-1:0] a_bitmap_s0;
    wire [TCU_TC_N_OP-1:0] b_bitmap_s0;

`ifndef TCU_DISABLE_S1
    wire [TCU_TC_M_OP-1:0] a_bitmap_s1;
    wire [TCU_TC_N_OP-1:0] b_bitmap_s1;
`endif

    wire [TCU_TC_M_OP-1:0] a_bitmap_s2;
    wire [TCU_TC_N_OP-1:0] b_bitmap_s2;

    wire [LG_TCU_TC_M_OP:0] a_zeros;
    wire [LG_TCU_TC_N_OP:0] b_zeros;

    wire [LG_TCU_TC_M_OP:0] a_non_zeros = TCU_TC_M_OP - a_zeros;
    wire [LG_TCU_TC_N_OP:0] b_non_zeros = TCU_TC_N_OP - b_zeros;

    wire [TCU_TC_M_OP-1:0][LG_TCU_TC_M_OP-1:0] a_addresses;
    wire [TCU_TC_N_OP-1:0][LG_TCU_TC_N_OP-1:0] b_addresses;

    assign a_bitmap_s0 = '1;
    assign b_bitmap_s0 = '1;

`ifndef TCU_DISABLE_S1

    /* Bitmap extraction supports down to 4-bit values */
    for (genvar i = 0; i < TCU_TC_M_OP; i++) begin : g_extract_bitmap
        assign a_bitmap_s1[i] = (i_ratio == 4'd1) ? |a_set_flat[(32'(i) << 5) +: 32] :
                                (i_ratio == 4'd2) ? |a_set_flat[(32'(i) << 4) +: 16] :
                                (i_ratio == 4'd4) ? |a_set_flat[(32'(i) << 3) +: 8]  :
                                (i_ratio == 4'd8) ? |a_set_flat[(32'(i) << 2) +: 4]  :
                                                    1'b1;
    end

    assign b_bitmap_s1 = Bitmap_buffered[(set >> LG_SETS_PER_S1_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][set & (SETS_PER_S1_BITMAP_BLOCK-1)];
`endif

    assign a_bitmap_s2 = Bitmap_buffered[(set >> LG_SETS_PER_S2_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][                         set & (SETS_PER_S2_BITMAP_BLOCK-1)];
    assign b_bitmap_s2 = Bitmap_buffered[(set >> LG_SETS_PER_S2_BITMAP_BLOCK) & (BITMAP_BUF_SLOTS-1)][(`NUM_LSU_LANES >> 1) + (set & (SETS_PER_S2_BITMAP_BLOCK-1))];

    assign a_bitmap_in = (sparsity == 2'd2) ? a_bitmap_s2 : 
                        `ifndef TCU_DISABLE_S1
                         (sparsity == 2'd1) ? a_bitmap_s1 :
                        `endif
                                              a_bitmap_s0;

    assign b_bitmap_in = (sparsity == 2'd2) ? b_bitmap_s2 :
                        `ifndef TCU_DISABLE_S1
                         (sparsity == 2'd1) ? b_bitmap_s1 :
                        `endif 
                                              b_bitmap_s0;

    VX_tcu_32_way_sorter #(
    ) a_sorter (
        .in_bitmap(a_bitmap_in),
        .out_bitmap(a_bitmap_out),
        .out_address(a_addresses),
        .zero_pop_count(a_zeros)  // Total zeros in the 32 elements given
    );

    VX_tcu_32_way_sorter #(
    ) b_sorter (
        .in_bitmap(b_bitmap_in),
        .out_bitmap(b_bitmap_out),
        .out_address(b_addresses),
        .zero_pop_count(b_zeros)  // Total zeros in the 32 elements given
    );

// BITMAP PROCESSING
// *************************************************************************************************************
// ADDRESS EXTRACTION

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] a_step_addresses = a_addresses[m +: TCU_FEOP_BLOCK_M_SIZE];
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] b_step_addresses = b_addresses[n +: TCU_FEOP_BLOCK_N_SIZE];

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] a_step_valids = a_bitmap_out[m +: TCU_FEOP_BLOCK_M_SIZE];
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] b_step_valids = b_bitmap_out[n +: TCU_FEOP_BLOCK_N_SIZE];

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] a_step_addresses_delayed;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] b_step_addresses_delayed;

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] a_step_valids_delayed;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] b_step_valids_delayed;
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

    localparam int A_BUF_W = `NUM_LSU_LANES * `XLEN;
    localparam int B_BUF_W = `NUM_LSU_LANES * `XLEN;
    // NEW: Deal with assymetric set size
    // TODO: Limit a_offset size to WorstCaseScenario: $clog2(`NUM_LSU_LANES * biggest_i_ratio) = 5+3=8 bits
    reg [`XLEN-1:0] a_offset;
    reg [`XLEN-1:0] b_offset;

    localparam int A_SET_W = TCU_TC_M_OP * `XLEN;
    localparam int B_SET_W = TCU_TC_N_OP * `XLEN;

    wire [A_SET_W-1:0] a_set_flat_processed;
`ifdef TCU_DISABLE_S1
    assign a_set_flat_processed = a_set_flat;
`else // TCU_DISABLE_S1
    reg [A_SET_W-1:0] a_set_flat_compressed; // Only used in S1 case

    // Compress the 32 logical A elements (width = 32 / i_ratio) into contiguous positions.
    // a_addresses[i] points to the source logical-element address for compressed position i.
    always @(*) begin
        a_set_flat_compressed = '0;
        for (int i = 0; i < TCU_TC_M_OP; ++i) begin
            if (a_bitmap_out[i]) begin
                case (i_ratio)
                    4'd1: a_set_flat_compressed[(i << 5) +: 32] = a_set_flat[(int'(a_addresses[i]) << 5) +: 32];
                    4'd2: a_set_flat_compressed[(i << 4) +: 16] = a_set_flat[(int'(a_addresses[i]) << 4) +: 16];
                    4'd4: a_set_flat_compressed[(i << 3) +: 8]  = a_set_flat[(int'(a_addresses[i]) << 3) +: 8];
                    4'd8: a_set_flat_compressed[(i << 2) +: 4]  = a_set_flat[(int'(a_addresses[i]) << 2) +: 4];
                    default: begin
                        // Unsupported ratio: leave compressed data zeroed.
                    end
                endcase
            end
        end
    end

    assign a_set_flat_processed = (sparsity == 2'd1) ? a_set_flat_compressed : a_set_flat;
`endif

    wire [A_BUF_W * A_BUF_SLOTS-1:0] A_window   = {A_buffered[a_active_block[0]], A_buffered[~a_active_block[0]]};
    wire [A_SET_W-1:0]               a_set_flat = (A_SET_W)'(A_window >> (a_offset << ($clog2(`XLEN) - 32'(lg_i_ratio))));

    wire [B_BUF_W * B_BUF_SLOTS-1:0] B_window   = {B_buffered[b_active_block[0]], B_buffered[~b_active_block[0]]};
    wire [B_SET_W-1:0]               b_set_flat = (B_SET_W)'(B_window >> (b_offset << ($clog2(`XLEN) - 32'(lg_i_ratio))));

    wire last_set_in_block_a = (a_offset + 32'(a_set_elems)) >= (32'(i_ratio) << $clog2(`NUM_LSU_LANES));
    wire last_set_in_block_b = (b_offset + 32'(b_non_zeros)) >= (32'(i_ratio) << $clog2(`NUM_LSU_LANES));

    always @ (posedge clk) begin
        if (reset) begin
            a_offset <= '0;
            b_offset <= '0;
        end else begin
            if (execute_fire) begin
                a_offset <= '0;
                b_offset <= '0; 
            end
            if (last_step_in_set) begin
                a_offset <= (a_offset + 32'(a_set_elems)) & ((32'(i_ratio) << $clog2(`NUM_LSU_LANES))-1);
                b_offset <= (b_offset + 32'(b_non_zeros)) & ((32'(i_ratio) << $clog2(`NUM_LSU_LANES))-1);
            end
        end
    end
// TODO: ^ Move to FSM VARIABLES

    // TODO: Remove for-genvar and make 1 feop module that produces BLOCK_M x BLOCK_N output 
    for (genvar id = 0; id < TCU_FEOP_BLOCK_M_SIZE; id++) begin : g_feop_units

        wire [`XLEN-1:0] a_elem = `XLEN'(a_set_flat_processed >> (((32'(m) + 32'(id))) << ($clog2(`XLEN) - 32'(lg_i_ratio)))); 
        // TODO: Fix the flattening here too
        wire [TCU_FEOP_BLOCK_N_SIZE*`XLEN-1:0] b_row_flat = (TCU_FEOP_BLOCK_N_SIZE*`XLEN)'(b_set_flat >> ((32'(n) >> lg_i_ratio) << $clog2(`XLEN)));
        wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`XLEN-1:0] b_row = b_row_flat;

        wire [TCU_FEOP_BLOCK_N_SIZE-1:0] feop_bitmap = a_step_valids[id] ? b_step_valids : '0;

        VX_tcu_feop #(
            .N (TCU_FEOP_BLOCK_N_SIZE),
            .FMUL_LATENCY (FMUL_LATENCY),
            .FRND_LATENCY (FRND_LATENCY),
            .ID (id) // DEBUGGING ONLY
        ) feop (
            .clk             (clk),
            .reset           (reset),
            .enable          (feop_enable), // Enables FEOP processing
            .valid_in        (issue_busy),  // Set in every new assignment
            .valid_in_bitmap (feop_bitmap), // Valid bits for the current step
            .fmt_s           (fmt_s),
            .fmt_d           (fmt_d),
            .a_elem          (a_elem),
            .b_row           (b_row),
            .d_block         (d_block[id])
        );

    `ifdef DBG_TRACE_TCU
        always @(posedge clk) begin
            if (issue_busy) begin
                `TRACE(1, ("%t: FEOP-enq(%0d): wid=%0d, a_elem(idx=%0d)=0x%0h, set=%0d, m=%0d, n=%0d, id=%0d, step=%0d\n", $time, id, execute_if.data.header.wid,  ((32'(m) + 32'(id)) >> lg_i_ratio), a_elem, set, m, n, id, step));
                `TRACE(1, ("b_row="));
                `TRACE_ARRAY1D(1, "0x%0h", b_row, (TCU_FEOP_BLOCK_N_SIZE >> lg_i_ratio));
                `TRACE(1, ("\n"));

                `TRACE(1, ("\n"));
                `TRACE(1, ("a_set="));
                `TRACE_ARRAY1D(1, "0x%0h", a_set, (TCU_TC_M_OP >> lg_i_ratio));
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_set="));
                `TRACE_ARRAY1D(1, "0x%0h", b_set, (TCU_TC_N_OP >> lg_i_ratio));
                `TRACE(1, ("\n"));

                `TRACE(1, ("\n"));
                `TRACE(1, ("a_bitmap_in="));
                `TRACE_ARRAY1D(1, "%b", a_bitmap_in, TCU_TC_M_OP);
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_bitmap_in="));
                `TRACE_ARRAY1D(1, "%b", b_bitmap_in, TCU_TC_N_OP);
                `TRACE(1, ("\n"));
                `TRACE(1, ("a_bitmap_out="));
                `TRACE_ARRAY1D(1, "%b", a_bitmap_out, TCU_TC_M_OP);
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_bitmap_out="));
                `TRACE_ARRAY1D(1, "%b", b_bitmap_out, TCU_TC_N_OP);
                `TRACE(1, ("\n"));
                `TRACE(1, ("a_addresses="));
                `TRACE_ARRAY1D(1, "%x", a_addresses, TCU_TC_M_OP);
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_addresses="));
                `TRACE_ARRAY1D(1, "%x", b_addresses, TCU_TC_N_OP);
                `TRACE(1, ("\n"));

                `TRACE(1, ("\n"));
                `TRACE(1, ("a_offset=%0d b_offset=%0d\n", a_offset, b_offset));
                `TRACE(1, ("a_zeros=%0d a_non_zeros=%0d vertical_steps=%0d vertical_skips=%0d\n",     a_zeros, a_non_zeros, vertical_steps, vertical_skips));
                `TRACE(1, ("b_zeros=%0d b_non_zeros=%0d horizontal_steps=%0d horizontal_skips=%0d\n", b_zeros, b_non_zeros, horizontal_steps, horizontal_skips));
                `TRACE(1, ("last_set_in_block_a=%b last_set_in_block_b=%b, last step_in_block_a=%b last_step_in_block_b=%b\n", last_set_in_block_a, last_set_in_block_b, last_step_in_block_a, last_step_in_block_b));
                
                `TRACE(1, ("\n"));
                `TRACE(1, ("a_step_addresses="));
                `TRACE_ARRAY1D(1, "%x", a_step_addresses, TCU_FEOP_BLOCK_M_SIZE);
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_step_addresses="));
                `TRACE_ARRAY1D(1, "%x", b_step_addresses, TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(1, ("\n"));
                `TRACE(1, ("a_step_valids="));
                `TRACE_ARRAY1D(1, "%b", a_step_valids, TCU_FEOP_BLOCK_M_SIZE);
                `TRACE(1, ("\n"));
                `TRACE(1, ("b_step_valids="));
                `TRACE_ARRAY1D(1, "%b", b_step_valids, TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(1, ("\n"));
            end
        end
    `endif // DBG_TRACE_TCU
    end

// FEOPs
// *************************************************************************************************************
    // NEW

    // Models FEOP latency for control signals valid_in (issue_busy)
    VX_pipe_register #(
        .DATAW  (1),
        .RESETW (1),
        .DEPTH  (FEOP_LATENCY)
    ) pipe_fmul_ctrl (
        .clk     (clk),
        .reset   (reset),
        .enable  (feop_enable),
        .data_in (issue_busy),
        .data_out(valid_in_delayed)
    );

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][`XLEN-1:0] write_data = accumulate_c ? C_feop_block : d_block;

// NEW^2 BEGIN: Code dealing with the ACCU addresses
    localparam int STEP_ELEM_CNT = TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE;

    wire [LG_TCU_FEOP_STEPS-1:0] read_block_idx =
    {
      d_line_to_flush_delayed[LG_TCU_FEOP_BLOCK_M_SIZE + LG_TCU_FEOP_N_STEPS +: (LG_TCU_FEOP_STEPS - LG_TCU_FEOP_N_STEPS)],
      d_line_to_flush_delayed[LG_TCU_FEOP_N_STEPS-1:0]
    };

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] write_addr_row;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] write_addr_col;

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] read_row_valid;
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0] write_addr_row_valid;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0] write_addr_col_valid;

    `UNUSED_VAR({write_addr_row, write_addr_col, write_addr_row_valid, write_addr_col_valid});

    wire [STEP_ELEM_CNT-1:0][`XLEN-1:0] read_data;
    wire [LG_TCU_FEOP_STEPS-1:0] write_block_idx = c_blk_idx[LG_TCU_FEOP_STEPS-1:0];
    wire [LG_TCU_FEOP_BLOCK_M_SIZE-1:0] read_row_in_block =
        LG_TCU_FEOP_BLOCK_M_SIZE'((32'(d_line_to_flush_delayed) >> LG_TCU_FEOP_N_STEPS) & (TCU_FEOP_BLOCK_M_SIZE-1));
// NEW^2 END

//NEW^3 BEGIN: Reduce registers for addresses
    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][LG_TCU_TC_M_OP-1:0] c_blk_rows;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][LG_TCU_TC_N_OP-1:0] c_blk_cols;

    for (genvar i = 0; i < TCU_FEOP_BLOCK_M_SIZE; i++) begin : g_row_addresses
        // While flushing, read only the row that feeds the current d_line.
        assign read_row_valid[i] = (read_row_in_block == LG_TCU_FEOP_BLOCK_M_SIZE'(i));

        wire [LG_TCU_TC_M_OP-1:0] write_row = LG_TCU_TC_M_OP'(((32'(write_block_idx) >> LG_TCU_FEOP_N_STEPS) << LG_TCU_FEOP_BLOCK_M_SIZE) + i);
        assign write_addr_row[i] = write_row;
        assign write_addr_row_valid[i] = 1'b1;

        assign c_blk_rows[i] = LG_TCU_TC_M_OP'(((32'(c_blk_idx[LG_TCU_FEOP_STEPS-1:0]) >> LG_TCU_FEOP_N_STEPS) << LG_TCU_FEOP_BLOCK_M_SIZE) + i);
    end
    for (genvar i = 0; i < TCU_FEOP_BLOCK_N_SIZE; i++) begin : g_col_addresses
        wire [LG_TCU_TC_N_OP-1:0] write_col = LG_TCU_TC_N_OP'(((32'(write_block_idx) & (TCU_FEOP_N_STEPS-1)) << LG_TCU_FEOP_BLOCK_N_SIZE) + i);
        assign write_addr_col[i] = write_col;
        assign write_addr_col_valid[i] = 1'b1;

        assign c_blk_cols[i] = LG_TCU_TC_N_OP'(((32'(c_blk_idx[LG_TCU_FEOP_STEPS-1:0]) & (TCU_FEOP_N_STEPS-1)) << LG_TCU_FEOP_BLOCK_N_SIZE) + i);
    end
//NEW^3 END

    wire accu_read_en = ready_to_flush_delayed;
    wire accu_write_valid = busy && (valid_in_delayed || accumulate_c);
    wire accu_enable = busy;

    wire accu_queues_ready;
    wire accu_ready_to_flush;

    VX_tcu_feop_accu #(
        .BLOCK_M      (TCU_FEOP_BLOCK_M_SIZE),
        .BLOCK_N      (TCU_FEOP_BLOCK_N_SIZE),
        .FACC_LATENCY (FACC_LATENCY),
        .XBAR_LATENCY (XBAR_LATENCY)
    ) feop_accu (
        .clk    (clk),
        .reset  (reset),
        .enable (accu_enable),
        .fmt_d  (fmt_d),

        .read_en             (accu_read_en),
        .read_row_valid      (read_row_valid),
        .read_block_idx      (read_block_idx),
        .read_data           (read_data),

        // All write inputs from FEOPs are delayed FMUL cycles
        .write_valid          (accu_write_valid),
        .write_ready          (accu_queues_ready),  // Not ready when even one of them is full
        .write_addr_row       (accumulate_c ? c_blk_rows : a_step_addresses_delayed),
        .write_addr_row_valid (accumulate_c ? '1 : a_step_valids_delayed),
        .write_addr_col       (accumulate_c ? c_blk_cols : b_step_addresses_delayed),
        .write_addr_col_valid (accumulate_c ? '1 : b_step_valids_delayed),
        .write_data           (write_data),
        .overwrite            (accumulate_c),
        .accu_ready_to_flush  (accu_ready_to_flush)
    );

    // NEW
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

// WORK ASSIGNMENT TO FEOPS
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// FLUSHING AND RESULT HANDLING

    wire [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE:0] d_lines_ready = (issuing_done && (c_blk_idx == TCU_FEOP_STEPS)) ? (LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE + 1)'((TCU_TC_M_OP * TCU_FEOP_N_STEPS)) : '0;
    reg  [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE:0] d_line_to_flush;

    // TODO: Fix condition - eg. simplify d_lines_ready (d_lines_ready == 32 && d_lines_ready != d_line_to_flush)
    // Stall flushing until all accumulator xbar queues are fully drained.
    // TODO: If ~(accu_ready_to_flush && tcu_lsu_mem_if.req_ready), dont stall ready_to_flush but valid_out
    // TODO: Move accu_ready_to_flush && tcu_lsu_mem_if.req_ready to delay only ready_to_flush_delayed
    wire ready_to_flush_raw = (d_lines_ready > d_line_to_flush) && accu_ready_to_flush && tcu_lsu_mem_if.req_ready; // start flushing after the steps 0,1 have been fully calculated
    wire ready_to_flush = flush_flag && ready_to_flush_raw;

    /* We are ready to commit at the moment we write the last d_line to LMEM */
    wire no_flush_complete = busy && ~flush_flag && ready_to_flush_raw && (d_line_to_flush == '0) && ~result_pending_r;
    wire result_pulse = no_flush_complete
                     || (busy && flush_flag && (32'(d_line_to_flush_delayed) == TCU_TC_M_OP * TCU_FEOP_N_STEPS - 1) && wr_req_fire);
    reg  result_pending_r;

// ----------------------------------- tx_bar HANDLING ----------------------------------------------
    wire [`XLEN-1:0] txbar_bar_id;
    
    assign txbar_bar_id = execute_if.data.rs2_data[6];
     
    wire [BAR_ADDR_W-1:0] txbar_addr;
    if (`NUM_WARPS > 1) begin : g_txbar_addr_w
        assign txbar_addr = {txbar_bar_id[NW_BITS-1:0], txbar_bar_id[BAR_ID_SHIFT +: NB_BITS]};
    end else begin : g_txbar_addr_wo
        assign txbar_addr = BAR_ADDR_W'(txbar_bar_id[BAR_ID_SHIFT +: NB_BITS]);
    end

    wire [BAR_ADDR_W-1:0] op_ctx_bar_addr;
    wire op_ctx_empty;
    wire result_pending = result_pending_r || result_pulse;
    wire result_txbar_req = result_pending && ~op_ctx_empty && result_if.ready;
    assign result_if.valid = result_pending && ~op_ctx_empty && txbar_bus_if.ready;
    wire result_fire = result_txbar_req && txbar_bus_if.ready;

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
        if (reset || execute_fire) begin
            result_pending_r <= 1'b0;
        end else begin
            if (result_pulse) begin
                result_pending_r <= 1'b1;
                `TRACE (1, ("%t: [tcu_op_core]: result_pulse=%b no_flush_complete=%b flush_flag=%b busy=%b busy_r=%b result_pending_r=%b d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d ready_to_flush_raw=%b ready_to_flush=%b accu_ready_to_flush=%b wr_req_fire=%b\n",
                    $time, result_pulse, no_flush_complete, flush_flag, busy, busy_r, result_pending_r, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, ready_to_flush_raw, ready_to_flush, accu_ready_to_flush, wr_req_fire));
            end
            if (execute_if.valid && ~execute_if.ready) begin
                `TRACE(1, ("%t: [tcu_op_core]: execute stall op_type=0x%0h busy=%b busy_r=%b mqueue_full=%b op_ctx_full=%b txbar_ready=%b result_pending=%b result_pending_r=%b result_pulse=%b result_fire=%b\n",
                    $time, execute_if.data.op_type, busy, busy_r, mqueue_full, op_ctx_full, txbar_bus_if.ready, result_pending, result_pending_r, result_pulse, result_fire));
                `TRACE(1, ("%t: [tcu_op_core]: stalled payload wid=%0d pc=0x%0h uuid=%0d rs1={A=0x%0h B=0x%0h C=0x%0h D=0x%0h Abm=0x%0h Bbm=0x%0h} rs2={Ablk=%0d Bblk=%0d K=%0d fmt_s=%0d fmt_d=%0d sparse=%0d bar=0x%0h flags=0x%0h}\n",
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
            // if (execute_fire || result_pulse || result_fire || result_fire_hazard) begin
            //     `TRACE(1, ("%t: [tcu_op_core]: lifecycle execute_valid=%b execute_ready=%b execute_fire=%b busy=%b busy_r=%b issue_busy=%b issuing_done=%b valid_in_delayed=%b accumulate_c=%b feop_enable=%b accu_queues_ready=%b op_ctx_empty=%b op_ctx_full=%b txbar_valid=%b txbar_ready=%b txbar_done_ready=%b result_pending=%b result_pending_r=%b result_pulse=%b result_fire=%b hazard=%b\n",
            //         $time, execute_if.valid, execute_if.ready, execute_fire, busy, busy_r, issue_busy, issuing_done, valid_in_delayed, accumulate_c, feop_enable, accu_queues_ready, op_ctx_empty, op_ctx_full, txbar_bus_if.valid, txbar_bus_if.ready, txbar_done_ready, result_pending, result_pending_r, result_pulse, result_fire, result_fire_hazard));
            // end
            if (ready_to_flush_raw || ready_to_flush || wr_req_fire || valid_out) begin
                `TRACE(1, ("%t: [tcu_op_core]: flush-state flush_flag=%b ready_to_flush_raw=%b ready_to_flush=%b valid_out=%b mem_stall=%b wr_req_fire=%b d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d accu_ready_to_flush=%b tcu_req_ready=%b\n",
                    $time, flush_flag, ready_to_flush_raw, ready_to_flush, valid_out, mem_stall, wr_req_fire, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, accu_ready_to_flush, tcu_lsu_mem_if.req_ready));
            end
            if (no_flush_complete) begin
                `TRACE(1, ("%t: [tcu_op_core]: no-flush completion path no_flush_complete=%b flush_flag=%b ready_to_flush_raw=%b d_line_to_flush=%0d result_pending_r=%b execute_valid=%b execute_ready=%b execute_fire=%b\n",
                    $time, no_flush_complete, flush_flag, ready_to_flush_raw, d_line_to_flush, result_pending_r, execute_if.valid, execute_if.ready, execute_fire));
            end
            if (result_fire) begin
                result_pending_r <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (~reset) begin
            if (execute_fire) begin
                `TRACE(1, ("%t: [tcu_op_core-txbar] START fire wid=%0d bar_id=0x%0h addr=%0d\n",
                    $time, execute_if.data.header.wid, txbar_bar_id, txbar_addr))
                `TRACE(1, ("%t: [tcu_op_core]: accepted payload uuid=%0d pc=0x%0h rs1={A=0x%0h B=0x%0h C=0x%0h D=0x%0h Abm=0x%0h Bbm=0x%0h} rs2={Ablk=%0d Bblk=%0d K=%0d fmt_s=%0d fmt_d=%0d sparse=%0d bar=0x%0h flags=0x%0h}\n",
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
                `TRACE(1, ("%t: [tcu_op_core-txbar] RESULT pulse pending=%0b opctx_empty=%0b txbar_ready=%0b\n",
                    $time, result_pending_r, op_ctx_empty, txbar_bus_if.ready))
            end
            if (result_fire) begin
                `TRACE(1, ("%t: [tcu_op_core-txbar] DONE fire wid=%0d addr=%0d\n",
                    $time, result_if.data.header.wid, op_ctx_bar_addr))
            end
            if (txbar_bus_if.valid && txbar_bus_if.ready) begin
                `TRACE(1, ("%t: [tcu_op_core-txbar] TXBAR xfer addr=%0d is_done=%0b\n",
                    $time, txbar_bus_if.data.addr, txbar_bus_if.data.is_done))
            end
        end
    end
// ----------------------------------- tx_bar HANDLING ----------------------------------------------


    // Stores u-ops until commit side accepts completion.
    wire [MDATA_WIDTH-1:0] mdata_queue_din, mdata_queue_dout;
    wire mqueue_full;

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

    assign {result_if.data.header.uuid, 
            result_if.data.header.wid, 
            result_if.data.header.PC, 
            result_if.data.header.rd} = mdata_queue_dout;

    assign result_if.data.header.wb    = 1'b0;
    assign result_if.data.header.wr_xregs = '0;
    assign result_if.data.header.tmask = {`NUM_THREADS{1'b1}};
    assign result_if.data.data  = '0;
    assign result_if.data.header.pid   =  0;
    assign result_if.data.header.sop   = 1'b1;
    assign result_if.data.header.eop   = 1'b1;

    wire [TCU_FEOP_BLOCK_M_SIZE-1:0][TCU_FEOP_BLOCK_N_SIZE-1:0][`XLEN-1:0] d_block;
    wire [TCU_FEOP_BLOCK_N_SIZE-1:0][`XLEN-1:0] d_line = read_data[(((32'(d_line_to_flush_delayed) >> LG_TCU_FEOP_N_STEPS) & (TCU_FEOP_BLOCK_M_SIZE-1)) << LG_TCU_FEOP_BLOCK_N_SIZE) +: TCU_FEOP_BLOCK_N_SIZE];


    /* Flush the D line to MEM - Pad with zeros to fill the 32 spots */
    localparam int PAD_LANES = `NUM_LSU_LANES - TCU_FEOP_BLOCK_N_SIZE;
    assign tcu_lsu_mem_if.req_data.data = rd_req_valid ? '0 : {{PAD_LANES{`XLEN'(0)}}, d_line};
    
    wire wr_req_fire = (valid_out && feop_enable) && tcu_lsu_mem_if.req_ready && (tcu_lsu_mem_if.req_data.rw == 1'b1);

    wire ready_to_flush_delayed;
    wire [LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE-1:0] d_line_to_flush_delayed;
    // Delay WB control signals to match FEOP latency
    VX_pipe_register #(
        .DATAW  (1 + $bits(d_line_to_flush_delayed)),
        .RESETW (1 + $bits(d_line_to_flush_delayed)),
        .DEPTH  (FEOP_LATENCY + FACC_LATENCY + XBAR_LATENCY)
    ) pipe_flush_dummy (
        .clk     (clk),
        .reset   (reset),
        .enable  (feop_enable && ~mem_stall),
        .data_in ({ready_to_flush,         d_line_to_flush[LG_TCU_TC_M_OP + LG_TCU_FEOP_BLOCK_N_SIZE - 1:0]}),
        .data_out({ready_to_flush_delayed, d_line_to_flush_delayed})
    );

    // Use the same stage as read_data (ready_to_flush_delayed) to avoid misalignment
    wire valid_out = ready_to_flush_delayed;

    always @ (posedge clk) begin
        if (reset) begin
            d_line_to_flush <= '0;
            busy_r <= 1'b0;
        end else begin

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
            end
        end
    end


// FLUSHING AND RESULT HANDLING
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// MEMORY INTERFACE

    /*                                   ready to read         ready to write    */
    assign tcu_lsu_mem_if.req_valid    = rd_req_valid || (valid_out && feop_enable); // feop_enable here ensures that we wont accumulate the same valid_out value twice
    assign tcu_lsu_mem_if.req_data.rw  = rd_req_valid ? 1'b0 /* read */ : 1'b1; /* write */
    /* tag shows which accu block is returned.
       Matrix ID is stored in the upper UUID bits, block index in the remaining UUID bits.
       Value bits are left for arbiters to overwrite. */
    // TODO: Clean this up - remove the zeros
    assign tcu_lsu_mem_if.req_data.tag.uuid[MATRIX_ID_BITS-1:0]  = rd_req_valid ? grant_onehot : '0;
    assign tcu_lsu_mem_if.req_data.tag.value = '0;

    wire [`NUM_LSU_LANES-1:0] wr_mask = {{(`NUM_LSU_LANES - TCU_FEOP_BLOCK_N_SIZE){1'b0}}, {TCU_FEOP_BLOCK_N_SIZE{1'b1}}};
    // For bitmap reads with K<16, request identical 16-bit lane masks for A-half and B-half:
    // [31:16] = 0...01...1 (K ones in LSBs), [15:0] = same.
    // TODO: Unnecessary checks for K >= 16
    wire [15:0] bitmap_half_mask = (16'hFFFF     >> (5'd16 - K[4:0]));
    wire [`NUM_LSU_LANES-1:0] bitmap_small_k_mask = {bitmap_half_mask, bitmap_half_mask};

`ifndef TCU_DISABLE_S1
    wire [31:0] bitmap_s1_mask   = (32'hFFFFFFFF >> (6'd32 - K[5:0]));
`endif
    assign tcu_lsu_mem_if.req_data.mask   = ~rd_req_valid ? wr_mask : 
                                            (grant_onehot == MATRIX_ID_BITS'(1) && K < 16 && sparsity == 2'd2) ? bitmap_small_k_mask : 
                                        `ifndef TCU_DISABLE_S1
                                            (grant_onehot == MATRIX_ID_BITS'(1) && K < 32 && sparsity == 2'd1) ? bitmap_s1_mask : 
                                        `endif
                                            {`NUM_LSU_LANES{1'b1}};
    assign tcu_lsu_mem_if.req_data.byteen = {`NUM_LSU_LANES{{LSU_WORD_SIZE{1'b1}}}};
    
    // preserve full byte address for correct LMEM detection
    localparam MEM_ASHIFT = `CLOG2(`MEM_BLOCK_SIZE);      // bytes -> block
    localparam MEM_ADDRW  = `MEM_ADDR_WIDTH - MEM_ASHIFT; // block address width
    localparam REQ_ASHIFT = `CLOG2(LSU_WORD_SIZE);        // bytes -> LSU word
    localparam [MEM_ADDRW-1:0] LMEM_ADDR_START = MEM_ADDRW'(`XLEN'(`LMEM_BASE_ADDR) >> MEM_ASHIFT);
    localparam [MEM_ADDRW-1:0] LMEM_ADDR_END   = MEM_ADDRW'((`XLEN'(`LMEM_BASE_ADDR) + `XLEN'(1 << `LMEM_LOG_SIZE)) >> MEM_ASHIFT);

    for (genvar l = 0; l < `NUM_LSU_LANES; l++) begin : g_mem_addr
        // wire [`XLEN-1:0] lane_byte_addr;
        // if (rd_req_valid) begin : g_rd
        //     if (grant_onehot[0] && (sparsity == 2'd2)) begin : g_bitmap
        //         if (l < `NUM_LSU_LANES/2) begin : g_a_bitmap
        //             assign lane_byte_addr = a_bitmap_addr + (`XLEN'(l) * LSU_WORD_SIZE);
        //         end else begin : g_b_bitmap
        //             assign lane_byte_addr = b_bitmap_addr + ((`XLEN'(l) - `NUM_LSU_LANES/2) * LSU_WORD_SIZE);
        //         end
        //     end else begin : g_data
        //         assign lane_byte_addr = req_rd_addr + (`XLEN'(l) * LSU_WORD_SIZE);
        //     end
        // end else begin : g_wr
        //     assign lane_byte_addr = d_tile_addr + (`XLEN'(l) * LSU_WORD_SIZE);
        // end
        wire [`XLEN-1:0] lane_byte_addr = rd_req_valid ? 
                                            ((grant_onehot[0] && sparsity == 2'd2)? 
                                                (l < `NUM_LSU_LANES/2 ? 
                                                    a_bitmap_addr + (`XLEN'(l) << $clog2(LSU_WORD_SIZE)) : 
                                                    b_bitmap_addr + ((`XLEN'(l) - (`NUM_LSU_LANES >> 1)) << $clog2(LSU_WORD_SIZE))) : 
                                                req_rd_addr + (`XLEN'(l) << $clog2(LSU_WORD_SIZE))) :
                                            d_tile_addr + (`XLEN'(l) << $clog2(LSU_WORD_SIZE));

        `UNUSED_VAR (lane_byte_addr[1:0]);
        wire [LSU_ADDR_WIDTH-1:0] word_addr = lane_byte_addr[LSU_ADDR_WIDTH + REQ_ASHIFT - 1 : REQ_ASHIFT]; // LSU word address per lane
        wire [MEM_ADDRW-1:0] block_addr = lane_byte_addr[`MEM_ADDR_WIDTH-1:MEM_ASHIFT];                     // MEM block address for LMEM flagging
        wire is_lmem = (block_addr >= LMEM_ADDR_START) && (block_addr < LMEM_ADDR_END);

        assign tcu_lsu_mem_if.req_data.flags[l][MEM_REQ_FLAG_FLUSH] = 1'b0;
        assign tcu_lsu_mem_if.req_data.flags[l][MEM_REQ_FLAG_IO]    = 1'b0;
        assign tcu_lsu_mem_if.req_data.flags[l][MEM_REQ_FLAG_LOCAL] = is_lmem;

        assign tcu_lsu_mem_if.req_data.addr[l] = word_addr;

        always @(posedge clk) begin
            if (~reset && rd_req_valid && ~is_lmem) begin
                `TRACE(1, ("%t: [tcu_op_core]: ERROR: Address 0x%0h is not in LMEM (word_addr=0x%0h, block_addr=0x%0h, is_lmem=%b)\n", 
                            $time, lane_byte_addr, word_addr, block_addr, is_lmem));
            end
        end
    end

    // TODO: In flushing, write back directly to GMEM

// MEMORY INTERFACE
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DEBUGGING TRACES

// NEW: Manage double-buffering DXA BEGIN
    always @ (posedge clk) begin
        if (reset || result_fire) begin
            init_r  <= 1'b0;
            flush_r <= 1'b0;
        end
        else begin
            if (execute_fire) begin
                init_r  <= execute_if.data.rs2_data[7][1];
                flush_r <= execute_if.data.rs2_data[7][0];
                `TRACE(1, ("init_flag=%b, flush_flag=%b\n", init_flag, flush_flag));
            end
        end
    end

// NEW: Manage double-buffering DXA END

    always_ff @(posedge clk) begin
        if (~reset) begin
            if (execute_fire) begin
                `TRACE(1, ("%t: [tcu_op_core] TCU execution fired, \nA_addr=0x%x, B_addr=0x%x, C_addr=0x%x, D_addr=0x%x, \nA_compressed_blocks=%0d, B_compressed_blocks=%0d, K=%0d, fmt_s=%0d, fmt_d=%0d, sparsity=%0d, A_bitmap_addr=0x%x, B_bitmap_addr=0x%x, barrier_ID=0x%x, flags=%b\n", 
                            $time, execute_if.data.rs1_data[0], execute_if.data.rs1_data[1], execute_if.data.rs1_data[2], execute_if.data.rs1_data[3], 
                            execute_if.data.rs2_data[0], execute_if.data.rs2_data[1], execute_if.data.rs2_data[2], execute_if.data.rs2_data[3], execute_if.data.rs2_data[4], execute_if.data.rs2_data[5],
                            execute_if.data.rs1_data[4], execute_if.data.rs1_data[5], execute_if.data.rs2_data[6], execute_if.data.rs2_data[7]));   
            
            end
            if (execute_if.valid && ~execute_if.ready) begin
                `TRACE(1, ("%t: [tcu_op_core] pending op snapshot uuid=%0d pc=0x%0h busy=%b busy_r=%b issue_busy=%b issuing_done=%b op_ctx_full=%b mqueue_full=%b txbar_ready=%b ready_to_flush=%b d_lines_ready=%0d\n",
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
                `TRACE(1, ("%t: c_blk_idx=%0d, step=%0d / %0d, set=%0d, m=%0d / %0d, n=%0d / %0d\n", 
                            $time, c_blk_idx, step, TCU_FEOP_STEPS, set, m, TCU_TC_M_OP, n, TCU_TC_N_OP));
            end
            if (busy && ready_to_flush) begin
                `TRACE(1, ("%t: [tcu_op_core] Commit Busy processing\n", $time));
            end
            if (feop_enable && last_step_in_execution) begin
                `TRACE(1, ("%t: [tcu_op_core] All iterations issued\n", $time));
            end
            if (result_fire) begin
                `TRACE(1, ("%t: [tcu_op_core] Result fired downstream\n", $time));
                `TRACE(1, ("%t: [tcu_op_core] Full-queue stall cycles=%0d\n", $time, full_queue_stall_cycles));
                if (full_queue_stall_cycles > 0 && sparsity == 2'b00) begin
                    `TRACE(1, ("%t: [tcu_op_core] ERROR: Full queue stalls should not happen for dense workloads\n", $time));
                end
            end
            if (execute_if.valid && mqueue_full) begin
                `TRACE(1, ("%t: [tcu_op_core]: ERROR: Back-pressure on issue side, mqueue_full=%b\n", $time, mqueue_full));
            end
            if (busy && ~accu_queues_ready) begin
                `TRACE(1, ("%t: [tcu_op_core] FULL QUEUES: accu_queues_ready=%b\n", $time, accu_queues_ready));
            end
            if (rd_req_fire) begin
                // Print requested info
                `TRACE(1, ("%t: LMEM: Issuing read request, req_addr[0]=0x%x mask=%b, tag=%b matrix=%0s, c_blocks_requested=%0d\n", 
                            $time, tcu_lsu_mem_if.req_data.addr[0] * LSU_WORD_SIZE, tcu_lsu_mem_if.req_data.mask, tcu_lsu_mem_if.req_data.tag,
                            (grant_onehot == 4'b0001) ? "Bitmap" : (grant_onehot == 4'b0010) ? "A" : (grant_onehot == 4'b0100) ? "B" : "C",
                            c_blocks_requested));
                `TRACE(1, ("a_req_blocks_remaining=%0d, b_req_blocks_remaining=%0d, total_c_blocks=%0d, bitmap_req_blocks_remaining=%0d\n",
                            a_req_blocks_remaining, b_req_blocks_remaining, TCU_C_BLOCKS_IN_ACCU, bitmap_req_blocks_remaining));
            end
            if (rd_rsp_fire) begin
                `TRACE(1, ("%t: LMEM: read rsp: tag=%x mask=%b,   matrix=%0s accumulate_c=%b\nc_blocks_loaded=%0d, c_blocks_accumulated=%0d, C_buf_idx=%0d\n", $time,
                            tcu_lsu_mem_if.rsp_data.tag, tcu_lsu_mem_if.rsp_data.mask,
                            (rsp_matrix_id == 4'b0001) ? "Bitmap" : (rsp_matrix_id == 4'b0010) ? "A" : (rsp_matrix_id == 4'b0100) ? "B" : "C",
                            accumulate_c, c_blocks_loaded, c_blocks_accumulated, c_blocks_loaded % (C_BUF_SLOTS+1)));
                
                for (integer l = 0; l < `NUM_LSU_LANES; l++) begin
                    if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                        if (rsp_matrix_id == 4'b0001) begin // Bitmap printing
                            `TRACE(1, ("    lane[%0d]: data=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                        end else begin
                            `TRACE(1, ("    lane[%0d]: data=%x\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                        end
                    end
                end
                if (rsp_matrix_id == 4'b0001) begin
                    `TRACE(1, ("\n"));
                    if (sparsity == 2'd2) begin
                        for (integer l = 0; l < `NUM_LSU_LANES; l++) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                if (l < `NUM_LSU_LANES / 2) begin
                                    `TRACE(1, ("    A_bitmap[%0d]=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                                end
                                else if (l >= `NUM_LSU_LANES / 2) begin
                                    `TRACE(1, ("    B_bitmap[%0d]=%b\n", l - `NUM_LSU_LANES / 2, tcu_lsu_mem_if.rsp_data.data[l]));
                                end
                            end
                        end
                    end
                `ifndef TCU_DISABLE_S1
                    else if (sparsity == 2'd1) begin
                        for (integer l = 0; l < `NUM_LSU_LANES; l++) begin
                            if (tcu_lsu_mem_if.rsp_data.mask[l]) begin
                                `TRACE(1, ("    B_bitmap[%0d]=%b\n", l, tcu_lsu_mem_if.rsp_data.data[l]));
                            end
                        end
                    end
                `endif
                end
                if (~(&tcu_lsu_mem_if.rsp_data.mask) && (rsp_matrix_id != 4'b0001 || K >= 32'(32))) begin
                    `TRACE(1, ("%t: [tcu_op_core]: ERROR: not all lanes valid in LMEM response\n", $time));
                end
            end
            if (valid_out && feop_enable && rd_req_valid) begin
                `TRACE(1, ("ERROR: both MEM read/write are valid\n"));
            end
            if (issue_busy && (32'(d_line_to_flush) == TCU_TC_M_OP * TCU_FEOP_BLOCK_N_SIZE)) begin
                `TRACE(1, ("%t:[tcu_op_core]: d_line_to_flush=%0d\n", $time, d_line_to_flush));
            end
            if (wr_req_fire) begin
                `TRACE(1, ("%t: [tcu_op_core]: Flushing, d_lines_ready=%0d d_line_to_flush=%0d d_line_to_flush_delayed=%0d\n read_block=%0d read_data=\n", $time, d_lines_ready, d_line_to_flush, d_line_to_flush_delayed, read_block_idx));
                `TRACE_ARRAY1D(1, "0x%0h ", read_data, TCU_FEOP_BLOCK_M_SIZE * TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(1, ("\n"));
                `TRACE(1, ("%t: LMEM: Issuing write request, req_addr[0]=0x%x mask=%b, tag=%b\n", 
                            $time, d_tile_addr, tcu_lsu_mem_if.req_data.mask, tcu_lsu_mem_if.req_data.tag));
                `TRACE(1, ("d_line being written to MEM: "));
                `TRACE_ARRAY1D(1, "0x%0h ", d_line, TCU_FEOP_BLOCK_N_SIZE);
                `TRACE(1, ("\n"));

                `TRACE(1, ("[tcu_op_core]: req_data.data="));
                `TRACE_ARRAY1D(1, "0x%0h", tcu_lsu_mem_if.req_data.data, `NUM_LSU_LANES);
                `TRACE(1, ("\n"));
            end
            if (busy_r) begin
            `ifndef TCU_DISABLE_S1
                if ((sparsity != 2'd0) && (sparsity != 2'd1) && (sparsity != 2'd2)) begin
                    `TRACE(1, ("[tcu_op_core]: ERROR: Invalid sparsity mode: %0d\n", sparsity));
                end
            `else
                if ((sparsity != 2'd0) && (sparsity != 2'd2)) begin
                    `TRACE(1, ("[tcu_op_core]: ERROR: Invalid sparsity mode: %0d\n", sparsity));
                end
            `endif
            end
            // if (busy) begin
            //     `TRACE(1, ("[tcu_op_core]: issue_busy=%b busy=%b feop_enable=%b issuing_done=%b a_window_ready=%b b_window_ready=%b bitmap_block_ready=%b\n", issue_busy, busy, feop_enable, issuing_done, a_window_ready, b_window_ready, bitmap_block_ready));
            //     `TRACE(1, ("[tcu_op_core]: accumulate_c=%b accu_queues_ready=%b valid_in_delayed=%b\n", accumulate_c, accu_queues_ready, valid_in_delayed));
            //     `TRACE(1, ("a_curr_loaded=%b a_next_loaded=%b a_window_ready=%b\n", a_curr_loaded, a_next_loaded, a_window_ready));
            //     `TRACE(1, ("b_curr_loaded=%b b_next_loaded=%b b_window_ready=%b\n", b_curr_loaded, b_next_loaded, b_window_ready));
            //     `TRACE(1, ("bitmap_blocks_loaded=%0d bitmap_block_ready=%b\n", bitmap_blocks_loaded, bitmap_block_ready));
            //     `TRACE(1, ("a_blk_ld_bits=%b a_active_block=%b a_load_block=%b\n", a_blk_ld_bits, a_active_block, a_load_block));
            //     `TRACE(1, ("b_blk_ld_bits=%b b_active_block=%b b_load_block=%b\n", b_blk_ld_bits, b_active_block, b_load_block));
            // end
        end
    end

// DEBUGGING TRACES
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

endmodule
