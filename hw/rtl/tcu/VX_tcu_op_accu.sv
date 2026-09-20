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

//------------------------------------------------------------------------------
// TCU_OP fixed-point accumulator.
//
// Holds the engine's TCU_TC_M_OP x TCU_TC_N_OP output tile as BANKS banks of
// TCU_FEOP_STEPS slots. Each slot carries
//
//     { exceptions, sticky, sum[W-1:0], carry[W-1:0], c_word[31:0] }
//
// where the accumulated products are kept in CARRY-SAVE form: the per-cycle
// update is a single 3:2 compressor, so the accumulate loop has no carry chain
// at all (a W-bit ripple in a RAM read-modify-write loop was the first version's
// critical path). The carry-propagate add happens once, on the flush port.
//
// Products arrive exact and unrounded from VX_tcu_op_mul and are aligned against
// a per-format CONSTANT anchor (tcu_op_anchor) over two stages. Alignment
// depends only on the product's own exponent, never on the stored value, so it
// is feed-forward and its depth is free. Because the bank read is asynchronous
// and a write is visible to the next cycle's read (RDW_MODE "W"), back-to-back
// accumulations into the same slot need NO hazard interlock: the crossbar
// outputs are always ready and the only back-pressure left is a genuine bank
// collision, which the per-input queues absorb.
//
// The C term never enters the fixed-point sum -- an fp32 C spans 254 binades
// against the products' 58, so no window anchored at the product scale holds
// both. It is parked in c_word (routed there by the same crossbar that places
// products) and fused in at flush: normalize the product sum, then a three-lane
// aligned add against C, rounding once. Rounding therefore happens once per
// output element per tile, not once per product and once per k-step.
//------------------------------------------------------------------------------
`include "VX_define.vh"

`ifdef TCU_OP

module VX_tcu_op_accu import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter int BLOCK_M          = 2,
    parameter int BLOCK_N          = 16,
    // Kept in the interface because the core sizes its address and valid
    // pipelines from the same number; nothing inside this module reads it.
    parameter int XBAR_LATENCY     = 1,
    parameter int XBAR_QUEUE_DEPTH = 2,
    // A queue with more than this many entries withholds issue credit.
    parameter int CREDIT_LIMIT     = 1
) (
    input wire clk,
    input wire reset,
    input wire enable,

    input wire [TCU_FMT_WIDTH-1:0] fmt_s, // selects the anchor constant

    // ---- READ / FLUSH PORT ----
    // flush_enable freezes the flush stages in lockstep with the core's flush
    // control when the memory port back-pressures.
    input  wire                              flush_enable,
    input  wire                              read_en,
    input  wire [BLOCK_M-1:0]                read_row_valid,
    input  wire [$clog2(TCU_FEOP_STEPS)-1:0] read_block_idx,
    output wire [BLOCK_N-1:0][`VX_CFG_XLEN-1:0] read_row_data, // fp32, rounded

    // ---- WRITE PORT ----
    input  wire                                        write_valid,
    output wire                                        write_ready,
    input  wire [BLOCK_M-1:0][$clog2(TCU_TC_M_OP)-1:0] write_addr_row,
    input  wire [BLOCK_M-1:0]                          write_addr_row_valid,
    input  wire [BLOCK_N-1:0][$clog2(TCU_TC_N_OP)-1:0] write_addr_col,
    input  wire [BLOCK_N-1:0]                          write_addr_col_valid,
    input  wire [BLOCK_M*BLOCK_N-1:0][TCU_OP_EXP_W-1:0] write_exp,
    input  wire [BLOCK_M*BLOCK_N-1:0][TCU_OP_MAG_W-1:0] write_mag,
    input  wire [BLOCK_M*BLOCK_N-1:0]                   write_sign,
    input  fedp_excep_t [BLOCK_M*BLOCK_N-1:0]           write_exc,
    input  wire                                        overwrite, // replace, don't add
    input  wire                                        write_is_c,
    input  wire [BLOCK_M*BLOCK_N-1:0][`VX_CFG_XLEN-1:0] write_c_data,

    // Registered: clear while every queue has room for all in-flight steps.
    // The core gates issue on this, so the write port can never refuse (D1).
    output wire                                        credit_ok,

    output wire                                        accu_ready_to_flush
);
    localparam LG_BLOCK_M = $clog2(BLOCK_M);
    localparam LG_BLOCK_N = $clog2(BLOCK_N);
    localparam int BANKS   = BLOCK_M * BLOCK_N;
    localparam int SLOTS   = TCU_FEOP_STEPS;
    localparam int BANK_AW = $clog2(SLOTS);
    localparam int BANK_BW = $clog2(BANKS);

    localparam int W        = TCU_OP_ACC_W;
    localparam int MAG_W    = TCU_OP_MAG_W;
    localparam int EXP_W    = TCU_OP_EXP_W;
    localparam int ACC_FRAC = TCU_OP_ACC_FRAC;
    // A term whose exponent equals the anchor occupies [ACC_FRAC+MAG_W-1:ACC_FRAC].
    localparam int PAD_W    = ACC_FRAC + MAG_W;
    localparam int SH_W     = $clog2(PAD_W) + 1;
    // Bank entry: {exc_nan, exc_inf, exc_sign, sticky, sum, carry, c_word}
    localparam int ENTRY_W  = 4 + 2 * W + `VX_CFG_XLEN;

    localparam int XBAR_INPUTS  = BANKS;
    localparam int XBAR_OUTPUTS = BANKS;
    localparam int XBAR_SELW    = $clog2(XBAR_OUTPUTS);
    // {is_c, overwrite, sign, exp, mag, slot, nan, inf}. During a C beat the
    // 32-bit C word rides in {exp[7:0], mag}, which alignment ignores.
    localparam int XBAR_DATAW   = 1 + 1 + 1 + EXP_W + MAG_W + BANK_AW + 2;
    localparam int QUEUE_W      = XBAR_DATAW + BANK_BW;
    localparam int QUEUE_OCCW   = `CLOG2(XBAR_QUEUE_DEPTH + 1);
    `STATIC_ASSERT (CREDIT_LIMIT >= 0 && CREDIT_LIMIT < XBAR_QUEUE_DEPTH,
        ("tcu_op_accu: CREDIT_LIMIT %0d outside [0, %0d)", CREDIT_LIMIT, XBAR_QUEUE_DEPTH))

    // Flush stages: entry, |sum+carry|, lzc, normalize, fuse-align, fuse-add, round.
    localparam int FLUSH_LATENCY = 7;
    `UNUSED_PARAM (XBAR_LATENCY)
    `UNUSED_PARAM (FLUSH_LATENCY)

    `STATIC_ASSERT (XBAR_INPUTS <= 32, ("tcu_op_accu: crossbar supports at most 32 inputs, got %0d", XBAR_INPUTS))
    `STATIC_ASSERT (PAD_W <= W, ("tcu_op_accu: TCU_OP_ACC_W too small for the magnitude and growth"))
    `STATIC_ASSERT (EXP_W + MAG_W >= `VX_CFG_XLEN, ("tcu_op_accu: payload too narrow for the C word"))

    // Anchor is a per-format constant, so alignment carries no state.
    wire [EXP_W-1:0] anchor = EXP_W'(tcu_op_anchor(fmt_s));
    // Exponent of a flushed term whose 24-bit magnitude is the normalized top of
    // the accumulator (see the normalize stage).
    wire [EXP_W-1:0] term_exp_base = EXP_W'(tcu_op_anchor(fmt_s) - ACC_FRAC + (W - MAG_W));

    localparam int FUSE_W = 32;
    // VX_tcu_tfr_align hard-codes its exponent ports to TCU_EXP_BITS, so the
    // fuse works in that width. Both fuse terms are fp32-scale, so their
    // exponents fit.
    localparam int FUSE_EXP_W = TCU_EXP_BITS;
    // A term with exponent == max lands at bits [25:2] of the fuse field
    // (VX_tcu_tfr_align pre-shifts product lanes by WI-23), so the field's LSB
    // has weight (max_exp - TCU_OP_EXP_BIAS - 2) and norm_round wants
    // max_exp + FUSE_W - (TCU_OP_EXP_BIAS + 2) + 254.
    localparam int FUSE_EXP_ADJ = TCU_OP_EXP_BIAS + 2 - 254 - FUSE_W;
    `STATIC_ASSERT ((254 + TCU_OP_C_EXP_K) < (1 << FUSE_EXP_W),
        ("tcu_op_accu: fuse exponent does not fit TCU_EXP_BITS"))

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ADDRESS DECODE

    wire [BANKS-1:0][BANK_AW-1:0] write_block;
    wire [BANKS-1:0][BANK_BW-1:0] write_bank;

    for (genvar i = 0; i < BANKS; ++i) begin : g_addr_decode
        wire [$clog2(TCU_TC_M_OP)-1:0] write_row = write_addr_row[i >> LG_BLOCK_N];
        wire [$clog2(TCU_TC_N_OP)-1:0] write_col = write_addr_col[i & (BLOCK_N-1)];

        assign write_block[i] = BANK_AW'(((32'(write_row) >> LG_BLOCK_M) << LG_TCU_FEOP_N_STEPS) + (32'(write_col) >> LG_BLOCK_N));
        assign write_bank[i]  = BANK_BW'(((32'(write_row) & (BLOCK_M-1)) << LG_BLOCK_N) + (32'(write_col) & (BLOCK_N-1)));
    end

    wire [XBAR_INPUTS-1:0] xbar_queue_full;
    wire [XBAR_INPUTS-1:0] queue_low_credit;
    wire [XBAR_INPUTS-1:0] xbar_queue_empty;
    wire [XBAR_INPUTS-1:0] xbar_ready_in;

    wire [XBAR_INPUTS-1:0] write_lane_valid;
    wire [XBAR_INPUTS-1:0] write_lane_blocked = write_lane_valid & xbar_queue_full;
    // Credits (D1) guarantee every in-flight step a slot, so acceptance is
    // unconditional. Critically, that means a lane's push must NOT depend on any
    // other lane: routing the 32-way reduction into write_lane_fire simply moved
    // the old global stall from clock-enable pins onto the queues' write-enable
    // pins, which was the worst path once feop_enable was gone.
    assign write_ready = 1'b1;
    `RUNTIME_ASSERT (~(enable && write_valid && (|write_lane_blocked)),
        ("%t: *** tcu_op_accu: write refused while credits are in force (blocked=%b)", $time, write_lane_blocked))
    wire write_fire = write_valid;
    wire [XBAR_INPUTS-1:0] write_lane_fire = write_lane_valid & {XBAR_INPUTS{write_fire}};

    for (genvar i = 0; i < BLOCK_M; ++i) begin : g_lane_valid_row
        for (genvar j = 0; j < BLOCK_N; ++j) begin : g_lane_valid_col
            assign write_lane_valid[(i << LG_BLOCK_N) + j] = write_addr_row_valid[i] && write_addr_col_valid[j];
        end
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// INPUT QUEUES

    wire [XBAR_INPUTS-1:0] xbar_use_queue;
    wire [XBAR_INPUTS-1:0] xbar_queue_push;
    wire [XBAR_INPUTS-1:0] xbar_queue_pop;
    wire [XBAR_INPUTS-1:0][XBAR_DATAW-1:0] xbar_queue_payload;
    wire [XBAR_INPUTS-1:0][BANK_BW-1:0]    xbar_queue_bank;

    for (genvar i = 0; i < XBAR_INPUTS; ++i) begin : g_xbar_queue
        wire [EXP_W-1:0] exp_field = write_is_c ? EXP_W'(write_c_data[i][31:24]) : write_exp[i];
        wire [MAG_W-1:0] mag_field = write_is_c ? MAG_W'(write_c_data[i][23:0])  : write_mag[i];
        wire [XBAR_DATAW-1:0] payload_in = {write_is_c, overwrite, write_sign[i], exp_field, mag_field,
                                            write_block[i], write_exc[i].is_nan, write_exc[i].is_inf};
        wire [QUEUE_W-1:0] queue_din = {payload_in, write_bank[i]};
        wire [QUEUE_W-1:0] queue_dout;
        wire [QUEUE_OCCW-1:0] queue_occ;
        assign {xbar_queue_payload[i], xbar_queue_bank[i]} = queue_dout;

        assign xbar_use_queue[i]  = ~write_lane_fire[i] && enable && ~xbar_queue_empty[i];
        assign xbar_queue_push[i] = write_lane_fire[i] && ~xbar_ready_in[i];
        assign xbar_queue_pop[i]  = xbar_use_queue[i]  &&  xbar_ready_in[i];

        VX_fifo_queue #(
            .DATAW  (QUEUE_W),
            .DEPTH  (XBAR_QUEUE_DEPTH),
            .LUTRAM (1)
        ) xbar_queue (
            .clk      (clk),
            .reset    (reset),
            .push     (xbar_queue_push[i]),
            .pop      (xbar_queue_pop[i]),
            .data_in  (queue_din),
            .data_out (queue_dout),
            .empty    (xbar_queue_empty[i]),
            `UNUSED_PIN(alm_empty),
            .full     (xbar_queue_full[i]),
            `UNUSED_PIN(alm_full),
            .size     (queue_occ)
        );

        // Compared against a constant, per queue: no wide arithmetic, and the
        // 32 results reduce to one registered bit below.
        assign queue_low_credit[i] = (32'(queue_occ) > CREDIT_LIMIT);
    end

    VX_pipe_register #(
        .DATAW  (1),
        .RESETW (1)
    ) pipe_credit (
        .clk      (clk),
        .reset    (reset),
        .enable   (1'b1),
        .data_in  (~(|queue_low_credit)),
        .data_out (credit_ok)
    );

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// CROSSBAR

    wire [XBAR_INPUTS-1:0][XBAR_DATAW-1:0] xbar_data_in;
    wire [XBAR_INPUTS-1:0][XBAR_SELW-1:0]  xbar_sel_in;
    wire [XBAR_INPUTS-1:0]                 xbar_valid_in;

    wire [XBAR_OUTPUTS-1:0][XBAR_DATAW-1:0] xbar_data_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_valid_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_ready_out;
    wire [XBAR_OUTPUTS-1:0]                 xbar_enable_out;

    wire [PERF_CTR_BITS-1:0] xbar_collisions;
    `UNUSED_VAR (xbar_collisions)

    for (genvar i = 0; i < XBAR_INPUTS; ++i) begin : g_xbar_inputs
        wire [EXP_W-1:0] d_exp_field = write_is_c ? EXP_W'(write_c_data[i][31:24]) : write_exp[i];
        wire [MAG_W-1:0] d_mag_field = write_is_c ? MAG_W'(write_c_data[i][23:0])  : write_mag[i];
        wire [XBAR_DATAW-1:0] direct_payload = {write_is_c, overwrite, write_sign[i], d_exp_field, d_mag_field,
                                                 write_block[i], write_exc[i].is_nan, write_exc[i].is_inf};
        assign xbar_data_in[i]  = xbar_use_queue[i] ? xbar_queue_payload[i] : direct_payload;
        assign xbar_sel_in[i]   = xbar_use_queue[i] ? xbar_queue_bank[i]    : write_bank[i];
        assign xbar_valid_in[i] = xbar_use_queue[i] || write_lane_fire[i];
    end

    // No accumulate hazard exists (see the header), so an output is ready
    // whenever the engine is running.
    for (genvar i = 0; i < XBAR_OUTPUTS; ++i) begin : g_xbar_outputs
        assign xbar_ready_out[i]  = enable;
        assign xbar_enable_out[i] = xbar_valid_out[i] && xbar_ready_out[i];
    end

    // D4: the destination splits into a group and a position, so the network is
    // BLOCK_M x (BLOCK_N -> BLOCK_N) followed by BLOCK_N x (BLOCK_M -> BLOCK_M)
    // rather than one BANKS x BANKS crossbar -- half the mux area, and 16-way
    // arbiters instead of 32-way. See VX_tcu_op_xbar.sv for why that is correct
    // for any traffic, and where it blocks where a full crossbar would not.
`ifdef TCU_OP_XBAR_FLAT
    // Escape hatch: the undecomposed crossbar with round-robin arbitration. This
    // is the configuration that passes today, and it passes only because
    // round-robin happens to mask an order dependence in the accumulate stage
    // (see the RUNTIME_ASSERT on establish ordering below). Selecting "P" here
    // reproduces the same corruption the decomposed network shows, on this very
    // topology -- which is how that order dependence was localised.
    wire xbar_busy = 1'b0;
    // Which network is live. Behind the trace level: several experiments were
    // once run believing this ifdef had selected the flat crossbar, with nothing
    // in the logs to confirm it.
    initial begin
        `TRACE(1, ("[tcu_op_accu] NETWORK=flat ARBITER=%s\n", FLAT_ARBITER))
    end
    localparam `STRING FLAT_ARBITER = "R";
    VX_stream_xbar #(
        .NUM_INPUTS    (XBAR_INPUTS),
        .NUM_OUTPUTS   (XBAR_OUTPUTS),
        .DATAW         (XBAR_DATAW),
        .ARBITER       (FLAT_ARBITER),
        .OUT_BUF       (0),
        .PERF_CTR_BITS (PERF_CTR_BITS)
    ) accu_xbar (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (xbar_valid_in),
        .data_in    (xbar_data_in),
        .sel_in     (xbar_sel_in),
        .ready_in   (xbar_ready_in),
        .valid_out  (xbar_valid_out),
        .data_out   (xbar_data_out),
        `UNUSED_PIN (sel_out),
        .ready_out  (xbar_ready_out),
        .collisions (xbar_collisions)
    );
`else
    wire xbar_busy;
    initial begin
        `TRACE(1, ("[tcu_op_accu] NETWORK=decomposed\n"))
    end
    VX_tcu_op_xbar #(
        .BLOCK_M       (BLOCK_M),
        .BLOCK_N       (BLOCK_N),
        .DATAW         (XBAR_DATAW),
        .PERF_CTR_BITS (PERF_CTR_BITS),
        .SEL_W         (XBAR_SELW)
    ) accu_xbar (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (xbar_valid_in),
        .data_in    (xbar_data_in),
        .sel_in     (xbar_sel_in),
        .ready_in   (xbar_ready_in),
        .valid_out  (xbar_valid_out),
        .data_out   (xbar_data_out),
        .ready_out  (xbar_ready_out),
        .busy       (xbar_busy),
        .collisions (xbar_collisions)
    );
`endif

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ALIGN (two stages: coarse shift, then fine shift and negate)

    wire [XBAR_OUTPUTS-1:0]              out_is_c;
    wire [XBAR_OUTPUTS-1:0]              out_overwrite;
    wire [XBAR_OUTPUTS-1:0]              out_sign;
    wire [XBAR_OUTPUTS-1:0][EXP_W-1:0]   out_exp;
    wire [XBAR_OUTPUTS-1:0][MAG_W-1:0]   out_mag;
    wire [XBAR_OUTPUTS-1:0][BANK_AW-1:0] out_slot;
    wire [XBAR_OUTPUTS-1:0]              out_nan;
    wire [XBAR_OUTPUTS-1:0]              out_inf;

    for (genvar i = 0; i < XBAR_OUTPUTS; ++i) begin : g_xbar_unpack
        assign {out_is_c[i], out_overwrite[i], out_sign[i], out_exp[i], out_mag[i],
                out_slot[i], out_nan[i], out_inf[i]} = xbar_data_out[i];
    end

    // Addend, ready for the compressor: already two's complement.
    wire [BANKS-1:0]                   a1_valid; // align stage 1 occupancy
    wire [BANKS-1:0]                   ac_valid;
    wire [BANKS-1:0][W-1:0]            ac_addend;
    wire [BANKS-1:0]                   ac_sticky;
    wire [BANKS-1:0][BANK_AW-1:0]      ac_slot;
    wire [BANKS-1:0]                   ac_establish; // overwrite or C beat
    wire [BANKS-1:0]                   ac_is_c;
    wire [BANKS-1:0]                   ac_nan;
    wire [BANKS-1:0]                   ac_inf;
    wire [BANKS-1:0]                   ac_sign;
    wire [BANKS-1:0][`VX_CFG_XLEN-1:0] ac_c_word;

    for (genvar i = 0; i < BANKS; ++i) begin : g_align
        // anchor >= exp for every representable term, so the difference is a
        // non-negative right shift. Past the field the term drops out into the
        // sticky bit rather than wrapping.
        wire [EXP_W-1:0] shift_full = anchor - out_exp[i];
        wire over_shift = (32'(shift_full) >= PAD_W);
        wire [SH_W-1:0] shift_amt = over_shift ? '0 : shift_full[SH_W-1:0];

        wire [PAD_W-1:0] padded = {out_mag[i], {ACC_FRAC{1'b0}}};

        // --- stage 1: coarse shift, in multiples of 8 ---
        wire [SH_W-1:0]  sh_hi = {shift_amt[SH_W-1:3], 3'b0};
        wire [PAD_W-1:0] coarse_w  = padded >> sh_hi;
        // Sticky: the low ACC_FRAC bits of `padded` are zero by construction, so
        // only the magnitude can ever be shifted out. Masking 24 bits replaces a
        // PAD_W-wide shift plus a PAD_W-wide OR, which was the stage's cost.
        wire [MAG_W-1:0] drop_mask = MAG_W'((MAG_W'(1) << (shift_amt - SH_W'(ACC_FRAC))) - MAG_W'(1));
        wire lost_bits = (32'(shift_amt) > ACC_FRAC) && (|(out_mag[i] & drop_mask));
        wire coarse_stick_w = over_shift ? (|out_mag[i]) : lost_bits;

        wire [PAD_W-1:0] coarse_r;
        wire [2:0]       sh_lo_r;
        wire             coarse_stick_r, over_r, sign_r1, is_c_r1, ovw_r1, nan_r1, inf_r1;
        wire             valid_r1;
        assign a1_valid[i] = valid_r1;
        wire [BANK_AW-1:0] slot_r1;
        wire [`VX_CFG_XLEN-1:0] c_word_r1;

        // A C beat carries its fp32 word in the exponent/magnitude fields.
        wire [`VX_CFG_XLEN-1:0] c_word_w = {out_exp[i][7:0], out_mag[i][23:0]};

        VX_pipe_register #(
            .DATAW  (1 + PAD_W + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + BANK_AW + `VX_CFG_XLEN),
            .RESETW (1),
            .DEPTH  (1)
        ) pipe_align1 (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({xbar_enable_out[i], coarse_w, shift_amt[2:0], coarse_stick_w, over_shift, out_sign[i], out_is_c[i], out_overwrite[i], out_nan[i], out_inf[i], out_slot[i], c_word_w}),
            .data_out ({valid_r1,           coarse_r, sh_lo_r,        coarse_stick_r, over_r,     sign_r1,     is_c_r1,     ovw_r1,           nan_r1,     inf_r1,     slot_r1,     c_word_r1})
        );

        // --- stage 2: fine shift, then two's complement ---
        wire [PAD_W-1:0] fine    = coarse_r >> sh_lo_r;
        wire [W-1:0] mag_field   = over_r ? '0 : W'(fine);
        // Negating here keeps the accumulate loop a pure compressor.
        wire [W-1:0] addend_w    = sign_r1 ? (~mag_field + W'(1)) : mag_field;
        wire sticky_w = coarse_stick_r;

        VX_pipe_register #(
            .DATAW  (1 + W + 1 + BANK_AW + 1 + 1 + 1 + 1 + 1 + `VX_CFG_XLEN),
            .RESETW (1),
            .DEPTH  (1)
        ) pipe_align2 (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  ({valid_r1,     addend_w,     sticky_w,     slot_r1,    (ovw_r1 | is_c_r1), is_c_r1,    nan_r1,    inf_r1,    sign_r1,    c_word_r1}),
            .data_out ({ac_valid[i],  ac_addend[i], ac_sticky[i], ac_slot[i], ac_establish[i],    ac_is_c[i], ac_nan[i], ac_inf[i], ac_sign[i], ac_c_word[i]})
        );
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// ACCUMULATE (carry-save: one compressor level, no carry chain)

    wire [BANKS-1:0][ENTRY_W-1:0] bank_rdata;
    wire [BANKS-1:0][ENTRY_W-1:0] bank_wdata;
    wire [BANKS-1:0]              bank_read_en;
    wire [BANKS-1:0]              bank_write_en;
    wire [BANKS-1:0][BANK_AW-1:0] bank_raddr;
    wire [BANKS-1:0][BANK_AW-1:0] bank_waddr;

    for (genvar i = 0; i < BANKS; ++i) begin : g_accumulate
        wire [W-1:0] cur_sum, cur_carry;
        wire         cur_sticky;
        wire         cur_nan, cur_inf, cur_inf_sign;
        wire [`VX_CFG_XLEN-1:0] cur_c;
        assign {cur_nan, cur_inf, cur_inf_sign, cur_sticky, cur_sum, cur_carry, cur_c} = bank_rdata[i];

        // 3:2 compression of {stored sum, stored carry, addend}.
        wire [W-1:0] csa_sum   = cur_sum ^ cur_carry ^ ac_addend[i];
        wire [W-1:0] csa_maj   = (cur_sum & cur_carry) | (cur_sum & ac_addend[i]) | (cur_carry & ac_addend[i]);
        // The top bit of csa_maj is shifted out by construction.
        wire [W-1:0] csa_carry = {csa_maj[W-2:0], 1'b0};
        `UNUSED_VAR (csa_maj)

        wire [W-1:0] new_sum   = ac_establish[i] ? (ac_is_c[i] ? W'(0) : ac_addend[i]) : csa_sum;
        wire [W-1:0] new_carry = ac_establish[i] ? W'(0) : csa_carry;

        wire new_sticky = ac_establish[i] ? (ac_is_c[i] ? 1'b0 : ac_sticky[i]) : (cur_sticky | ac_sticky[i]);
        // Opposite-signed infinities produce a NaN.
        wire inf_clash = cur_inf && ac_inf[i] && (cur_inf_sign != ac_sign[i]);
        wire new_nan  = ac_establish[i] ? (ac_is_c[i] ? 1'b0 : ac_nan[i]) : (cur_nan | ac_nan[i] | inf_clash);
        wire new_inf  = ac_establish[i] ? (ac_is_c[i] ? 1'b0 : ac_inf[i]) : (cur_inf | ac_inf[i]);
        wire new_isgn = ac_establish[i] ? ac_sign[i] : (cur_inf ? cur_inf_sign : ac_sign[i]);
        // A C beat parks its word and clears the product sum; anything else
        // leaves the parked word alone.
        wire [`VX_CFG_XLEN-1:0] new_c = ac_is_c[i] ? ac_c_word[i] : cur_c;

        assign bank_wdata[i]    = {new_nan, new_inf, new_isgn, new_sticky, new_sum, new_carry, new_c};
        assign bank_write_en[i] = enable && ac_valid[i];
        assign bank_waddr[i]    = ac_slot[i];
        // The accumulate read and the flush read are mutually exclusive: a flush
        // only starts once the pipeline has drained.
        assign bank_read_en[i]  = enable && ((read_en && read_row_valid[i >> LG_BLOCK_N]) || ac_valid[i]);
        assign bank_raddr[i]    = (enable && read_en) ? read_block_idx : ac_slot[i];

        VX_dp_ram #(
            .DATAW      (ENTRY_W),
            .SIZE       (SLOTS),
            .OUT_REG    (0),
            .RDW_MODE   ("W"), // a write is visible to the next cycle's read
            .RESET_RAM  (1),
            .INIT_VALUE (ENTRY_W'(0))
        ) accu_mem (
            .clk   (clk),
            .reset (reset),
            .read  (bank_read_en[i]),
            .write (bank_write_en[i]),
            .wren  (1'b1),
            .waddr (bank_waddr[i]),
            .wdata (bank_wdata[i]),
            .raddr (bank_raddr[i]),
            .rdata (bank_rdata[i])
        );
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// FLUSH: resolve the carry-save sum, normalize, fuse the C term, round once
//
// Seven stages on a port that moves one D line per cycle, so the depth costs
// nothing. Every stage was a separate critical path when they were merged.

    wire [LG_BLOCK_M-1:0] sel_row;
    if (BLOCK_M == 1) begin : g_one_row
        assign sel_row = '0;
    end else begin : g_enc_row
        VX_onehot_encoder #(
            .N (BLOCK_M)
        ) row_enc (
            .data_in    (read_row_valid),
            .data_out   (sel_row),
            `UNUSED_PIN (valid_out)
        );
    end

    for (genvar j = 0; j < BLOCK_N; ++j) begin : g_flush
        wire [ENTRY_W-1:0] sel_entry;
        if (BLOCK_M == 1) begin : g_direct
            assign sel_entry = bank_rdata[j];
        end else begin : g_mux_row
            wire [BLOCK_M-1:0][ENTRY_W-1:0] row_entries;
            for (genvar i = 0; i < BLOCK_M; ++i) begin : g_rows
                assign row_entries[i] = bank_rdata[(i << LG_BLOCK_N) + j];
            end
            assign sel_entry = row_entries[sel_row];
        end

        // ---- F1: the selected entry ----
        wire [ENTRY_W-1:0] f1;
        VX_pipe_register #(.DATAW (ENTRY_W)) pipe_f1 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in (sel_entry), .data_out (f1)
        );

        wire [W-1:0] f1_sum, f1_carry;
        wire         f1_sticky, f1_nan, f1_inf, f1_isgn;
        wire [`VX_CFG_XLEN-1:0] f1_c;
        assign {f1_nan, f1_inf, f1_isgn, f1_sticky, f1_sum, f1_carry, f1_c} = f1;

        // ---- F2: resolve carry-save, in sign-magnitude ----
        // Both polarities are computed in parallel so the stage holds one adder
        // delay, not two (the same trick VX_tcu_tfr_acc uses).
        wire [W-1:0] res_pos = f1_sum + f1_carry;
        wire [W-1:0] res_neg = ~f1_sum + ~f1_carry + W'(2);
        wire         res_sign = res_pos[W-1];
        wire [W-1:0] res_mag  = res_sign ? res_neg : res_pos;

        wire [W-1:0] f2_mag;
        wire         f2_sign, f2_sticky, f2_nan, f2_inf, f2_isgn;
        // f2_isgn is the element's stored infinity sign, not the value's sign.
        wire [`VX_CFG_XLEN-1:0] f2_c;
        VX_pipe_register #(.DATAW (W + 5 + `VX_CFG_XLEN)) pipe_f2 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in  ({res_mag, res_sign, f1_sticky, f1_nan, f1_inf, f1_isgn, f1_c}),
            .data_out ({f2_mag,  f2_sign,  f2_sticky, f2_nan, f2_inf, f2_isgn, f2_c})
        );

        // ---- F3: leading-zero count ----
        wire [$clog2(W)-1:0] lz;
        VX_lzc #(.N (W)) lzc_mag (
            .data_in    (f2_mag),
            .data_out   (lz),
            `UNUSED_PIN (valid_out)
        );

        wire [W-1:0] f3_mag;
        wire [$clog2(W)-1:0] f3_lz;
        wire f3_sign, f3_sticky, f3_nan, f3_inf, f3_isgn;
        wire [`VX_CFG_XLEN-1:0] f3_c;
        VX_pipe_register #(.DATAW (W + $clog2(W) + 5 + `VX_CFG_XLEN)) pipe_f3 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in  ({f2_mag, lz,    f2_sign, f2_sticky, f2_nan, f2_inf, f2_isgn, f2_c}),
            .data_out ({f3_mag, f3_lz, f3_sign, f3_sticky, f3_nan, f3_inf, f3_isgn, f3_c})
        );

        // ---- F4: normalize to a MAG_W-bit term, with its exponent ----
        wire [W-1:0] shifted = f3_mag << f3_lz;
        wire [MAG_W-1:0] term_mag = shifted[W-1 -: MAG_W];
        wire term_sticky = f3_sticky | (|shifted[W-MAG_W-1:0]);
        wire zero_mag = ~|f3_mag;
        // exponent of term_mag in the accumulator's domain; a zero sum carries
        // exponent 0 so it cannot win the fuse's max-exponent search.
        wire [EXP_W-1:0] term_exp = zero_mag ? '0 : (term_exp_base - EXP_W'(f3_lz));

        wire [MAG_W-1:0] f4_mag;
        wire [EXP_W-1:0] f4_exp;
        // Narrowed to FUSE_EXP_W at the fuse; both terms are fp32-scale there.
        `UNUSED_VAR (f4_exp)
        wire f4_sign, f4_sticky, f4_nan, f4_inf, f4_isgn;
        wire [`VX_CFG_XLEN-1:0] f4_c;
        VX_pipe_register #(.DATAW (MAG_W + EXP_W + 5 + `VX_CFG_XLEN)) pipe_f4 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in  ({term_mag, term_exp, f3_sign, term_sticky, f3_nan, f3_inf, f3_isgn, f3_c}),
            .data_out ({f4_mag,   f4_exp,   f4_sign, f4_sticky,   f4_nan, f4_inf, f4_isgn, f4_c})
        );

        // ---- F5: fuse alignment against the C term ----
        // VX_tcu_tfr_align reserves its LAST lane for the FEDP's own C operand
        // and pre-shifts it one bit less than the product lanes, so both of our
        // terms sit on product lanes and the reserved slot carries zero.
        tcu_op_term_t t_c;
        assign t_c = tcu_op_unpack_fp32(f4_c);

        wire [2:0][FUSE_EXP_W-1:0] fuse_exps = {FUSE_EXP_W'(0), FUSE_EXP_W'(t_c.exp), FUSE_EXP_W'(f4_exp)};
        wire [2:0][24:0]           fuse_sigs = {25'd0, {t_c.sign, t_c.mag}, {f4_sign, f4_mag}};
        wire [2:0]                 fuse_sel;

        VX_tcu_tfr_max_exp #(.N (3), .WIDTH (FUSE_EXP_W)) fuse_max_exp (
            .exponents (fuse_exps),
            .sel_exp   (fuse_sel)
        );

        wire [2:0][FUSE_W-1:0] aligned_w;
        wire [2:0]             astick_w;
        wire [2:0]             anegs_w;
        wire [FUSE_EXP_W-1:0]  amax_w;

        VX_tcu_tfr_align #(.N (3), .WI (25), .WO (FUSE_W)) fuse_align (
            .clk         (clk),
            .valid_in    (1'b0),
            .req_id      (32'd0),
            .exponents   (fuse_exps),
            .sel_exp     (fuse_sel),
            .lane_mask   (2'b11),
            .sigs_in     (fuse_sigs),
            .is_int      (1'b0),
            .max_exp     (amax_w),
            .sigs_out    (aligned_w),
            .sticky_bits (astick_w),
            .fp_negs     (anegs_w)
        );

        wire inf_clash_w = f4_inf && t_c.is_inf && (f4_isgn != t_c.sign);
        fedp_excep_t exc_w;
        assign exc_w.is_nan = f4_nan | t_c.is_nan | inf_clash_w;
        assign exc_w.is_inf = (f4_inf | t_c.is_inf) & ~inf_clash_w;
        assign exc_w.sign   = f4_inf ? f4_isgn : t_c.sign;

        wire [2:0][FUSE_W-1:0] f5_aligned;
        wire [2:0]             f5_astick;
        wire [2:0]             f5_anegs;
        wire [FUSE_EXP_W-1:0]  f5_max;
        wire                   f5_sticky;
        fedp_excep_t           f5_exc;
        VX_pipe_register #(.DATAW (3*FUSE_W + 3 + 3 + FUSE_EXP_W + 1 + $bits(fedp_excep_t))) pipe_f5 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in  ({aligned_w,   astick_w,  anegs_w,  amax_w, f4_sticky, exc_w}),
            .data_out ({f5_aligned,  f5_astick, f5_anegs, f5_max, f5_sticky, f5_exc})
        );

        // ---- F6: fuse addition ----
        wire [FUSE_W-1:0] fmag_w;
        wire              fsign_w, fstick_w;
        VX_tcu_tfr_acc #(.N (3), .WO (FUSE_W)) fuse_acc (
            .clk        (clk),
            .valid_in   (1'b0),
            .req_id     (32'd0),
            .sigs_in    (f5_aligned),
            .sticky_in  (f5_astick),
            .fp_negs    (f5_anegs),
            .sig_out    (fmag_w),
            .sign_out   (fsign_w),
            .sticky_out (fstick_w)
        );

        wire [FUSE_W-1:0] f6_mag;
        wire              f6_sign, f6_sticky;
        wire [FUSE_EXP_W-1:0] f6_max;
        fedp_excep_t      f6_exc;
        VX_pipe_register #(.DATAW (FUSE_W + 2 + FUSE_EXP_W + $bits(fedp_excep_t))) pipe_f6 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in  ({fmag_w, fsign_w, (fstick_w | f5_sticky), f5_max, f5_exc}),
            .data_out ({f6_mag, f6_sign, f6_sticky,              f6_max, f6_exc})
        );

        // ---- F7: normalize and round, once ----
        wire [`VX_CFG_XLEN-1:0] rounded;
        VX_tcu_tfr_norm_round #(
            .WA     (FUSE_W),
            .EXP_W  (FUSE_EXP_W),
            .C_HI_W (7)
        ) fuse_round (
            .clk        (clk),
            .valid_in   (1'b0),
            .req_id     (32'd0),
            .max_exp    (f6_max - FUSE_EXP_W'(FUSE_EXP_ADJ)),
            .acc_sig    (f6_mag),
            .acc_sign   (f6_sign),
            .sticky_in  (f6_sticky),
            .cval_hi    (7'd0),
            .is_int     (1'b0),
            .exceptions (f6_exc),
            .result     (rounded)
        );

        VX_pipe_register #(.DATAW (`VX_CFG_XLEN)) pipe_f7 (
            .clk (clk), .reset (reset), .enable (flush_enable),
            .data_in (rounded), .data_out (read_row_data[j])
        );
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DRAIN

    // xbar_busy covers items the network is holding internally: those are in
    // neither the input queues nor on xbar_valid_out, and the decomposed network
    // can hold one for several cycles when its second stage refuses it.
    assign accu_ready_to_flush = (&xbar_queue_empty) && (~|xbar_valid_out) && ~xbar_busy
                              && (~|a1_valid) && (~|ac_valid);

    // A flush reads the accumulator while nothing may be in flight to it.
    `RUNTIME_ASSERT(~(enable && read_en && ((|xbar_enable_out) || xbar_busy || (|a1_valid) || (|ac_valid))),
        ("%t: *** tcu_op_accu: flush read while products are still in flight", $time))

`ifdef SIMULATION
    // Scoreboard: products accepted at the write port but not yet written.
    integer dbg_pending;
    always @(posedge clk) begin
        if (reset) begin
            dbg_pending <= 0;
        end else begin
            dbg_pending <= dbg_pending + $countones(write_lane_fire) - $countones(bank_write_en);
        end
    end
    `RUNTIME_ASSERT(~(accu_ready_to_flush && (dbg_pending != 0)),
        ("%t: *** tcu_op_accu: reports drained with %0d products pending", $time, dbg_pending))
`endif

endmodule

`endif // TCU_OP
