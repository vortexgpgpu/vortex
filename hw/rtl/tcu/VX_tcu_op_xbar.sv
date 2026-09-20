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
// TCU_OP product routing network (D4).
//
// The accumulator needs BANKS = BLOCK_M * BLOCK_N products delivered to BANKS
// banks, and a full crossbar of that size is the wrong shape for it. Input k is
// lane (row = k >> LG_N, col = k & (BLOCK_N-1)) of the step, and the bank it
// wants is
//
//     bank = (row_of_lane % BLOCK_M) * BLOCK_N + (col_of_lane % BLOCK_N)
//
// so the destination splits into a high field (which group of BLOCK_N banks) and
// a low field (which position inside that group). Route the two fields in two
// stages -- a radix-(BLOCK_N, BLOCK_M) decomposition of the crossbar:
//
//   stage 1: BLOCK_M crossbars of BLOCK_N -> BLOCK_N, one per input row, each
//            placing its items at position bank[LG_N-1:0];
//   stage 2: BLOCK_N crossbars of BLOCK_M -> BLOCK_M, one per position, each
//            routing that position's items to bank[BANK_BW-1:LG_N].
//
// For the shipped 2 x 16 geometry that is 2 x (16->16) plus 16 x (2->2) in place
// of 1 x (32->32): a bit over half the mux area, and -- the part that matters at
// 250 MHz -- 16-way arbiters instead of 32-way, with each stage-1 crossbar
// spanning half the payload bits so it can be placed as a compact region.
//
// WHY THIS IS UNCONDITIONALLY CORRECT. Note what the argument does NOT rest on.
// It does not claim the group is constant per input, nor the position: both are
// data, and with per-input queues a queued item carries the destination that was
// computed when it was pushed, so two items sitting in the same row's queues can
// disagree about either field. That is exactly the trap an earlier version of
// this module fell into by hoisting `row_group` and `col_pos` out as per-step
// signals. Here each item is routed by the fields of *its own* stored bank: it
// is placed at position bank[LG_N-1:0] of whichever stage-1 crossbar its input is
// hardwired to, then sent to group bank[BANK_BW-1:LG_N]. Every input can reach
// every bank, whatever the traffic.
//
// The decomposition IS blocking, which a full crossbar is not: two items in the
// same input row with equal low fields and different high fields must serialize
// here, where a full crossbar would pass both. Dense steps never hit it -- their
// BLOCK_N columns are consecutive, hence distinct mod BLOCK_N -- and the input
// queues absorb the sparse cases.
//------------------------------------------------------------------------------
`include "VX_define.vh"

`ifdef TCU_OP

// No package import: PERF_CTR_BITS below would shadow the package constant of
// the same name, and nothing else here needs VX_gpu_pkg.
module VX_tcu_op_xbar #(
    parameter int BLOCK_M       = 2,
    parameter int BLOCK_N       = 16,
    parameter int DATAW         = 45,
    parameter     ARBITER       = "R",
    parameter int PERF_CTR_BITS = `CLOG2(BLOCK_M*BLOCK_N+1),
    parameter int SEL_W         = `LOG2UP(BLOCK_M*BLOCK_N)
) (
    input wire clk,
    input wire reset,

    input  wire [BLOCK_M*BLOCK_N-1:0]              valid_in,
    input  wire [BLOCK_M*BLOCK_N-1:0][DATAW-1:0]   data_in,
    input  wire [BLOCK_M*BLOCK_N-1:0][SEL_W-1:0]   sel_in,
    output wire [BLOCK_M*BLOCK_N-1:0]              ready_in,

    output wire [BLOCK_M*BLOCK_N-1:0]              valid_out,
    output wire [BLOCK_M*BLOCK_N-1:0][DATAW-1:0]   data_out,
    input  wire [BLOCK_M*BLOCK_N-1:0]              ready_out,

    // High while any item is held inside the network. Stage 1 buffers its
    // outputs internally (VX_stream_arb gives its fanout slices OUT_BUF(3)), and
    // stage 2 can refuse one for several cycles, so such an item is in neither
    // the caller's input queues nor on valid_out. A drain check that watched only
    // those two would let a flush start on top of products still in flight --
    // an exposure the undecomposed crossbar did not have, because its outputs
    // were unconditionally ready and anything buffered left the next cycle.
    output wire                                    busy,

    output wire [PERF_CTR_BITS-1:0]                collisions
);
    localparam int BANKS = BLOCK_M * BLOCK_N;
    localparam int LG_M  = `LOG2UP(BLOCK_M);
    localparam int LG_N  = `LOG2UP(BLOCK_N);

    `STATIC_ASSERT (BLOCK_M == (1 << `CLOG2(BLOCK_M)),
        ("tcu_op_xbar: BLOCK_M must be a power of two, got %0d", BLOCK_M))
    `STATIC_ASSERT (BLOCK_N == (1 << `CLOG2(BLOCK_N)),
        ("tcu_op_xbar: BLOCK_N must be a power of two, got %0d", BLOCK_N))
    `STATIC_ASSERT (SEL_W >= `CLOG2(BANKS),
        ("tcu_op_xbar: sel_in too narrow for %0d banks", BANKS))

    // The group field rides through stage 1 with the payload, so stage 2 can
    // route each item by its own destination rather than a shared per-step one.
`ifdef SIMULATION
    // In simulation each item also carries the bank it was asked for, so the
    // output side can assert it actually arrived there.
    localparam int S1_DATAW = DATAW + LG_M + SEL_W;
    localparam int S2_DATAW = DATAW + SEL_W;
`else
    localparam int S1_DATAW = DATAW + LG_M;
    localparam int S2_DATAW = DATAW;
`endif

    wire [BLOCK_M-1:0][BLOCK_N-1:0]                s1_valid;
    wire [BLOCK_M-1:0][BLOCK_N-1:0][S1_DATAW-1:0]  s1_data;
    wire [BLOCK_M-1:0][BLOCK_N-1:0]                s1_ready;
    wire [BLOCK_M-1:0][BLOCK_N-1:0]                s1_ready_out;

    wire [BLOCK_M-1:0][PERF_CTR_BITS-1:0] s1_collisions;
    wire [BLOCK_N-1:0][PERF_CTR_BITS-1:0] s2_collisions;

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// STAGE 1: position within the group (bank[LG_N-1:0]), per input row

    for (genvar g = 0; g < BLOCK_M; ++g) begin : g_row
        wire [BLOCK_N-1:0]                 rv;
        wire [BLOCK_N-1:0][S1_DATAW-1:0]   rd;
        wire [BLOCK_N-1:0][LG_N-1:0]       rs;

        for (genvar j = 0; j < BLOCK_N; ++j) begin : g_in
            localparam int K = (g * BLOCK_N) + j;
            // With a single bank group the high field does not exist in sel_in,
            // so it must not be sliced -- LG_M is 1 even then (LOG2UP), and the
            // slice would run off the end of sel_in.
            wire [LG_M-1:0] grp;
            if (BLOCK_M == 1) begin : g_one_group
                assign grp = '0;
            end else begin : g_many_groups
                assign grp = sel_in[K][LG_N +: LG_M];
            end
            assign rv[j] = valid_in[K];
        `ifdef SIMULATION
            assign rd[j] = {sel_in[K], grp, data_in[K]};
        `else
            assign rd[j] = {grp, data_in[K]};
        `endif
            assign rs[j] = sel_in[K][LG_N-1:0];
            assign ready_in[K] = s1_ready[g][j];
        end

        VX_stream_xbar #(
            .NUM_INPUTS    (BLOCK_N),
            .NUM_OUTPUTS   (BLOCK_N),
            .DATAW         (S1_DATAW),
            .ARBITER       (ARBITER),
            .OUT_BUF       (0),
            .PERF_CTR_BITS (PERF_CTR_BITS)
        ) row_xbar (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (rv),
            .data_in    (rd),
            .sel_in     (rs),
            .ready_in   (s1_ready[g]),
            .valid_out  (s1_valid[g]),
            .data_out   (s1_data[g]),
            `UNUSED_PIN (sel_out),
            .ready_out  (s1_ready_out[g]),
            .collisions (s1_collisions[g])
        );
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// STAGE 2: group (bank[BANK_BW-1:LG_N]), per position

    for (genvar p = 0; p < BLOCK_N; ++p) begin : g_pos
        wire [BLOCK_M-1:0]                pv;
        wire [BLOCK_M-1:0][S2_DATAW-1:0]  pd;
        wire [BLOCK_M-1:0][LG_M-1:0]      ps;
        wire [BLOCK_M-1:0]                pr;

        for (genvar g = 0; g < BLOCK_M; ++g) begin : g_from_row
        `ifdef SIMULATION
            wire [SEL_W-1:0] want;
            assign {want, ps[g]} = s1_data[g][p][S1_DATAW-1:DATAW];
            assign pd[g] = {want, s1_data[g][p][DATAW-1:0]};
            // Stage 1 must have placed this item at the position its bank names.
            `RUNTIME_ASSERT (~pv[g] || (want[LG_N-1:0] == LG_N'(p)),
                ("%t: *** tcu_op_xbar: stage 1 row %0d placed bank %0d at position %0d",
                 $time, g, want, p))
        `else
            assign {ps[g], pd[g]} = s1_data[g][p];
        `endif
            assign pv[g] = s1_valid[g][p];
            assign s1_ready_out[g][p] = pr[g];
        end

        wire [BLOCK_M-1:0]                qv;
        wire [BLOCK_M-1:0][S2_DATAW-1:0]  qd;
        wire [BLOCK_M-1:0]                qr;

        for (genvar g = 0; g < BLOCK_M; ++g) begin : g_to_bank
            localparam int K = (g * BLOCK_N) + p;
            assign valid_out[K] = qv[g];
            assign data_out[K]  = qd[g][DATAW-1:0];
            assign qr[g]        = ready_out[K];
        `ifdef SIMULATION
            // End to end: the item leaving at bank K must be the one that asked
            // for bank K.
            `RUNTIME_ASSERT (~qv[g] || (qd[g][S2_DATAW-1:DATAW] == SEL_W'(K)),
                ("%t: *** tcu_op_xbar: bank %0d received an item bound for bank %0d",
                 $time, K, qd[g][S2_DATAW-1:DATAW]))
        `endif
        end

        VX_stream_xbar #(
            .NUM_INPUTS    (BLOCK_M),
            .NUM_OUTPUTS   (BLOCK_M),
            .DATAW         (S2_DATAW),
            .ARBITER       (ARBITER),
            .OUT_BUF       (0),
            .PERF_CTR_BITS (PERF_CTR_BITS)
        ) pos_xbar (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (pv),
            .data_in    (pd),
            .sel_in     (ps),
            .ready_in   (pr),
            .valid_out  (qv),
            .data_out   (qd),
            `UNUSED_PIN (sel_out),
            .ready_out  (qr),
            .collisions (s2_collisions[p])
        );
    end

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// DRAIN

    // Every item stage 1 holds shows up here, including one stage 2 is accepting
    // this cycle, so this is conservative by at most a cycle.
    assign busy = |s1_valid;

// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
// PERF

    // Collisions from both stages, saturating: the counter is advisory, and a
    // decomposed network can report a stage-1 and a stage-2 collision for the
    // same cycle where a full crossbar would have reported one.
    wire [PERF_CTR_BITS:0] coll_sum;
    VX_reduce_tree #(
        .IN_W  (PERF_CTR_BITS),
        .OUT_W (PERF_CTR_BITS+1),
        .N     (BLOCK_M + BLOCK_N),
        .OP    ("+")
    ) coll_reduce (
        .data_in  ({s1_collisions, s2_collisions}),
        .data_out (coll_sum)
    );
    assign collisions = coll_sum[PERF_CTR_BITS] ? '1 : coll_sum[PERF_CTR_BITS-1:0];

endmodule

`endif // TCU_OP
