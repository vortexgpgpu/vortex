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

`include "VX_raster_define.vh"

// VX_raster_arb — cluster / socket raster bus arbiter
//
// Routes {stamps, done} packets from N producers to M consumers, plus
// fans `begin_pulse` upstream (consumer → producer, OR-reduced).
//
// Three direction cases:
//   1. N == M: 1:1 pairing — VX_stream_arb handles.
//   2. N >  M: fan-in / merge — VX_stream_arb handles.
//   3. N <  M: fan-out — custom round-robin. The underlying VX_stream_arb
//      only fans valid_in[0] to output[0] and leaves outputs 1..M-1 dead
//      in this direction, so we bypass it.
//
// Additionally:
//   - Per-output sticky-done state (`consumer_served[o]`) so a single
//     drain sentinel served to consumer o stays "done" for any
//     subsequent vx_rast() from any warp on that consumer.
//   - Per-output activity tracking (`consumer_was_active[o]`) so
//     never-asking consumers don't gate `frame_drained`.
//   - Gated flush on the first `begin_pulse_any` after `frame_drained=1`
//     — clears arb state and resets every per-output OUT_BUF, eliminating
//     stale `{done=1}` packets from the previous frame.
//   - Producer-side req_valid gating: a single instance that finished
//     while peers are still producing is suppressed (its `{stamps=0,
//     done=1}` won't leak to consumers as `{stamps=0, done_all=0}`).

module VX_raster_arb import VX_raster_pkg::*; #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter OUT_BUF        = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_raster_bus_if.slave  bus_in_if  [NUM_INPUTS],
    VX_raster_bus_if.master bus_out_if [NUM_OUTPUTS]
);
    localparam REQ_DATAW   = NUM_LANES * $bits(raster_stamp_t) + 1;
    localparam IS_FANOUT   = (NUM_INPUTS < NUM_OUTPUTS);
    localparam LOG_OUTPUTS = `LOG2UP(NUM_OUTPUTS);

    // ── begin_pulse fan-in (combinational OR-reduce, master → slave) ──
    wire [NUM_OUTPUTS-1:0] begin_pulse_in;
    for (genvar j = 0; j < NUM_OUTPUTS; ++j) begin : g_begin_pulse_in
        assign begin_pulse_in[j] = bus_out_if[j].begin_pulse;
    end
    wire begin_pulse_any = (| begin_pulse_in);
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_begin_pulse_out
        assign bus_in_if[i].begin_pulse = begin_pulse_any;
    end

    // ── done_all aggregation ──────────────────────────────────────────
    wire [NUM_INPUTS-1:0] done_mask;
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_done_mask
        assign done_mask[i] = bus_in_if[i].req_data.done;
    end
    wire done_all = (& done_mask);

    // ── Per-output sticky-done + activity state ───────────────────────
    reg [NUM_OUTPUTS-1:0] consumer_served;
    reg [NUM_OUTPUTS-1:0] consumer_was_active;
    reg                   frame_drained;

    wire [NUM_OUTPUTS-1:0] out_handshake;       // bus_out[o].req_valid && req_ready
    wire [NUM_OUTPUTS-1:0] out_done_handshake;  // handshake with done=1

    // Drained only when AT LEAST ONE consumer became active AND every active
    // consumer was served. Without the `|any_active` guard, frame_drained
    // latches to 1 on cycle 1 (consumer_was_active==0 ⇒ ~consumer_was_active is
    // all-ones ⇒ all_active_served=1), causing each warp's begin_pulse during
    // frame start-up to fire a spurious flush mid-frame, dropping in-flight quads.
    wire any_active        = (consumer_was_active != '0);
    wire all_active_served = any_active && (&(consumer_served | ~consumer_was_active));
    wire flush_trigger     = begin_pulse_any && frame_drained;

    always @(posedge clk) begin
        if (reset || flush_trigger) begin
            consumer_served     <= '0;
            consumer_was_active <= '0;
            frame_drained       <= 1'b0;
        end else begin
            for (int o = 0; o < NUM_OUTPUTS; ++o) begin
                if (out_handshake[o]) begin
                    consumer_was_active[o] <= 1'b1;
                    if (out_done_handshake[o])
                        consumer_served[o] <= 1'b1;
                end
            end
            if (all_active_served && !frame_drained)
                frame_drained <= 1'b1;
        end
    end

    // ── Per-input filtering: only forward inputs that have real quads
    //    OR all producers are done. Suppresses a single instance's
    //    early {stamps=0, done=1} from leaking with done_all=0. ──────
    wire [NUM_INPUTS-1:0]                 req_valid_filtered;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_filtered;
    wire [NUM_INPUTS-1:0]                 req_ready_filtered;

    // The done-gate is needed ONLY when the consumer side merges multiple
    // producers (fan-in N>M, or fan-out N<M which routes a single merged
    // stream to many outputs). For strict 1:1 pairing (N==M) every consumer
    // is owned by exactly one producer, so an early `done=1` is the truth
    // for that consumer — gating it on the slower peers' done deadlocks the
    // already-finished pair (e.g. 4 raster_cores × 4 sockets / NUM_RASTER_CORES=4).
    localparam GATE_DONE = (NUM_INPUTS != NUM_OUTPUTS);
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_filter
        if (GATE_DONE) begin : g_gated
            assign req_valid_filtered[i] = bus_in_if[i].req_valid
                                        && (~bus_in_if[i].req_data.done || done_all);
            assign req_data_filtered[i]  = {bus_in_if[i].req_data.stamps, done_all};
        end else begin : g_passthru
            assign req_valid_filtered[i] = bus_in_if[i].req_valid;
            assign req_data_filtered[i]  = {bus_in_if[i].req_data.stamps,
                                            bus_in_if[i].req_data.done};
        end
        assign bus_in_if[i].req_ready = req_ready_filtered[i];
    end
    if (!GATE_DONE) begin : g_dead_done_all
        `UNUSED_VAR (done_all)
    end

    // ── Routing engine ────────────────────────────────────────────────
    wire [NUM_OUTPUTS-1:0]                arb_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] arb_data_out;
    wire [NUM_OUTPUTS-1:0]                arb_ready_out;

    if (!IS_FANOUT) begin : g_fanin_or_pass
        // N >= M: use the existing stream_arb. It handles fan-in (N>M)
        // and 1:1 pairing (N==M) correctly.
        VX_stream_arb #(
            .NUM_INPUTS (NUM_INPUTS),
            .NUM_OUTPUTS(NUM_OUTPUTS),
            .DATAW      (REQ_DATAW),
            .ARBITER    (ARBITER),
            .OUT_BUF    (0)              // OUT_BUF lives outside (flushable)
        ) req_arb (
            .clk        (clk),
            .reset      (reset | flush_trigger),
            .valid_in   (req_valid_filtered),
            .ready_in   (req_ready_filtered),
            .data_in    (req_data_filtered),
            .data_out   (arb_data_out),
            .valid_out  (arb_valid_out),
            .ready_out  (arb_ready_out),
            `UNUSED_PIN (sel_out)
        );
    end else begin : g_fanout
        // N < M: fan-out. Two-stage design:
        //   Stage 1 (input-side, N→1): VX_stream_arb collapses N producers
        //   into a single merged stream. Handles input-side arbitration
        //   correctly (including ARBITER policy + ready/valid handshake)
        //   AND skips inputs whose data is currently suppressed by the
        //   done-filter, so the merged stream never stalls on a single
        //   drained input while peers still have quads.
        //
        //   Stage 2 (output-side, 1→M): round-robin distribute the merged
        //   stream across M outputs. rr_output advances on handshake,
        //   guaranteeing every output is visited (one packet per cycle).

        wire                 merged_valid;
        wire [REQ_DATAW-1:0] merged_data;
        wire                 merged_ready;

        VX_stream_arb #(
            .NUM_INPUTS (NUM_INPUTS),
            .NUM_OUTPUTS(1),
            .DATAW      (REQ_DATAW),
            .ARBITER    (ARBITER),
            .OUT_BUF    (0)
        ) collapse_arb (
            .clk        (clk),
            .reset      (reset | flush_trigger),
            .valid_in   (req_valid_filtered),
            .ready_in   (req_ready_filtered),
            .data_in    (req_data_filtered),
            .data_out   (merged_data),
            .valid_out  (merged_valid),
            .ready_out  (merged_ready),
            `UNUSED_PIN (sel_out)
        );

        reg [LOG_OUTPUTS-1:0] rr_output;

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_arb_out_fanout
            assign arb_valid_out[o] = (rr_output == LOG_OUTPUTS'(o)) && merged_valid;
            assign arb_data_out[o]  = merged_data;
        end

        assign merged_ready = arb_ready_out[rr_output];

        wire any_handshake = arb_valid_out[rr_output] && arb_ready_out[rr_output];
        always @(posedge clk) begin
            if (reset || flush_trigger) begin
                rr_output <= '0;
            end else if (any_handshake) begin
                if (rr_output == LOG_OUTPUTS'(NUM_OUTPUTS - 1))
                    rr_output <= '0;
                else
                    rr_output <= rr_output + LOG_OUTPUTS'(1);
            end
        end
    end

    // ── Per-output mux: sticky-done overlays the routing engine ──────
    //
    // When `consumer_served[o]=1`, output o emits `{stamps=0, done=1}`
    // synthetically (independent of producer activity). This makes
    // every subsequent vx_rast from any warp on consumer o exit cleanly.
    //
    // Otherwise, output o forwards whatever the routing engine produced.
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_bus_out
        wire emit_sticky = consumer_served[o];
        wire [REQ_DATAW-1:0] sticky_data;
        assign sticky_data[0] = 1'b1;                              // done=1
        assign sticky_data[REQ_DATAW-1:1] = {(REQ_DATAW-1){1'b0}}; // stamps=0

        wire mux_valid = emit_sticky | arb_valid_out[o];
        wire [REQ_DATAW-1:0] mux_data = emit_sticky ? sticky_data : arb_data_out[o];

        wire buf_valid_out;
        wire [REQ_DATAW-1:0] buf_data_out;
        wire buf_ready_in;

        VX_elastic_buffer #(
            .DATAW  (REQ_DATAW),
            .SIZE   (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG(`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk      (clk),
            .reset    (reset | flush_trigger),
            .valid_in (mux_valid),
            .data_in  (mux_data),
            .ready_in (buf_ready_in),
            .valid_out(buf_valid_out),
            .data_out (buf_data_out),
            .ready_out(bus_out_if[o].req_ready)
        );

        // Arb path only "pushes" when we're NOT in sticky mode AND the
        // buffer can accept. Sticky packets are pulled by emit_sticky
        // path alone; arb's output is ignored in that mode.
        assign arb_ready_out[o] = buf_ready_in & ~emit_sticky;

        assign bus_out_if[o].req_valid = buf_valid_out;
        assign {bus_out_if[o].req_data.stamps, bus_out_if[o].req_data.done} = buf_data_out;

        assign out_handshake[o]      = bus_out_if[o].req_valid && bus_out_if[o].req_ready;
        assign out_done_handshake[o] = out_handshake[o] && bus_out_if[o].req_data.done;
    end

endmodule
