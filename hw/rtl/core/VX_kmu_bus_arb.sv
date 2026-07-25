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

// VX_kmu_bus_arb — merges launch masters onto the device launch stream and fans
// it back out across the hierarchy (device -> clusters -> sockets -> cores).
//
// The unit of arbitration is a MESSAGE, not a beat (see VX_kmu_bus_if). Both
// directions therefore LOCK an input->output pair at the header and hold it to
// `eop`, so a multi-beat fragment launch cannot be interleaved with another
// master's header, nor sprayed across two cores.
//
// The lock keys on `eop`, deliberately NOT on `valid` falling. A stream arbiter's
// usual "sticky until the request drops" rule is wrong on this bus: VX_kmu holds
// `valid` high continuously while it has CTAs left to issue, so a valid-keyed
// lock would never release and would starve every other master forever.
//
// Fan-out picks the output by kind:
//   FRAGMENT — routed by `dest`, this level's slice (DEST_LSB). Fragment work is
//              bin->core affine; load-balancing it would break same-pixel blend
//              order.
//   COMPUTE  — round-robin over the ready outputs, which is what the plain
//              stream arbiter did before and what keeps CTAs spread across cores.

module VX_kmu_bus_arb import VX_gpu_pkg::*; #(
    parameter NUM_INPUTS  = 1,
    parameter NUM_OUTPUTS = 1,
    parameter DEST_LSB    = 0,   // bit offset of this level's slice of `dest`
    parameter IN_BUF      = 0,
    parameter OUT_BUF     = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    VX_kmu_bus_if.slave  bus_in_if [NUM_INPUTS],
    VX_kmu_bus_if.master bus_out_if [NUM_OUTPUTS],

    // High while any beat is still inside this arb. A descriptor buffered here has
    // already left its producer but has not reached a consumer, so it is invisible
    // to both ends: without this the launch can be reported complete while CTAs are
    // still queued.
    output wire             busy
);
    // packed beat: {kind, eop, dest, data}
    localparam PW    = 1 + 1 + KMU_DEST_W + KMU_DATAW;
    localparam LOG_I = `CLOG2(NUM_INPUTS);
    localparam LOG_O = `CLOG2(NUM_OUTPUTS);
    localparam SEL_I = `UP(LOG_I);
    localparam SEL_O = `UP(LOG_O);

    // field offsets within a packed beat
    localparam PW_DEST = KMU_DATAW;
    localparam PW_EOP  = KMU_DATAW + KMU_DEST_W;
    localparam PW_KIND = KMU_DATAW + KMU_DEST_W + 1;

    wire [NUM_INPUTS-1:0]           bus_valid;
    wire [NUM_INPUTS-1:0][PW-1:0]   bus_data;
    wire [NUM_INPUTS-1:0]           bus_ready;

    wire [NUM_INPUTS-1:0]           in_valid;
    wire [NUM_INPUTS-1:0][PW-1:0]   in_data;
    wire [NUM_INPUTS-1:0]           in_ready;

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_in
        assign bus_valid[i] = bus_in_if[i].valid;
        assign bus_data[i]  = {bus_in_if[i].kind,
                               bus_in_if[i].eop,
                               bus_in_if[i].dest,
                               bus_in_if[i].data};
        assign bus_in_if[i].ready = bus_ready[i];
    end

    // Register the incoming request handshake so the ready driven back to the
    // producer is a registered signal. An arbitrated fan-out's input ready is the
    // arbiter grant, which is combinational no matter how the OUTPUT is buffered,
    // so an output skid cannot fix it -- buffering here is what keeps back-pressure
    // from crossing the whole distribution tree in one path. IN_BUF = 0 is a
    // passthrough, so single-input instances are unchanged.
    //
    // Beat order per input is preserved, so the eop-keyed message lock below still
    // sees each master's header and eop in order.
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_in_buf
        VX_elastic_buffer #(
            .DATAW   (PW),
            .SIZE    (`TO_OUT_BUF_SIZE(IN_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(IN_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(IN_BUF))
        ) in_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (bus_valid[i]),
            .ready_in  (bus_ready[i]),
            .data_in   (bus_data[i]),
            .data_out  (in_data[i]),
            .valid_out (in_valid[i]),
            .ready_out (in_ready[i])
        );
    end

    // ── fan-in: NUM_INPUTS masters -> one merged message stream ────────────
    wire            m_valid;
    wire [PW-1:0]   m_data;
    wire            m_ready;

    if (NUM_INPUTS > 1) begin : g_fanin

        // Message lock: once a header has been granted, only that input may
        // drive the merged stream until its `eop` goes through.
        reg             lock_v;
        reg [SEL_I-1:0] lock_sel;

        wire [NUM_INPUTS-1:0] lock_mask =
            lock_v ? (NUM_INPUTS'(1) << lock_sel) : {NUM_INPUTS{1'b1}};

        wire [NUM_INPUTS-1:0]         arb_valid_in = in_valid & lock_mask;
        wire [NUM_INPUTS-1:0]         arb_ready_in;
        wire [0:0][SEL_I-1:0]         arb_sel_out;

        // Ready must be masked too: an ungranted input whose valid we gated off
        // would otherwise see ready high and believe its beat was taken.
        assign in_ready = arb_ready_in & lock_mask;

        VX_stream_arb #(
            .NUM_INPUTS  (NUM_INPUTS),
            .NUM_OUTPUTS (1),
            .DATAW       (PW),
            .ARBITER     (ARBITER),
            .OUT_BUF     (0)
        ) fanin_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_in),
            .ready_in  (arb_ready_in),
            .data_in   (in_data),
            .data_out  (m_data),
            .sel_out   (arb_sel_out),
            .valid_out (m_valid),
            .ready_out (m_ready)
        );

        wire m_fire = m_valid && m_ready;
        always @(posedge clk) begin
            if (reset) begin
                lock_v   <= 1'b0;
                lock_sel <= '0;
            end else if (m_fire) begin
                lock_v   <= ~m_data[PW_EOP];
                lock_sel <= arb_sel_out[0];
            end
        end

    end else begin : g_fanin_1

        assign m_valid    = in_valid[0];
        assign m_data     = in_data[0];
        assign in_ready[0] = m_ready;

    end

    // ── fan-out: one merged stream -> NUM_OUTPUTS ──────────────────────────
    wire [NUM_OUTPUTS-1:0]         out_valid;
    wire [NUM_OUTPUTS-1:0][PW-1:0] out_data;
    wire [NUM_OUTPUTS-1:0]         out_ready;

    if (NUM_OUTPUTS > 1) begin : g_fanout

        reg             lock_v;
        reg [SEL_O-1:0] lock_sel;
        reg [SEL_O-1:0] rr_ptr;

        wire [SEL_O-1:0] dest_sel = SEL_O'(m_data[PW_DEST +: KMU_DEST_W] >> DEST_LSB);

        // Round-robin over the ready outputs, starting at rr_ptr. Descending
        // scan so the nearest ready output (smallest step from rr_ptr) is the
        // last assignment and therefore wins.
        reg [SEL_O-1:0] rr_sel;
        always @(*) begin
            rr_sel = rr_ptr;
            for (integer k = NUM_OUTPUTS - 1; k >= 0; --k) begin
                automatic logic [SEL_O-1:0] idx =
                    SEL_O'((32'(rr_ptr) + k) % NUM_OUTPUTS);
                if (out_ready[idx]) begin
                    rr_sel = idx;
                end
            end
        end

        wire [SEL_O-1:0] sel_new = (m_data[PW_KIND] == KMU_KIND_FRAGMENT)
                                 ? dest_sel : rr_sel;
        wire [SEL_O-1:0] sel = lock_v ? lock_sel : sel_new;

        for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_sel
            assign out_valid[o] = m_valid && (sel == SEL_O'(o));
            assign out_data[o]  = m_data;
        end
        assign m_ready = out_ready[sel];

        wire m_fire = m_valid && m_ready;
        wire [SEL_O-1:0] rr_next = (sel == SEL_O'(NUM_OUTPUTS - 1)) ? '0 : (sel + SEL_O'(1));

        always @(posedge clk) begin
            if (reset) begin
                lock_v   <= 1'b0;
                lock_sel <= '0;
                rr_ptr   <= '0;
            end else if (m_fire) begin
                lock_v   <= ~m_data[PW_EOP];
                lock_sel <= sel;
                if (m_data[PW_EOP]) begin
                    rr_ptr <= rr_next;
                end
            end
        end

    end else begin : g_fanout_1

        assign out_valid[0] = m_valid;
        assign out_data[0]  = m_data;
        assign m_ready      = out_ready[0];

    end

    // ── output buffering ───────────────────────────────────────────────────
    // A per-output elastic buffer is FIFO, so it preserves the beat order of a
    // locked message.
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out
        wire [PW-1:0] buf_data;

        VX_elastic_buffer #(
            .DATAW   (PW),
            .SIZE    (`TO_OUT_BUF_SIZE(OUT_BUF)),
            .OUT_REG (`TO_OUT_BUF_REG(OUT_BUF)),
            .LUTRAM  (`TO_OUT_BUF_LUTRAM(OUT_BUF))
        ) out_buf (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (out_valid[o]),
            .ready_in  (out_ready[o]),
            .data_in   (out_data[o]),
            .data_out  (buf_data),
            .valid_out (bus_out_if[o].valid),
            .ready_out (bus_out_if[o].ready)
        );

        assign bus_out_if[o].kind = buf_data[PW_KIND];
        assign bus_out_if[o].eop  = buf_data[PW_EOP];
        assign bus_out_if[o].dest = buf_data[PW_DEST +: KMU_DEST_W];
        assign bus_out_if[o].data = buf_data[KMU_DATAW-1:0];
    end

    // ── in-flight accounting ───────────────────────────────────────────────
    // Every beat that enters leaves exactly once, so a single up/down counter
    // covers all internal storage (input skid, output skid, and the beat the
    // message lock is mid-way through).
    localparam CNT_W = `CLOG2(NUM_INPUTS * `TO_OUT_BUF_SIZE(IN_BUF)
                            + NUM_OUTPUTS * `TO_OUT_BUF_SIZE(OUT_BUF)
                            + NUM_INPUTS + NUM_OUTPUTS + 1) + 1;

    wire [NUM_INPUTS-1:0]  in_fire;
    wire [NUM_OUTPUTS-1:0] out_fire;
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin : g_in_fire
        assign in_fire[i] = bus_in_if[i].valid && bus_in_if[i].ready;
    end
    for (genvar o = 0; o < NUM_OUTPUTS; ++o) begin : g_out_fire
        assign out_fire[o] = bus_out_if[o].valid && bus_out_if[o].ready;
    end

    reg [CNT_W-1:0] inflight_r;
    always @(posedge clk) begin
        if (reset) begin
            inflight_r <= '0;
        end else begin
            inflight_r <= inflight_r + CNT_W'($countones(in_fire))
                                     - CNT_W'($countones(out_fire));
        end
    end

    // `inflight_r` is registered, so it does not yet count a beat presented this
    // cycle. Report busy from the cycle a beat is seen: on the final CTA of a
    // launch the KMU's own busy drops the cycle after the beat fires, and where
    // the aggregation above is registered (NUM_SOCKETS > 1) nothing else covers
    // that cycle -- opening a one-cycle device-idle gap the host's edge-sensitive
    // idle-wait latches as premature completion. `bus_valid` is used rather than
    // in_fire because the input `ready` is the arbiter grant, and routing it here
    // would pull the downstream back-pressure chain into the busy tree.
    assign busy = (inflight_r != '0) || (| bus_valid);

endmodule
