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

// VX_raster_packer — fragment warp aggregator (Fragment Dispatch v2, occupancy lever).
//
// The rasterizer emits waves of NUM_LANES quads, one quad per warp lane, all from
// a single primitive. A small triangle covers only a few quads, so most lanes of a
// wave carry mask=0 (idle) — the warp runs under-occupied. This unit compacts the
// COVERED quads (mask!=0) across successive waves (and across primitives) into a
// full NUM_LANES-lane packed wave, so the downstream launcher issues one CTA per
// FULL warp instead of one per sparse wave.
//
// Order invariant: quads are consumed in arrival order (= per-bin submission order,
// established by VX_raster_bus_arb owner routing), and two quads covering the same
// (pos_x,pos_y) are never co-packed into one warp — the buffer flushes first so
// same-pixel fragments land in distinct, sequentially-launched warps. This
// preserves the pre-packer guarantee that a warp's quads address distinct pixels
// (what the OM unit's same-pixel RMW ordering relies on).
//
// Drain: the raster bus is a pure stream with no end-of-frame token, so a partial
// buffer flushes on an input-idle timeout; `busy` stays asserted while any quad is
// buffered so the frame cannot be observed complete before the tail warp launches.

`include "VX_raster_define.vh"

module VX_raster_packer import VX_gpu_pkg::*, VX_raster_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // sparse covered-quad waves in (from the raster arb)
    VX_raster_bus_if.slave  in_bus_if,
    // packed covered-quad waves out (to the per-core dispatcher)
    VX_raster_bus_if.master out_bus_if,

    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam CNT_W  = `CLOG2(NUM_LANES + 1);          // 0..NUM_LANES
    localparam QI_W   = `CLOG2(NUM_LANES);              // quad scan index
    localparam IDLE_FLUSH_CYCLES = 32;                  // tail-flush latency
    localparam IDLE_W = `CLOG2(IDLE_FLUSH_CYCLES + 1);

    // ── fill buffer + scan state ──────────────────────────────────────────
    raster_stamp_t [NUM_LANES-1:0] buf_r;               // packed covered quads
    reg [CNT_W-1:0]                count_r;             // filled slot count
    raster_stamp_t [NUM_LANES-1:0] wave_r;              // latched input wave
    reg                            wave_valid_r;        // holding an input wave
    reg [QI_W-1:0]                 qi_r;                // scan index into wave_r
    reg                            flush_r;             // emitting the buffer
    reg [IDLE_W-1:0]               idle_r;              // input-idle counter

    // current quad under the scan cursor
    wire [$bits(raster_stamp_t)-1:0] cur_raw = wave_r[qi_r];
    raster_stamp_t cur_quad;
    assign cur_quad = cur_raw;
    wire cur_covered = (|cur_quad.mask);
    wire last_qi     = (qi_r == QI_W'(NUM_LANES - 1));

    // same-pixel collision of the cursor quad against buffered quads
    reg collide;
    always @(*) begin
        collide = 1'b0;
        for (integer j = 0; j < NUM_LANES; ++j) begin
            if ((CNT_W'(j) < count_r)
             && (buf_r[j].pos_x == cur_quad.pos_x)
             && (buf_r[j].pos_y == cur_quad.pos_y)) begin
                collide = 1'b1;
            end
        end
    end

    wire buf_full   = (count_r == CNT_W'(NUM_LANES));
    // a covered cursor quad that cannot be appended right now forces a flush first
    wire need_flush_first = cur_covered && wave_valid_r && (buf_full || collide);
    // append the cursor quad this cycle
    wire do_append  = cur_covered && wave_valid_r && ~flush_r && ~need_flush_first;
    wire fills_last = do_append && (count_r == CNT_W'(NUM_LANES - 1));

    // ── output packed wave (pad unfilled slots with mask=0) ───────────────
    raster_stamp_t [NUM_LANES-1:0] out_stamps;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_out
        raster_stamp_t pad;
        always @(*) begin
            pad      = buf_r[l];
            pad.mask = '0;                              // masked-off lane
        end
        assign out_stamps[l] = (CNT_W'(l) < count_r) ? buf_r[l] : pad;
    end
    assign out_bus_if.req_valid      = flush_r;
    assign out_bus_if.req_data.stamps = out_stamps;
    wire out_fire = out_bus_if.req_valid && out_bus_if.req_ready;

    // accept a new input wave only when not holding one and not flushing
    assign in_bus_if.req_ready = ~wave_valid_r && ~flush_r;
    wire in_fire = in_bus_if.req_valid && in_bus_if.req_ready;

    // idle-timeout flush of a partial buffer at frame drain
    wire idle_timeout = ~wave_valid_r && ~flush_r && (count_r != '0)
                     && (idle_r == IDLE_W'(IDLE_FLUSH_CYCLES));

    always @(posedge clk) begin
        if (reset) begin
            count_r      <= '0;
            wave_valid_r <= 1'b0;
            qi_r         <= '0;
            flush_r      <= 1'b0;
            idle_r       <= '0;
        end else begin
            // ── accept input ──────────────────────────────────────────────
            if (in_fire) begin
                wave_r       <= in_bus_if.req_data.stamps;
                wave_valid_r <= 1'b1;
                qi_r         <= '0;
            end

            // ── scan / append ─────────────────────────────────────────────
            if (do_append) begin
                buf_r[count_r] <= cur_quad;
                count_r        <= count_r + CNT_W'(1);
            end
            // advance the scan cursor unless we must flush before this quad
            if (wave_valid_r && ~flush_r && ~need_flush_first) begin
                if (last_qi) begin
                    wave_valid_r <= 1'b0;              // wave consumed
                end else begin
                    qi_r <= qi_r + QI_W'(1);
                end
            end

            // ── raise a flush ─────────────────────────────────────────────
            if (~flush_r && (need_flush_first || fills_last || idle_timeout)) begin
                flush_r <= 1'b1;
            end
            // ── retire a flush ────────────────────────────────────────────
            if (flush_r && out_fire) begin
                flush_r <= 1'b0;
                count_r <= '0;
            end

            // ── idle counter (drain detection) ────────────────────────────
            if (in_fire || do_append || flush_r || wave_valid_r) begin
                idle_r <= '0;
            end else if (count_r != '0 && idle_r != IDLE_W'(IDLE_FLUSH_CYCLES)) begin
                idle_r <= idle_r + IDLE_W'(1);
            end
        end
    end

    assign busy = wave_valid_r || flush_r || (count_r != '0) || in_bus_if.req_valid;

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (out_fire) begin
            `TRACE(1, ("%d: %s packer flush: count=%0d\n", $time, INSTANCE_ID, count_r))
        end
    end
`endif

endmodule
