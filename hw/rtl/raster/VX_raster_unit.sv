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

// VX_raster_unit — per-core SFU PE for the single raster op (INST_SFU_RASTER):
//   vx_rast_fetch — pop the next covered-quad wave from the cluster raster bus and
//                   stage the per-lane frag_payload_t straight into the warp's gfx
//                   register window (slots GFXW_FRAG_SLOT_BASE..), then return a
//                   scoreboarded drained flag (rd: 1 = producer drained → the
//                   persistent fragment worker exits its loop). The FS reads the
//                   payload back with GETW — zero LMEM/LSU traffic (FWD-5, C1).
//                   No begin op (the producer auto-arms on its DCR config write),
//                   no bcoord CSRs, no pos_mask sentinel — the doctrine-clean
//                   (C2/C3) handoff.
//
// vx_rast_fetch is synchronous (completes when a wave is staged or the producer
// drains) — it always completes, so the head-of-line SFU switch never
// permanently blocks the OM ops the worker issues next.

`include "VX_raster_define.vh"

module VX_raster_unit import VX_gpu_pkg::*, VX_raster_pkg::*, VX_gfx_window_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0,
    parameter NUM_LANES = `VX_CFG_NUM_THREADS
) (
    input wire clk,
    input wire reset,

    // SFU PE-style interfaces
    VX_execute_if.slave    execute_if,
    VX_result_if.master    result_if,

    // Cluster-side raster bus (slave — agent pops descriptors)
    VX_raster_bus_if.slave raster_bus_if,

    // FWD payload-stage write port (vx_frag_fetch → gfx window slot RF).
    // One slot per cycle, all lanes in parallel; fire-and-forget (the window RF
    // always accepts — a dedicated 2nd consumer write port, disjoint slots).
    output wire                                   win_wr_en,
    output wire [NW_WIDTH-1:0]                     win_wr_wid,
    output wire [`CLOG2(`VX_CFG_NUM_THREADS)-1:0]  win_wr_tbase,
    output wire [NUM_LANES-1:0]                    win_wr_mask,
    output wire [GFXW_SLOT_BITS-1:0]               win_wr_slot,
    output wire [NUM_LANES-1:0][31:0]              win_wr_data
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (CORE_ID)

    wire is_fetch_op = 1'b1;   // the only raster op is vx_rast_fetch (funct3=3)

    // ── frag_payload_t window geometry ──
    localparam PAYLOAD_WORDS = GFXW_FRAG_WORDS;      // pos_mask, pid, bcoord[3][4]
    localparam SLOT_W        = `LOG2UP(PAYLOAD_WORDS);
    localparam LANE_BITS     = `CLOG2(NUM_LANES);
    localparam PID_W         = `LOG2UP(`VX_CFG_NUM_THREADS / NUM_LANES);
    localparam THREAD_BITS   = `CLOG2(`VX_CFG_NUM_THREADS);

    // ── frag-fetch window-stage FSM: one payload word per cycle ──
    localparam [0:0] S_IDLE = 1'd0, S_STAGE = 1'd1;
    reg state;
    reg [SLOT_W-1:0] slot_idx_r;

    wire raster_rsp_valid, raster_rsp_ready;

    // Producer state visible to the fetch op.
    wire bus_valid   = raster_bus_if.req_valid;
    wire bus_drained = raster_bus_if.req_data.done;

    // Quick-pop: capture a covered (non-drained) frag_fetch wave + free the bus
    // in one cycle, then window-stage from the latch. Holding the bus through the
    // multi-cycle stage would stall the cluster raster arb's fan-out.
    wire fetch_capture = is_fetch_op && execute_if.valid
                      && (state == S_IDLE) && bus_valid && ~bus_drained;

    // Latched wave + warp coords (the bus is freed at capture).
    raster_stamp_t [NUM_LANES-1:0]    wave_r;
    reg [NW_WIDTH-1:0]                 wr_wid_r;
    reg [THREAD_BITS-1:0]              wr_tbase_r;
    reg [NUM_LANES-1:0]               wr_mask_r;

    // Per-lane payload image: [0]=pos_mask, [1]=pid, [2+a*4+c]=bcoord[a][c].
    wire [NUM_LANES-1:0][PAYLOAD_WORDS-1:0][31:0] lane_payload;
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_lp
        assign lane_payload[l][0] = {wave_r[l].pos_y, wave_r[l].pos_x, wave_r[l].mask};
        assign lane_payload[l][1] = 32'(wave_r[l].pid);
        for (genvar a = 0; a < 3; ++a) begin : g_bc_a
            for (genvar c = 0; c < 4; ++c) begin : g_bc_c
                assign lane_payload[l][2 + a*4 + c] = wave_r[l].bcoords[a][c];
            end
        end
    end

    wire last_slot  = (slot_idx_r == SLOT_W'(PAYLOAD_WORDS - 1));
    wire stage_done = (state == S_STAGE) && last_slot;

    // ── window write drive (one slot per cycle, all latched lanes) ──
    assign win_wr_en    = (state == S_STAGE);
    assign win_wr_wid   = wr_wid_r;
    assign win_wr_tbase = `CLOG2(`VX_CFG_NUM_THREADS)'(wr_tbase_r);
    assign win_wr_mask  = wr_mask_r;
    assign win_wr_slot  = GFXW_SLOT_BITS'(GFXW_FRAG_SLOT_BASE + 32'(slot_idx_r));
    for (genvar l = 0; l < NUM_LANES; ++l) begin : g_wd
        assign win_wr_data[l] = lane_payload[l][slot_idx_r];
    end

    // ── op completion ──
    // A drained fetch completes immediately in IDLE (rd=1, worker exits); a
    // covered fetch completes when its payload stage finishes (rd=0).
    wire fetch_drained = is_fetch_op && execute_if.valid
                      && (state == S_IDLE) && bus_valid && bus_drained;
    wire fetch_complete = fetch_drained || stage_done;

    assign raster_rsp_valid = fetch_complete;

    assign execute_if.ready = fetch_complete && raster_rsp_ready;

    // Bus pop: drained fetch on complete; covered fetch at capture (quick-pop,
    // freeing the arb during the window stage).
    assign raster_bus_if.req_ready = (fetch_drained && raster_rsp_ready) || fetch_capture;

    // Auto-arm: a waiting vx_rast_fetch (op present, unit idle) asks the producer
    // to kick off its load — independent of req_valid (which only rises after the
    // load), so it breaks the load↔valid dependency. The producer dedupes.
    assign raster_bus_if.req_pending = is_fetch_op && execute_if.valid && (state == S_IDLE);

    wire [PID_W-1:0] fetch_pid = execute_if.data.header.pid;

    always @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            slot_idx_r <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    // quick-pop: latch the wave + warp coords, free the bus, stage next
                    if (fetch_capture) begin
                        wave_r     <= raster_bus_if.req_data.stamps;
                        wr_wid_r   <= execute_if.data.header.wid;
                        wr_tbase_r <= THREAD_BITS'(fetch_pid) << LANE_BITS;
                        wr_mask_r  <= execute_if.data.header.tmask;
                        state      <= S_STAGE;
                        slot_idx_r <= '0;
                    end
                end
                S_STAGE: begin
                    if (stage_done && raster_rsp_ready) begin
                        state <= S_IDLE;
                    end else if (~last_slot) begin
                        slot_idx_r <= slot_idx_r + SLOT_W'(1);
                    end
                end
            endcase
        end
    end

    // ── result word ──
    //   vx_rast_fetch: rd = drained flag (broadcast to all lanes).
    wire [NUM_LANES-1:0][31:0] response_data;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_response_data
        assign response_data[i] = {31'd0, fetch_drained};
    end

    sfu_result_t rsp_data_in;
    assign rsp_data_in.header = execute_if.data.header;
    for (genvar i = 0; i < NUM_LANES; ++i) begin : g_rsp_data
        assign rsp_data_in.data[i] = `VX_CFG_XLEN'(response_data[i]);
    end

    VX_elastic_buffer #(
        .DATAW ($bits(sfu_result_t)),
        .SIZE  (2)
    ) rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (raster_rsp_valid),
        .ready_in  (raster_rsp_ready),
        .data_in   (rsp_data_in),
        .data_out  (result_if.data),
        .valid_out (result_if.valid),
        .ready_out (result_if.ready)
    );

`ifdef DBG_TRACE_RASTER
    always @(posedge clk) begin
        if (execute_if.valid && execute_if.ready) begin
            `TRACE(1, ("%d: %s raster-op: wid=%0d, fetch=%b drained=%b (#%0d)\n",
                $time, INSTANCE_ID, execute_if.data.header.wid,
                is_fetch_op, raster_bus_if.req_data.done, execute_if.data.header.uuid))
        end
    end
`endif

endmodule
