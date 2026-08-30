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

// Rasterizer Early-Z (occlusion cull) stage
// Functionality: Receive a covered-quad wave with its screen-space depth plane.
// 1. Evaluate the per-pixel candidate depth (bit-identical to the FS late-Z).
// 2. Read the committed depth from the depth buffer (via the OM's ocache, so the
//    read is coherent with the ROP's write-through depth stores).
// 3. Clear coverage bits that fail the depth compare (VX_om_compare).
// 4. Forward the wave with the narrowed coverage mask.
//
// Cull is conservative and image-identical: the candidate depth matches the ROP
// late-Z exactly, and reading committed depth can only under-cull (stale depth
// reduces culling, never wrongly culls). Gated by dcrs.earlyz_safe — when clear,
// waves pass through unchanged (zero behavior change).

`include "VX_raster_define.vh"

`ifdef VX_CFG_RASTER_EARLYZ_ENABLE

module VX_raster_earlyz import VX_gpu_pkg::*; import VX_raster_pkg::*; import VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter OUTPUT_QUADS = 4
) (
    input wire clk,
    input wire reset,

    // Device configuration (shared depth-buffer config snooped from the OM DCRs)
    raster_dcrs_t                       dcrs,

    // Committed-depth read port (through the cluster ocache — coherent with the
    // OM's write-through depth stores, so early-Z culls the same set as the ROP).
    VX_mem_bus_if.master                cache_bus_if [OCACHE_NUM_REQS],

    // Input wave (from the slice)
    input wire                          valid_in,
    input raster_stamp_t [OUTPUT_QUADS-1:0] stamps_in,
    input wire [2:0][`RASTER_DATA_BITS-1:0] zplane_in,
    output wire                         ready_in,

    // Output wave (to the raster bus arbiter)
    output wire                         valid_out,
    output raster_stamp_t [OUTPUT_QUADS-1:0] stamps_out,
    input wire                          ready_out,

    // High while a wave is being depth-tested (keeps the frame drain honest)
    output wire                         busy
);
    localparam NUM_PIX     = OUTPUT_QUADS * 4;   // 4 pixels per quad
    localparam W_ADDR_BITS = (`RASTER_ADDR_BITS + 6) - 2;

    // dcrs carries the full raster config; early-Z uses only the depth fields.
    `UNUSED_VAR (dcrs)

    // ── Pass-through when early-Z is disabled ──────────────────────────────
    // A monotonic-depth / no-stencil frame enables early-Z; otherwise the wave
    // is forwarded verbatim (the ROP still does the authoritative late-Z).
    wire enabled = dcrs.earlyz_safe;

    // ═══════════════════════════════════════════════════════════════════════
    // Candidate depth: bit-identical to the FS late-Z (kernel PLANE_Z Q7.24 plane
    // MAC, saturated to the 24-bit zbuf range). Absolute pixel coords per pixel:
    //   X = pos_x*2 + (i&1),  Y = pos_y*2 + (i>>1)
    //
    // Late-Z truncates the int64 MAC to int32 before saturating, so the whole
    // evaluation is mod-2^32 and 32-bit products suffice. The quad origin term
    // za*2qx + zb*2qy + zc is computed once per quad from the registered wave;
    // each pixel then adds za/zb for its intra-quad offset a stage later.
    // ═══════════════════════════════════════════════════════════════════════

    wire [NUM_PIX-1:0] pix_cov_in;
    for (genvar q = 0; q < OUTPUT_QUADS; ++q) begin : g_cov_in
        for (genvar i = 0; i < 4; ++i) begin : g_pix
            assign pix_cov_in[q * 4 + i] = stamps_in[q].mask[i];
        end
    end

    // ═══════════════════════════════════════════════════════════════════════
    // Single-wave FSM: hold one wave, batch-read its covered pixels' committed
    // depth, compare, narrow coverage, forward. Serializing waves keeps the
    // logic simple; correctness (not throughput) is the goal.
    // ═══════════════════════════════════════════════════════════════════════
    localparam STATE_IDLE  = 3'd0;
    localparam STATE_EVAL1 = 3'd1;  // per-quad plane MAC / pitch products
    localparam STATE_EVAL2 = 3'd2;  // per-pixel offsets, saturation, addresses
    localparam STATE_READ  = 3'd3;  // issuing depth reads
    localparam STATE_WAIT  = 3'd4;  // waiting for the batch response
    localparam STATE_SEND  = 3'd5;  // forwarding the narrowed wave

    reg [2:0] state;

    raster_stamp_t [OUTPUT_QUADS-1:0] wave_stamps;
    reg [2:0][`RASTER_DATA_BITS-1:0]  wave_zplane;
    reg [NUM_PIX-1:0][23:0]           wave_cand;
    reg [NUM_PIX-1:0][W_ADDR_BITS-1:0] wave_waddr;
    reg [NUM_PIX-1:0]                  wave_cov;

    // Per-quad EVAL1 products (from the registered wave).
    reg [OUTPUT_QUADS-1:0][31:0]      quad_zbase;   // za*2qx + zb*2qy + zc (mod 2^32)
    reg [OUTPUT_QUADS-1:0][31:0]      quad_ypitch;  // 2qy * zbuf_pitch

    wire signed [31:0] za = wave_zplane[0];   // A' (Q7.24)
    wire signed [31:0] zb = wave_zplane[1];   // B'
    wire signed [31:0] zc = wave_zplane[2];   // C'

    wire [OUTPUT_QUADS-1:0][31:0] quad_zbase_n;
    wire [OUTPUT_QUADS-1:0][31:0] quad_ypitch_n;
    for (genvar q = 0; q < OUTPUT_QUADS; ++q) begin : g_quad_eval
        wire [`VX_RASTER_DIM_BITS-1:0] qx2 = {wave_stamps[q].pos_x, 1'b0};
        wire [`VX_RASTER_DIM_BITS-1:0] qy2 = {wave_stamps[q].pos_y, 1'b0};
        assign quad_zbase_n[q] = 32'(za * $signed({{(32-`VX_RASTER_DIM_BITS){1'b0}}, qx2})
                                   + zb * $signed({{(32-`VX_RASTER_DIM_BITS){1'b0}}, qy2})
                                   + zc);
        assign quad_ypitch_n[q] = 32'(qy2) * dcrs.zbuf_pitch;
    end

    // Per-pixel EVAL2 composition (from the registered per-quad products):
    //   zbits(p) = quad_zbase + i0*za + i1*zb  (mod 2^32), then saturate;
    //   waddr(p) = {zbuf_addr,4'b0} + X + ((2qy + i1) * pitch)[31:2]
    // The [31:2] shift is applied after the full 32-bit row product, so the
    // odd row adds the pitch before shifting (carries out of bits [1:0]).
    wire [NUM_PIX-1:0][23:0]           pix_cand_n;
    wire [NUM_PIX-1:0][W_ADDR_BITS-1:0] pix_waddr_n;
    for (genvar q = 0; q < OUTPUT_QUADS; ++q) begin : g_pix_eval
        for (genvar i = 0; i < 4; ++i) begin : g_pix
            localparam p = q * 4 + i;
            wire [`VX_RASTER_DIM_BITS-1:0] px = {wave_stamps[q].pos_x, 1'b0} | `VX_RASTER_DIM_BITS'(i[0]);
            wire signed [31:0] zbits = $signed(quad_zbase[q])
                                     + (i[0] ? za : 32'sd0)
                                     + (i[1] ? zb : 32'sd0);
            assign pix_cand_n[p] = zbits[31] ? 24'd0
                                 : (|zbits[30:24]) ? 24'(OM_DEPTH_MASK)
                                 : zbits[23:0];
            wire [31:0] y_pitch = quad_ypitch[q] + (i[1] ? dcrs.zbuf_pitch : 32'd0);
            wire [W_ADDR_BITS-1:0] baddr = {dcrs.zbuf_addr, 4'b0} + W_ADDR_BITS'(px);
            assign pix_waddr_n[p] = baddr + W_ADDR_BITS'(y_pitch[31:2]);
            `UNUSED_VAR (y_pitch)
        end
    end

    // Memory request/response (batch read of covered pixels)
    reg                          mreq_valid;
    wire                         mreq_ready;
    wire                         mreq_fire = mreq_valid && mreq_ready;

    wire                         mrsp_valid;
    wire [NUM_PIX-1:0]           mrsp_mask;
    wire [NUM_PIX-1:0][31:0]     mrsp_data;
    wire                         mrsp_ready;
    wire                         mrsp_fire = mrsp_valid && mrsp_ready;

    // Reflexive relaxation of the depth func: early-Z is read-only (the ROP is
    // the authoritative late-Z), and the committed depth we read is not causally
    // pinned to this fragment — it may already contain this fragment's own, a
    // co-planar (equal-depth), or a causally-later nearer write (fragments do not
    // reach the OM strictly in submission order). So a covered pixel may be
    // dropped only when it is STRICTLY behind the read depth. A visible fragment
    // has cand == final-buffer depth, which is on-or-ahead of any value we read,
    // so the strict-behind test can never cull it: enabling early-Z is
    // image-identical regardless of read freshness or pipeline ordering. Testing
    // with the exact func (culling on equality) is what wrongly drops such
    // fragments. Map LESS/LEQUAL -> LEQUAL, GREATER/GEQUAL -> GEQUAL; any other
    // (non-monotone) func never early-culls (ALWAYS keeps).
    reg [OM_DEPTH_FUNC_BITS-1:0] earlyz_func;
    always @(*) begin
        case (dcrs.depth_func)
            `VX_OM_DEPTH_FUNC_LESS,
            `VX_OM_DEPTH_FUNC_LEQUAL  : earlyz_func = `VX_OM_DEPTH_FUNC_LEQUAL;
            `VX_OM_DEPTH_FUNC_GREATER,
            `VX_OM_DEPTH_FUNC_GEQUAL  : earlyz_func = `VX_OM_DEPTH_FUNC_GEQUAL;
            default                   : earlyz_func = `VX_OM_DEPTH_FUNC_ALWAYS;
        endcase
    end

    // Depth compare per pixel (a = candidate, b = committed & mask). pix_pass is
    // the KEEP decision (reflexive-relaxed); a pixel is culled iff ~pix_pass.
    wire [NUM_PIX-1:0] pix_pass;
    for (genvar p = 0; p < NUM_PIX; ++p) begin : g_cmp
        wire [23:0] committed = mrsp_data[p][23:0] & 24'(OM_DEPTH_MASK);
        VX_om_compare #(
            .DATAW (24)
        ) cmp (
            .func   (earlyz_func),
            .a      (wave_cand[p]),
            .b      (committed),
            .result (pix_pass[p])
        );
    end

    // Narrowed coverage: keep a covered pixel only if it passes the compare.
    // Valid only in STATE_WAIT (uses the live depth response); latched into
    // wave_cov on the response fire for use in STATE_SEND.
    wire [NUM_PIX-1:0] narrowed_cov = wave_cov & pix_pass;

    // Rebuild the per-quad stamps with the (latched, narrowed) masks.
    raster_stamp_t [OUTPUT_QUADS-1:0] wave_stamps_n;
    for (genvar q = 0; q < OUTPUT_QUADS; ++q) begin : g_narrow
        always @(*) begin
            wave_stamps_n[q]      = wave_stamps[q];
            wave_stamps_n[q].mask = wave_cov[q*4 +: 4];
        end
    end

    // A wave whose coverage narrowed to nothing is dropped, not forwarded: the
    // rasterizer never emits an all-uncovered wave, and launching a fragment
    // shader with zero coverage would submit an empty OM request. Dropping is
    // image-identical (a fully-culled wave contributes no pixels).
    wire wave_all_empty = ~(| wave_cov);

    always @(posedge clk) begin
        if (reset) begin
            state      <= STATE_IDLE;
            mreq_valid <= 1'b0;
        end else begin
            case (state)
            STATE_IDLE: begin
                if (valid_in && enabled) begin
                    wave_stamps <= stamps_in;
                    wave_zplane <= zplane_in;
                    wave_cov    <= pix_cov_in;
                    state       <= STATE_EVAL1;
                end
            end
            STATE_EVAL1: begin
                quad_zbase  <= quad_zbase_n;
                quad_ypitch <= quad_ypitch_n;
                state       <= STATE_EVAL2;
            end
            STATE_EVAL2: begin
                wave_cand  <= pix_cand_n;
                wave_waddr <= pix_waddr_n;
                // No covered pixels at all → nothing to read; drop in SEND.
                mreq_valid <= (| wave_cov);
                state      <= (| wave_cov) ? STATE_READ : STATE_SEND;
            end
            STATE_READ: begin
                if (mreq_fire) begin
                    mreq_valid <= 1'b0;
                    state      <= STATE_WAIT;
                end
            end
            STATE_WAIT: begin
                if (mrsp_fire) begin
                    wave_cov <= narrowed_cov;
                    state    <= STATE_SEND;
                end
            end
            STATE_SEND: begin
                // Drop a fully-culled wave immediately; otherwise forward it.
                if (wave_all_empty || ready_out) begin
                    state <= STATE_IDLE;
                end
            end
            default: state <= STATE_IDLE;
            endcase
        end
    end

    // Accept a new wave only in IDLE (serialized). When disabled the wave is
    // passed straight through (combinational).
    wire fsm_accept = (state == STATE_IDLE) && enabled;

    // ── Output wave ────────────────────────────────────────────────────────
    // Enabled: forward the narrowed wave from STATE_SEND unless it is fully
    // culled. Disabled: pass the input straight through (no reads, no reorder).
    assign valid_out  = enabled ? ((state == STATE_SEND) && ~wave_all_empty) : valid_in;
    assign stamps_out = enabled ? wave_stamps_n                              : stamps_in;
    assign ready_in   = enabled ? fsm_accept                                 : ready_out;

    // ── Memory batch read (through the ocache) ───────────────────────────
    localparam SCHED_TAG_WIDTH = OCACHE_EARLYZ_REQ_TAG_WIDTH;

    wire [NUM_PIX-1:0][OCACHE_ADDR_WIDTH-1:0] mreq_addr;
    for (genvar p = 0; p < NUM_PIX; ++p) begin : g_mreq_addr
        assign mreq_addr[p] = OCACHE_ADDR_WIDTH'(wave_waddr[p]);
    end

    wire [NUM_PIX-1:0][OCACHE_WORD_SIZE-1:0] mreq_byteen;
    for (genvar p = 0; p < NUM_PIX; ++p) begin : g_mreq_byteen
        assign mreq_byteen[p] = {OCACHE_WORD_SIZE{1'b1}};
    end

    VX_lsu_mem_if #(
        .NUM_LANES (OCACHE_NUM_REQS),
        .DATA_SIZE (OCACHE_WORD_SIZE),
        .TAG_WIDTH (OCACHE_EARLYZ_REQ_TAG_WIDTH)
    ) mem_bus_if();

    VX_mem_scheduler #(
        .INSTANCE_ID  ($sformatf("%s-memsched", INSTANCE_ID)),
        .CORE_REQS    (NUM_PIX),
        .MEM_CHANNELS (OCACHE_NUM_REQS),
        .WORD_SIZE    (OCACHE_WORD_SIZE),
        .ADDR_WIDTH   (OCACHE_ADDR_WIDTH),
        .USER_WIDTH   (0),
        .TAG_WIDTH    (SCHED_TAG_WIDTH),
        .CORE_QUEUE_SIZE(`VX_CFG_RASTER_MEM_QUEUE_SIZE),
        .UUID_WIDTH   (UUID_WIDTH),
        .RSP_PARTIAL  (0),
        .RW_ENABLE    (0),
        .MEM_OUT_BUF  (2),
        .CORE_OUT_BUF (2)
    ) mem_scheduler (
        .clk            (clk),
        .reset          (reset),

        // Input request
        .core_req_valid (mreq_valid),
        .core_req_rw    (1'b0),
        .core_req_mask  (wave_cov),
        .core_req_byteen(mreq_byteen),
        .core_req_addr  (mreq_addr),
        .core_req_user  ('0),
        .core_req_data  ('0),
        .core_req_tag   (SCHED_TAG_WIDTH'(0)),
        .core_req_ready (mreq_ready),
        `UNUSED_PIN (req_queue_empty),
        `UNUSED_PIN (req_queue_rw_notify),

        // Output response
        .core_rsp_valid (mrsp_valid),
        .core_rsp_mask  (mrsp_mask),
        .core_rsp_data  (mrsp_data),
        `UNUSED_PIN (core_rsp_tag),
        `UNUSED_PIN (core_rsp_sop),
        `UNUSED_PIN (core_rsp_eop),
        .core_rsp_ready (mrsp_ready),

        // Memory request
        .mem_req_valid  (mem_bus_if.req_valid),
        .mem_req_rw     (mem_bus_if.req_data.rw),
        .mem_req_mask   (mem_bus_if.req_data.mask),
        .mem_req_byteen (mem_bus_if.req_data.byteen),
        .mem_req_addr   (mem_bus_if.req_data.addr),
        `UNUSED_PIN (mem_req_user),
        .mem_req_data   (mem_bus_if.req_data.data),
        .mem_req_tag    (mem_bus_if.req_data.tag),
        .mem_req_ready  (mem_bus_if.req_ready),

        // Memory response
        .mem_rsp_valid  (mem_bus_if.rsp_valid),
        .mem_rsp_mask   (mem_bus_if.rsp_data.mask),
        .mem_rsp_data   (mem_bus_if.rsp_data.data),
        .mem_rsp_tag    (mem_bus_if.rsp_data.tag),
        .mem_rsp_ready  (mem_bus_if.rsp_ready)
    );

    `UNUSED_VAR (mrsp_mask)

    // Early-Z never sets any memory attr; tie off the scheduler-driven LSU bus.
    assign mem_bus_if.req_data.user = '0;

    VX_lsu_adapter #(
        .NUM_LANES    (OCACHE_NUM_REQS),
        .DATA_SIZE    (OCACHE_WORD_SIZE),
        .TAG_WIDTH    (OCACHE_EARLYZ_REQ_TAG_WIDTH),
        .TAG_SEL_BITS (OCACHE_EARLYZ_REQ_TAG_WIDTH),
        .REQ_OUT_BUF  (0),
        .RSP_OUT_BUF  (0)
    ) lsu_adapter (
        .clk        (clk),
        .reset      (reset),
        .lsu_mem_if (mem_bus_if),
        .mem_bus_if (cache_bus_if)
    );

    // Wait for the batch response to finish before the wave is forwarded.
    assign mrsp_ready = (state == STATE_WAIT);

    // Busy while any wave is mid-flight (drain must cover in-flight depth reads).
    assign busy = enabled && (state != STATE_IDLE);

endmodule

`endif
