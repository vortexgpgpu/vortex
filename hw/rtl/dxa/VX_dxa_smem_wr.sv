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

// DXA SMEM Writer — OOO direct drain.
//
// Receives CLs directly from gmem_req on the `sw_*` channel, no rsp_buf
// in the middle. The pend slot is filled asynchronously by whatever rsp
// (real or OOB-synthetic) gmem_req presents. The CL marked `last` is
// special — it must drain LAST because its bus packet carries
// notify_smem_done; we hold it in `defer_*_r` until all other CLs have
// released, then promote it to pend.
//
// Drain: pend → barrel-shift → fb_data_r → SMEM_WORD beats.
// 1 SMEM-word/cycle steady state.

`include "VX_define.vh"

module VX_dxa_smem_wr import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter CL_SIZE         = `VX_CFG_L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE  = DXA_LMEM_WORD_SIZE,
    parameter SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W,
    parameter GMEM_DATAW      = CL_SIZE * 8
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire                        transfer_active,
    input  wire                        transfer_start,

    // cfill (stable during transfer).
    input  wire [31:0]                 cfill,

    // Metadata for SMEM bus tag + completion attr.
    input  wire [NC_WIDTH-1:0]         active_core_id,
    input  wire [UUID_WIDTH-1:0]       active_uuid,
    input  wire [BAR_ADDR_W-1:0]       active_bar_addr,
    input  wire                        active_notify_smem_done,

    // Direct drain channel (from gmem_req).
    input  wire                        sw_valid,
    output wire                        sw_ready,
    input  wire [TAG_W-1:0]            sw_tag,
    input  wire [GMEM_DATAW-1:0]       sw_data,
    input  wire [DXA_SMEM_ADDR_W-1:0]  sw_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      sw_byte_offset,
    input  wire [CL_OFF_BITS:0]        sw_valid_length,
    input  wire                        sw_oob,
    input  wire                        sw_last,
    input  wire [SEQ_W-1:0]            sw_outstanding,

    // Resource release to gmem_req (per-tag, OOO).
    output wire                        release_en,
    output wire [TAG_W-1:0]            release_tag,

    // SMEM bus interface (writes only).
    VX_mem_bus_if.master               smem_bus_if,

    // Completion.
    output wire                        transfer_done,
    output wire [31:0]                 wr_done_count,
    output wire                        smem_req_fire,

    // Multicast (always available; active when is_multicast is set).
    input  wire                        is_multicast,
    input  wire [`VX_CFG_NUM_WARPS-1:0]       cta_mask,
    input  wire [31:0]                 smem_stride,

    // K-major scatter mode (stable per transfer):
    //   dest_kmajor=1 → drain one element (elem_bytes wide) per SMEM beat,
    //   with per-beat addr += per_lane_stride_bytes. byteen masks all bytes
    //   except the `elem_bytes`-wide window at the current in-word offset.
    input  wire                        dest_kmajor,
    input  wire [15:0]                 per_lane_stride_bytes,
    input  wire [3:0]                  elem_bytes

`ifdef PERF_ENABLE
    ,
    output wire [31:0]                 perf_lmem_writes
`endif
);
    localparam CL_OFF_BITS  = `CLOG2(CL_SIZE);
    localparam SMEM_OFF_W   = `CLOG2(SMEM_WORD_SIZE);
    localparam SMEM_DATAW   = SMEM_WORD_SIZE * 8;
    localparam TAG_W        = `CLOG2(MAX_OUTSTANDING);
    localparam SEQ_W        = `CLOG2(MAX_OUTSTANDING + 1);
    localparam FILL_CAP     = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W       = `CLOG2(FILL_CAP + 1);

    localparam ENGINE_VALUE_W = DXA_LMEM_ENGINE_TAG_W - UUID_WIDTH;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;

    // ════════════════════════════════════════════════════════════════════
    // Cfill replication (for OOB CLs)
    // ════════════════════════════════════════════════════════════════════
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < CL_SIZE / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cfill;
    end

    // ════════════════════════════════════════════════════════════════════
    // Pending slot (1-deep skid from sw channel).
    // ════════════════════════════════════════════════════════════════════
    reg                       pend_valid_r;
    reg [TAG_W-1:0]           pend_tag_r;
    reg [DXA_SMEM_ADDR_W-1:0] pend_smem_byte_addr_r;
    reg [CL_OFF_BITS:0]       pend_valid_length_r;
    reg                       pend_last_r;
    // Payload held compressed (CL leading byte_offset dropped); position shift
    // applied at fb-load.
    reg [GMEM_DATAW-1:0]      pend_data_r;

    // ════════════════════════════════════════════════════════════════════
    // Deferred-last slot — holds the CL marked `last` while other CLs
    // drain ahead of it. Promoted to pend when only this CL remains.
    // ════════════════════════════════════════════════════════════════════
    reg                       defer_valid_r;
    reg [TAG_W-1:0]           defer_tag_r;
    reg [DXA_SMEM_ADDR_W-1:0] defer_smem_byte_addr_r;
    reg [CL_OFF_BITS:0]       defer_valid_length_r;
    reg [GMEM_DATAW-1:0]      defer_data_r;

    // ════════════════════════════════════════════════════════════════════
    // Drain fill buffer
    // ════════════════════════════════════════════════════════════════════
    reg                        fb_active_r;
    reg [TAG_W-1:0]            fb_tag_r;
    reg                        fb_last_r;
    reg [FILL_CAP*8-1:0]       fb_data_r /*verilator split_var*/;
    reg [FILL_W-1:0]           fb_level_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]       fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_start_word_r;
    // K-major scatter state: per-beat target byte address (this CL's
    // element-0 destination plus N*per_lane_stride per beat).
    reg [DXA_SMEM_ADDR_W-1:0] fb_byte_addr_r;

    // K-major drain quantum (in bytes) = 1 element per beat in scatter mode,
    // SMEM_WORD_SIZE bytes per beat in row-major streaming mode.
    wire [FILL_W-1:0] drain_q_bytes = dest_kmajor ? FILL_W'(elem_bytes) : FILL_W'(SMEM_WORD_SIZE);
    wire [SMEM_OFF_W-1:0] km_in_word_off = fb_byte_addr_r[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] km_word_addr = SMEM_ADDR_WIDTH'(fb_byte_addr_r >> SMEM_OFF_W);

    // ════════════════════════════════════════════════════════════════════
    // Drain-side combinational
    // ════════════════════════════════════════════════════════════════════
    // Mode-aware drain accounting.
    //   Row-major:  word at a time, possibly with a trailing partial word.
    //   K-major:    one element at a time, no partials (valid_length is a
    //               multiple of elem_bytes by descriptor invariant).
    wire has_full_word    = (fb_level_r >= drain_q_bytes);
    wire has_last_partial = !dest_kmajor && !has_full_word && (fb_level_r > 0);
    wire drain_valid      = fb_active_r && (has_full_word || has_last_partial);
    wire drain_will_empty = (fb_level_r <= drain_q_bytes);

    wire smem_wr_ready_internal;
    wire drain_fire = drain_valid && smem_wr_ready_internal;
    wire drain_emptying_now = drain_fire && drain_will_empty;
    wire fb_will_be_empty   = drain_emptying_now || ~fb_active_r;

    // SMEM word data and byteen from fill buffer.
    //
    // Row-major (default): fb_data_r is pre-shifted at load to start at the
    //   destination in-word offset; per beat we emit SMEM_WORD bytes with a
    //   leading/trailing mask. is_first_word covers the leading mask;
    //   trailing-tail bytes are masked by `byte_has_data` running out.
    //
    // K-major scatter: fb_data_r holds source bytes starting at element 0;
    //   per beat we emit one element worth of bytes shifted to km_in_word_off
    //   and a byteen of `elem_bytes` contiguous ones at that offset.
    wire is_first_word = (fb_word_addr_r == fb_start_word_r);

    // Per-beat element bytes (low `elem_bytes` bytes of fb_data_r) shifted
    // to km_in_word_off byte position. elem_bytes ≤ 8 always, so a 64-bit
    // slice suffices for the source data window.
    wire [63:0] km_elem_bytes_slice = fb_data_r[63:0];
    wire [SMEM_DATAW-1:0] km_elem_data_shifted =
        SMEM_DATAW'(km_elem_bytes_slice) << ({3'b000, km_in_word_off} << 3);

    wire [SMEM_DATAW-1:0] fb_word_data = dest_kmajor ? km_elem_data_shifted
                                                     : fb_data_r[SMEM_DATAW-1:0];

    // K-major byteen: `elem_bytes` contiguous bytes at km_in_word_off.
    wire [SMEM_WORD_SIZE-1:0] km_elem_mask_raw =
        SMEM_WORD_SIZE'((SMEM_WORD_SIZE'(1) << elem_bytes) - SMEM_WORD_SIZE'(1));
    wire [SMEM_WORD_SIZE-1:0] km_byteen = km_elem_mask_raw << km_in_word_off;

    wire [SMEM_WORD_SIZE-1:0] rm_byteen;
    for (genvar i = 0; i < SMEM_WORD_SIZE; ++i) begin : g_byteen
        wire byte_has_data = (FILL_W'(i) < fb_level_r);
        if (i < SMEM_WORD_SIZE - 1) begin : g_with_offset
            wire byte_is_offset = is_first_word && (SMEM_OFF_W'(i) < fb_byte_offset_r);
            assign rm_byteen[i] = byte_has_data && !byte_is_offset;
        end else begin : g_no_offset
            assign rm_byteen[i] = byte_has_data;
        end
    end
    wire [SMEM_WORD_SIZE-1:0] fb_word_byteen = dest_kmajor ? km_byteen : rm_byteen;

    // ════════════════════════════════════════════════════════════════════
    // sw-channel accept + load scheduling
    // ════════════════════════════════════════════════════════════════════
    //
    // The arriving sw CL takes one of two paths:
    //   (a) Capture into pend (the 1-deep skid; loaded into fb on the next
    //       fill-empty boundary). All non-deferred CLs use this path.
    //   (b) Capture into defer_*_r (it's the `last` CL and we still have
    //       other CLs outstanding to drain first).
    //
    // Promotion from defer_*_r to fb fires when the last drain leaves only the
    // deferred CL outstanding.

    wire sw_is_last_defer = sw_valid && sw_last && (sw_outstanding > SEQ_W'(1));
    wire can_defer        = sw_is_last_defer && ~defer_valid_r;

    wire can_capture_pend = ~pend_valid_r;

    // sw_ready conditions:
    //   - If sw is the last-to-defer: defer is empty → accept.
    //   - Else: pend can capture → accept.
    assign sw_ready = sw_is_last_defer ? can_defer : can_capture_pend;

    wire sw_accept     = sw_valid && sw_ready;
    wire sw_defer_path = sw_accept && sw_is_last_defer;
    // Every non-deferred CL is captured into pend (no combinational sw→fb
    // bypass): the fb_data_r load source is then always a pre-aligned
    // register, keeping both load barrel shifters off the fb_data_r setup
    // path. The extra cycle only occurs at fill-empty boundaries, off the
    // throughput-bound steady state.
    wire sw_pend_path  = sw_accept && ~sw_is_last_defer;

    // Deferred-CL promotion to fb: fires when fb is emptying and we are
    // about to be down to 1 outstanding (= just the deferred).
    // Equivalent: sw_outstanding == 2 && drain_emptying_now, or
    // sw_outstanding == 1 with fb idle (no other CL in flight).
    wire promote_defer = defer_valid_r && fb_will_be_empty && ~pend_valid_r
                      && ((sw_outstanding == SEQ_W'(1))
                          || (sw_outstanding == SEQ_W'(2) && drain_emptying_now));

    // pend → fb load (existing pend has data ready, fb emptying).
    wire load_pend_to_fb = pend_valid_r && fb_will_be_empty;

    // ════════════════════════════════════════════════════════════════════
    // fb-load source select (pend preferred over a promoted defer) + metadata.
    // ════════════════════════════════════════════════════════════════════
    wire use_pend_for_fb  = load_pend_to_fb;
    wire use_defer_for_fb = promote_defer && ~use_pend_for_fb;
    wire fb_load_now      = use_pend_for_fb || use_defer_for_fb;

    // pend/defer hold the payload already compressed (CL leading byte_offset
    // dropped). The in-word position shift is applied at fb-load — splitting
    // the two barrel shifts across the capture and load stages so neither the
    // gmem_req-slot-RAM-read→compress path nor the load→shift path carries both.
    wire [GMEM_DATAW-1:0]      fb_load_data_compressed = use_pend_for_fb ? pend_data_r : defer_data_r;
    wire [DXA_SMEM_ADDR_W-1:0] fb_load_smem_byte_addr  = use_pend_for_fb ? pend_smem_byte_addr_r : defer_smem_byte_addr_r;
    wire [CL_OFF_BITS:0]       fb_load_valid_length    = use_pend_for_fb ? pend_valid_length_r : defer_valid_length_r;
    wire [TAG_W-1:0]           fb_load_tag             = use_pend_for_fb ? pend_tag_r : defer_tag_r;
    wire                       fb_load_last            = use_pend_for_fb ? pend_last_r : 1'b1; // promoted defer is the last

    wire [SMEM_OFF_W-1:0]      new_smem_byte_off = fb_load_smem_byte_addr[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] new_start_word    = SMEM_ADDR_WIDTH'(fb_load_smem_byte_addr >> SMEM_OFF_W);
    //   Row-major fill level = leading-offset padding + valid bytes.
    //   K-major fill level   = exactly valid_length (each elem drained alone).
    wire [FILL_W-1:0]          new_fill_level    =
        dest_kmajor ? FILL_W'(fb_load_valid_length)
                    : (FILL_W'(new_smem_byte_off) + FILL_W'(fb_load_valid_length));
    // In-word position shift (bits), applied to the compressed payload at load.
    wire [FILL_W+2:0]          new_bit_offset    = {FILL_W'(new_smem_byte_off), 3'b000};

    // sw-side compress (drop the CL leading byte_offset) — the only barrel
    // shift on the capture path, which sits behind the gmem_req slot RAM read.
    wire [GMEM_DATAW-1:0] sw_payload    = sw_oob ? cfill_replicated : sw_data;
    wire [GMEM_DATAW-1:0] sw_compressed = (sw_valid_length != 0)
        ? (sw_payload >> {sw_byte_offset, 3'b000}) : '0;

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    // Per-beat fb_data_r shift amount (in bits) for K-major drain.
    wire [9:0] km_shift_bits = {3'b000, elem_bytes, 3'b000};  // elem_bytes * 8

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            pend_valid_r     <= 1'b0;
            defer_valid_r    <= 1'b0;
            fb_active_r      <= 1'b0;
            fb_data_r        <= '0;
            fb_level_r       <= '0;
            fb_word_addr_r   <= '0;
            fb_byte_offset_r <= '0;
            fb_start_word_r  <= '0;
            fb_byte_addr_r   <= '0;
            fb_tag_r         <= '0;
            fb_last_r        <= 1'b0;
        end else begin
            // ── Drain advance (mid-CL beat shift) ──
            if (drain_fire && ~drain_will_empty) begin
                if (dest_kmajor) begin
                    // K-major fill level = valid_length ≤ CL_SIZE, so all live
                    // data sits in the low GMEM_DATAW bits; bound the variable
                    // shift to that window instead of the full fb_data_r.
                    fb_data_r      <= (FILL_CAP*8)'(fb_data_r[GMEM_DATAW-1:0] >> km_shift_bits);
                    fb_level_r     <= fb_level_r - drain_q_bytes;
                    fb_byte_addr_r <= fb_byte_addr_r + DXA_SMEM_ADDR_W'(per_lane_stride_bytes);
                end else begin
                    fb_data_r      <= fb_data_r >> SMEM_DATAW;
                    fb_level_r     <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                    fb_word_addr_r <= fb_word_addr_r + SMEM_ADDR_WIDTH'(1);
                end
            end

            // ── Load fill buffer with the next CL ──
            //   The compressed payload (from pend/defer) is position-shifted to
            //   its in-word byte offset here — the load-stage barrel shift,
            //   gated by the now-registered fb_load_now control (Fix 1).
            //   Row-major: data sits at its in-word byte offset; per-beat drain
            //              slices the low SMEM_DATAW bits.
            //   K-major:   data lives at offset 0; km_elem_data_shifted scatters
            //              it per beat (combinational).
            if (fb_load_now) begin
                fb_data_r        <= dest_kmajor
                                  ? (FILL_CAP*8)'(fb_load_data_compressed)
                                  : ((FILL_CAP*8)'(fb_load_data_compressed) << new_bit_offset);
                fb_level_r       <= new_fill_level;
                fb_word_addr_r   <= new_start_word;
                fb_byte_offset_r <= new_smem_byte_off;
                fb_start_word_r  <= new_start_word;
                fb_byte_addr_r   <= fb_load_smem_byte_addr;
                fb_tag_r         <= fb_load_tag;
                fb_last_r        <= fb_load_last;
                fb_active_r      <= 1'b1;
            end else if (drain_emptying_now) begin
                fb_data_r   <= '0;
                fb_level_r  <= '0;
                fb_active_r <= 1'b0;
            end

            // ── pend slot management ──
            if (sw_pend_path) begin
                pend_tag_r            <= sw_tag;
                pend_smem_byte_addr_r <= sw_smem_byte_addr;
                pend_valid_length_r   <= sw_valid_length;
                pend_last_r           <= sw_last;
                pend_data_r           <= sw_compressed;
                pend_valid_r          <= 1'b1;
            end else if (use_pend_for_fb) begin
                pend_valid_r <= 1'b0;
            end

            // ── defer slot management ──
            if (sw_defer_path) begin
                defer_tag_r            <= sw_tag;
                defer_smem_byte_addr_r <= sw_smem_byte_addr;
                defer_valid_length_r   <= sw_valid_length;
                defer_data_r           <= sw_compressed;
                defer_valid_r          <= 1'b1;
            end else if (use_defer_for_fb) begin
                defer_valid_r <= 1'b0;
            end
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Resource release (per-tag, OOO)
    // ════════════════════════════════════════════════════════════════════
    wire drain_complete = drain_fire && drain_will_empty;

    assign release_en  = drain_complete;
    assign release_tag = fb_tag_r;

    // ════════════════════════════════════════════════════════════════════
    // SMEM write output (with optional multicast replay)
    // ════════════════════════════════════════════════════════════════════
    wire                       smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] smem_wr_addr;
    wire                       smem_wr_last_pkt;

    // LOG2UP (not CLOG2): NUM_WARPS=1 would make CLOG2 0 and the index vector
    // [-1:0]. VX_priority_encoder.index_out is `LOG2UP(N)` wide — match it.
    localparam MC_NW_BITS = `LOG2UP(`VX_CFG_NUM_WARPS);
    // visit_count width must match VX_popcount's output (CLOG2(N+1)) exactly,
    // else the -Wall WIDTHTRUNC check fires.
    localparam VISIT_CNT_W = `CLOG2(`VX_CFG_NUM_WARPS + 1);

    // ════════════════════════════════════════════════════════════════════
    // Pipelined multicast replay control
    // ════════════════════════════════════════════════════════════════════
    // cta_mask/smem_stride are stable per transfer, so the replay walk is a
    // deterministic iteration over the set bits of a fixed mask. The
    // NUM_WARPS-wide priority encoder + popcount are kept in the
    // register-input path; only registered control reaches the drain-ready
    // feedback, so smem_wr_ready_internal no longer traverses the PE.
    reg  [`VX_CFG_NUM_WARPS-1:0] replay_vec_r;       // active remaining mask
    reg  [MC_NW_BITS-1:0]        replay_next_idx_r;  // PE index of replay_vec_r
    reg  [`VX_CFG_NUM_WARPS-1:0] replay_onehot_r;    // PE onehot of replay_vec_r
    reg                          replay_has_rem_r;   // PE valid of replay_vec_r
    reg                          replay_is_last_r;   // replay_vec_r is one bit
    reg  [VISIT_CNT_W-1:0]       replay_visit_r;     // receivers serviced/word

    wire [MC_NW_BITS-1:0]  replay_next_idx      = replay_next_idx_r;
    wire                   replay_has_remaining = replay_has_rem_r;
    wire                   replay_is_last       = replay_is_last_r;
    wire [VISIT_CNT_W-1:0] visit_count          = replay_visit_r;

    wire mc_write_valid = transfer_active && drain_valid
                       && (!is_multicast || replay_has_remaining);
    wire mc_write_fire = mc_write_valid && smem_bus_if.req_ready;
    assign smem_wr_ready_internal = is_multicast
        ? (smem_bus_if.req_ready && (!replay_has_remaining || replay_is_last))
        : smem_bus_if.req_ready;

    // Next-cycle replay mask: clear the bit that fires this beat, reload
    // cta_mask at word boundaries (after the last receiver) and whenever the
    // engine is idle, so the first beat of any word starts from the full mask
    // without a reload bubble (reproduces the old combinational reload_now).
    wire replay_beat_fire = mc_write_fire && replay_has_remaining;
    wire [`VX_CFG_NUM_WARPS-1:0] replay_vec_next =
          ~drain_valid      ? cta_mask                          // idle: keep primed
        : ~replay_beat_fire ? replay_vec_r                      // stalled mid-word
        : replay_is_last    ? cta_mask                          // word done: reload
        :                     (replay_vec_r & ~replay_onehot_r); // advance in word

    // Pre-decode the next mask for the registered control.
    wire [MC_NW_BITS-1:0]        replay_idx_next;
    wire [`VX_CFG_NUM_WARPS-1:0] replay_onehot_next;
    wire                         replay_hasrem_next;
    VX_priority_encoder #(
        .N (`VX_CFG_NUM_WARPS)
    ) replay_pe (
        .data_in    (replay_vec_next),
        .onehot_out (replay_onehot_next),
        .index_out  (replay_idx_next),
        .valid_out  (replay_hasrem_next)
    );
    wire replay_islast_next = replay_hasrem_next
                           && (replay_vec_next == replay_onehot_next);
    wire [VISIT_CNT_W-1:0] replay_visit_next;
    `POP_COUNT(replay_visit_next, (cta_mask & ~replay_vec_next));

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            replay_vec_r      <= cta_mask;
            replay_next_idx_r <= '0;
            replay_onehot_r   <= '0;
            replay_has_rem_r  <= 1'b0;
            replay_is_last_r  <= 1'b0;
            replay_visit_r    <= '0;
        end else if (transfer_active && is_multicast) begin
            replay_vec_r      <= replay_vec_next;
            replay_next_idx_r <= replay_idx_next;
            replay_onehot_r   <= replay_onehot_next;
            replay_has_rem_r  <= replay_hasrem_next;
            replay_is_last_r  <= replay_islast_next;
            replay_visit_r    <= replay_visit_next;
        end
    end

    // r × stride: stride is byte-units on the bus; convert to word units
    // (SMEM_OFF_W = log2(SMEM_WORD_SIZE)) before adding to the word-aligned
    // fb_word_addr_r. The dispatcher guarantees aligned_lmem_size is a
    // multiple of MEM_BLOCK_SIZE (≥ SMEM_WORD_SIZE), so smem_stride
    // (host-set to the per-CTA LMEM size) is word-aligned and the shift is
    // lossless. visit_count is registered, so the multiply feeds only the
    // address output, not the drain-ready path.
    wire [SMEM_ADDR_WIDTH-1:0] stride_words = SMEM_ADDR_WIDTH'(smem_stride >> SMEM_OFF_W);
    wire [SMEM_ADDR_WIDTH-1:0] beat_offset  = stride_words * SMEM_ADDR_WIDTH'(visit_count);
    // Base word addr per beat: km_word_addr in K-major (per-elem scatter),
    // fb_word_addr_r in row-major (streamed). Replay then adds per-CTA offset.
    wire [SMEM_ADDR_WIDTH-1:0] base_word_addr = dest_kmajor ? km_word_addr : fb_word_addr_r;
    wire [SMEM_ADDR_WIDTH-1:0] replay_addr  = base_word_addr + beat_offset;
    // Only the low SMEM_ADDR_WIDTH+SMEM_OFF_W bits of smem_stride are used.
    `UNUSED_VAR (smem_stride)
    // per_lane_stride_bytes is LMEM-bounded; only the low DXA_SMEM_ADDR_W bits feed fb_byte_addr_r.
    `UNUSED_VAR (per_lane_stride_bytes[15:DXA_SMEM_ADDR_W])

    assign smem_wr_valid   = mc_write_valid;
    assign smem_wr_addr    = is_multicast ? replay_addr : base_word_addr;
    assign smem_req_fire   = mc_write_fire;

    wire is_last_drain = fb_last_r && drain_will_empty;
    assign smem_wr_last_pkt = is_last_drain && (!is_multicast || replay_is_last);

    // Completion attr: bar_stride hardcoded to 1.
    // Use mc_write_valid (not mc_write_fire) to break a UNOPTFLAT loop:
    // mc_write_fire = mc_write_valid && smem_bus_if.req_ready, and using
    // it on the request side closes a comb cycle through the downstream
    // LMEM arbiter. Attr is sampled by the receiver only when req_valid
    // && req_ready, so qualifying with req_ready here is redundant.
    wire smem_wr_attr_last = active_notify_smem_done && (
        is_multicast ? (mc_write_valid && is_last_drain) : smem_wr_last_pkt);
    wire [BAR_ADDR_W-1:0] smem_wr_attr_bar = is_multicast
        ? BAR_ADDR_W'(active_bar_addr + (BAR_ADDR_W'(replay_next_idx) << NB_BITS))
        : active_bar_addr;

    // ════════════════════════════════════════════════════════════════════
    // SMEM bus wiring
    // ════════════════════════════════════════════════════════════════════
    assign smem_bus_if.req_valid       = smem_wr_valid;
    assign smem_bus_if.req_data.rw     = 1'b1;
    assign smem_bus_if.req_data.addr   = smem_wr_addr;
    assign smem_bus_if.req_data.data   = fb_word_data;
    assign smem_bus_if.req_data.byteen = fb_word_byteen;
    assign smem_bus_if.req_data.attr   = {smem_wr_attr_last, smem_wr_attr_bar};
    assign smem_bus_if.req_data.tag.uuid  = active_uuid;   // tag DXA write with its issuing uuid
    assign smem_bus_if.req_data.tag.value = SMEM_TAG_VALUE_W'(active_core_id) << ENGINE_VALUE_W;
    assign smem_bus_if.rsp_ready       = 1'b0;

    `UNUSED_VAR (smem_bus_if.rsp_valid)
    `UNUSED_VAR (smem_bus_if.rsp_data)

    // ════════════════════════════════════════════════════════════════════
    // Completion tracking
    // ════════════════════════════════════════════════════════════════════
    reg [31:0] wr_count_r;

    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wr_count_r <= '0;
        end else if (smem_req_fire) begin
            wr_count_r <= wr_count_r + 32'd1;
        end
    end

    assign wr_done_count = wr_count_r;
    assign transfer_done = transfer_active && smem_req_fire && smem_wr_last_pkt;

`ifdef PERF_ENABLE
    reg [31:0] wrp_total_lmem_writes_r;
    always @(posedge clk) begin
        if (reset || transfer_start) begin
            wrp_total_lmem_writes_r <= '0;
        end else if (smem_req_fire) begin
            wrp_total_lmem_writes_r <= wrp_total_lmem_writes_r + 32'd1;
        end
    end
    assign perf_lmem_writes = wrp_total_lmem_writes_r + 32'(smem_req_fire);
`endif

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (sw_accept) begin
                $write("DXA_PIPE,%0d,SW_RX,tag=%0d,oob=%0d,last=%0d,path=%s\n",
                    $time, sw_tag, sw_oob, sw_last,
                    sw_defer_path ? "defer" : "pend");
            end
            if (smem_req_fire) begin
                $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                    $time, smem_wr_addr, fb_word_byteen, smem_wr_last_pkt);
            end
        end
    end
`endif

endmodule
