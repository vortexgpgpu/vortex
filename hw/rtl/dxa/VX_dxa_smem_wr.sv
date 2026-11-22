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
// (real or OOB-synthetic) gmem_req presents, from any transfer in flight;
// each CL carries its transfer's context. The transfer's notify_smem_done
// rides the write of whichever CL drains last: the one loaded into the fill
// buffer once its transfer has issued every CL and has nothing else pending.
//
// Drain: pend → barrel-shift → fb_data_r → SMEM_WORD beats.
// 1 SMEM-word/cycle steady state.

`include "VX_define.vh"

module VX_dxa_smem_wr import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter MAX_OUTSTANDING = 8,
    parameter NUM_XFERS       = 2,
    parameter CL_SIZE         = `VX_CFG_L1_LINE_SIZE,
    parameter SMEM_WORD_SIZE  = DXA_LMEM_WORD_SIZE,
    parameter SMEM_ADDR_WIDTH = DXA_LMEM_ADDR_W,
    parameter GMEM_DATAW      = CL_SIZE * 8,
    localparam CL_OFF_BITS    = `CLOG2(CL_SIZE),
    localparam TAG_W          = `CLOG2(MAX_OUTSTANDING),
    localparam SEQ_W          = `CLOG2(MAX_OUTSTANDING + 1),
    localparam XFER_W         = `LOG2UP(NUM_XFERS)
) (
    input  wire                        clk,
    input  wire                        reset,

    // Transfer contexts (registered read for the captured CL, combinational
    // read of the arrival fields), and per transfer: all CLs issued, CLs not
    // yet written.
    output wire [XFER_W-1:0]           pend_raddr,
    input  dxa_xfer_t                  pend_rdata,
    input  wire [31:0]                 cap_cfill,
    input  wire                        cap_scatter,
    input  wire                        cap_tiled,
    input  wire [3:0]                  cap_lg_tcN,
    input  wire [NUM_XFERS-1:0]        xfer_issued,
    input  wire [NUM_XFERS-1:0][SEQ_W-1:0] xfer_pending,

    // Direct drain channel (from gmem_req).
    input  wire                        sw_valid,
    output wire                        sw_ready,
    input  wire [TAG_W-1:0]            sw_tag,
    input  wire [XFER_W-1:0]           sw_xfer,
    input  wire [GMEM_DATAW-1:0]       sw_data,
    input  wire [DXA_SMEM_ADDR_W-1:0]  sw_smem_byte_addr,
    input  wire [CL_OFF_BITS-1:0]      sw_byte_offset,
    input  wire [CL_OFF_BITS:0]        sw_valid_length,
    input  wire                        sw_oob,
    input  wire [15:0]                 sw_k_row,    // tiled scatter
    input  wire [15:0]                 sw_n_base,   // tiled scatter

    // Resource release to gmem_req (per-tag, OOO).
    output wire                        release_en,
    output wire [TAG_W-1:0]            release_tag,
    output wire [XFER_W-1:0]           release_xfer,

    // SMEM bus interface (writes only).
    VX_mem_bus_if.master               smem_bus_if,

    // Completion.
    output wire                        xfer_done,
    output wire [XFER_W-1:0]           xfer_done_idx,
    output wire                        smem_req_fire
);
    localparam SMEM_OFF_W   = `CLOG2(SMEM_WORD_SIZE);
    localparam SMEM_DATAW   = SMEM_WORD_SIZE * 8;
    localparam FILL_CAP     = CL_SIZE + SMEM_WORD_SIZE;
    localparam FILL_W       = `CLOG2(FILL_CAP + 1);
    // Positioning is split fine (intra-word, at capture) + coarse (whole-word, at
    // fb-load) so no single-cycle barrel spans the full FILL_CAP.
    localparam SMEM_DATAW_LOG = SMEM_OFF_W + 3;
    localparam SMEM_WORDS_CAP = (FILL_CAP + SMEM_WORD_SIZE - 1) / SMEM_WORD_SIZE;
    localparam COARSE_W       = `CLOG2(SMEM_WORDS_CAP + 1);

    localparam ENGINE_VALUE_W = DXA_LMEM_ENGINE_TAG_W - UUID_WIDTH;
    localparam SMEM_TAG_VALUE_W = DXA_LMEM_TAG_W - UUID_WIDTH;

    // The context of the captured CL (dest calc, fb-load) comes from the
    // context RAM, addressed one cycle ahead so it lands with pend_xfer_r; the
    // arriving CL (capture) reads only its few arrival fields. A CL's context is
    // copied into fb_x_r at fb-load, so every beat-rate cone starts at a local
    // register.
    dxa_xfer_t pend_x;
    reg [XFER_W-1:0] pend_xfer_r;
    assign pend_x = pend_rdata;
    wire [15:0] cap_tcn_mask = (16'd1 << cap_lg_tcN) - 16'd1;

    // ════════════════════════════════════════════════════════════════════
    // Cfill replication (for OOB CLs)
    // ════════════════════════════════════════════════════════════════════
    wire [GMEM_DATAW-1:0] cfill_replicated;
    for (genvar i = 0; i < CL_SIZE / 4; ++i) begin : g_cfill
        assign cfill_replicated[i*32 +: 32] = cap_cfill;
    end

    // ════════════════════════════════════════════════════════════════════
    // Pending slot (1-deep skid from sw channel).
    // ════════════════════════════════════════════════════════════════════
    reg                       pend_valid_r;
    reg [TAG_W-1:0]           pend_tag_r;
    reg [DXA_SMEM_ADDR_W-1:0] pend_smem_byte_addr_r;
    reg [CL_OFF_BITS:0]       pend_valid_length_r;
    // Tiled scatter: the CL's raw (k, n) captured from the sw channel, its n
    // phase within an n-block, and the element-0 destination computed by the
    // shared permute one cycle after capture (pend_calc_r marks it pending;
    // fb-load waits on it — a CL drains over many beats, so the extra cycle
    // is off the steady state).
    reg [15:0]                pend_k_r, pend_n_r;
    reg [DXA_SMEM_ADDR_W-1:0] pend_dest_addr_r;
    reg [15:0]                pend_n_in_r;
    reg                       pend_calc_r;
    reg                       pend_addr_rdy_r;
    // Fine-positioned payload; pend_coarse_r is the whole-word residual applied at
    // fb-load. Width FILL_CAP (CL + one SMEM word of in-word-offset headroom).
    reg [FILL_CAP*8-1:0]      pend_data_r;
    reg [COARSE_W-1:0]        pend_coarse_r;

    // ════════════════════════════════════════════════════════════════════
    // Drain fill buffer
    // ════════════════════════════════════════════════════════════════════
    reg                        fb_active_r;
    reg [TAG_W-1:0]            fb_tag_r;
    reg [XFER_W-1:0]           fb_xfer_r;
    dxa_xfer_t                 fb_x_r;
    reg                        fb_last_r;
    reg [FILL_CAP*8-1:0]       fb_data_r;
    reg [FILL_W-1:0]           fb_level_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_word_addr_r;
    reg [SMEM_OFF_W-1:0]       fb_byte_offset_r;
    reg [SMEM_ADDR_WIDTH-1:0]  fb_start_word_r;
    // K-major scatter state: per-beat target byte address (this CL's
    // element-0 destination plus N*per_lane_stride per beat).
    reg [DXA_SMEM_ADDR_W-1:0] fb_byte_addr_r;
    // K-major read offset (bytes into fb_data_r), advances by elem_bytes per
    // beat. A read pointer instead of a per-beat shift of fb_data_r keeps a
    // variable barrel shift out of the fb_data_r path.
    reg [CL_OFF_BITS-1:0]     km_rd_off_r;
    // Tiled scatter (Flat/BlockMajor): the per-element SMEM destination is the
    // bbuf-native permuted index, not a uniform stride. fb_byte_addr_r holds
    // it directly (loaded with the CL's precomputed element-0 dest, advanced
    // by the step/wrap constants per beat); the in-block phase counter and
    // its registered wrap flag pick which constant applies.
    reg [15:0]                fb_n_in_r;
    reg                       fb_n_wrap_r;

    // K-major/tiled drain quantum (in bytes) = 1 element per beat in scatter
    // mode, SMEM_WORD_SIZE bytes per beat in row-major streaming mode.
    wire scatter = fb_x_r.scatter;
    wire [FILL_W-1:0] drain_q_bytes = scatter ? FILL_W'(fb_x_r.elem_bytes) : FILL_W'(SMEM_WORD_SIZE);
    // Effective per-element SMEM byte address: both scatter sub-modes read
    // the fb_byte_addr_r accumulator.
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
    wire has_last_partial = !scatter && !has_full_word && (fb_level_r > 0);
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

    // Per-beat element bytes: the 64-bit window at km_rd_off_r (a moving read
    // pointer, not a shifted register), then scattered to km_in_word_off below.
    // elem_bytes ≤ 8, so a 64-bit window covers the live element.
    wire [$clog2(FILL_CAP*8)-1:0] km_rd_bit = ($clog2(FILL_CAP*8))'({km_rd_off_r, 3'b000});
    wire [63:0] km_elem_bytes_slice = fb_data_r[km_rd_bit +: 64];
    wire [SMEM_DATAW-1:0] km_elem_data_shifted =
        SMEM_DATAW'(km_elem_bytes_slice) << ({3'b000, km_in_word_off} << 3);

    wire [SMEM_DATAW-1:0] fb_word_data = scatter ? km_elem_data_shifted
                                                 : fb_data_r[SMEM_DATAW-1:0];

    // K-major byteen: `elem_bytes` contiguous bytes at km_in_word_off.
    wire [SMEM_WORD_SIZE-1:0] km_elem_mask_raw =
        SMEM_WORD_SIZE'((SMEM_WORD_SIZE'(1) << fb_x_r.elem_bytes) - SMEM_WORD_SIZE'(1));
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
    wire [SMEM_WORD_SIZE-1:0] fb_word_byteen = scatter ? km_byteen : rm_byteen;

    // ════════════════════════════════════════════════════════════════════
    // sw-channel accept + load scheduling
    // ════════════════════════════════════════════════════════════════════
    //
    // Every arriving CL is captured into pend (no combinational sw→fb bypass),
    // so the fb-load source is always a register; only the shallow coarse shift
    // is on the load path. The extra cycle is at fill-empty boundaries only.
    assign sw_ready = ~pend_valid_r;
    wire sw_accept  = sw_valid && sw_ready;
    assign pend_raddr = sw_accept ? sw_xfer : pend_xfer_r;

    // pend → fb load (pend has data and, for tiled loads, its dest address).
    wire fb_load_now = pend_valid_r && pend_addr_rdy_r && fb_will_be_empty;

    // The CL being loaded is its transfer's final one when the transfer has
    // issued every CL and this is the only one still pending (the CL draining
    // out of fb this cycle releases its count as this one loads).
    wire [SEQ_W-1:0] pend_xfer_left = xfer_pending[pend_xfer_r]
        - SEQ_W'(drain_emptying_now && (fb_xfer_r == pend_xfer_r));
    wire pend_final = xfer_issued[pend_xfer_r] && (pend_xfer_left == SEQ_W'(1));

    // Apply the coarse whole-word shift the capture stage deferred; with the fine
    // shift already in pend this fully positions the payload.
    wire [FILL_CAP*8-1:0]      fb_load_data           = pend_data_r >> {pend_coarse_r, {SMEM_DATAW_LOG{1'b0}}};
    wire [DXA_SMEM_ADDR_W-1:0] fb_load_smem_byte_addr = pend_smem_byte_addr_r;
    wire [CL_OFF_BITS:0]       fb_load_valid_length   = pend_valid_length_r;

    wire [SMEM_OFF_W-1:0]      new_smem_byte_off = fb_load_smem_byte_addr[SMEM_OFF_W-1:0];
    wire [SMEM_ADDR_WIDTH-1:0] new_start_word    = SMEM_ADDR_WIDTH'(fb_load_smem_byte_addr >> SMEM_OFF_W);
    //   Row-major fill level = leading-offset padding + valid bytes.
    //   K-major fill level   = exactly valid_length (each elem drained alone).
    wire [FILL_W-1:0]          new_fill_level    =
        pend_x.scatter ? FILL_W'(fb_load_valid_length)
                : (FILL_W'(new_smem_byte_off) + FILL_W'(fb_load_valid_length));

    // Positioning drops the CL byte_offset and lands bytes at their destination;
    // POS_BIAS keeps the amount non-negative. sw_pos_amt splits into fine
    // (capture) + coarse (fb-load), bit-identical to the one byte-shift but with
    // neither stage barreling the full FILL_CAP.
    localparam POS_BIAS = SMEM_WORD_SIZE - 1;
    // Positioning amount = POS_BIAS (≤ SMEM_WORD_SIZE-1) + byte_offset (≤ CL_SIZE-1),
    // less the SMEM in-word offset. With wide SMEM words (XLEN=64 → SMEM_WORD_SIZE
    // up to 256B) POS_BIAS+byte_offset exceeds 8 bits, so size to FILL_W to avoid
    // wrap (which would mis-shift word-aligned rows and drop their data).
    wire [SMEM_OFF_W-1:0] sw_smem_off = sw_smem_byte_addr[SMEM_OFF_W-1:0];
    wire [FILL_W-1:0] sw_pos_amt = cap_scatter
        ? (FILL_W'(POS_BIAS) + FILL_W'(sw_byte_offset))
        : ((FILL_W'(POS_BIAS) + FILL_W'(sw_byte_offset)) - FILL_W'(sw_smem_off));
    // Fine = intra-word bytes (capture barrel); coarse = whole words (at fb-load).
    wire [SMEM_OFF_W-1:0] sw_fine_amt   = sw_pos_amt[SMEM_OFF_W-1:0];
    wire [COARSE_W-1:0]   sw_coarse_amt = COARSE_W'(sw_pos_amt >> SMEM_OFF_W);
    // Per-CL element-0 tiled destination — the only place the full permute
    // runs; drain beats advance fb_byte_addr_r by the precomputed step/wrap
    // constants. The permute is deeper than one cycle (two nested shift-add
    // trees), so it is pipelined over two: stage 1 decomposes the (k, n)
    // captured the cycle before and folds every element-granular term into
    // an esize-scaled intra term; stage 2 shifts the block index by the
    // transfer's constant and sums with the base. Per-CL rate makes the
    // latency free (a CL drains over many beats). pend holds its CL, hence
    // its transfer, until both stages have landed.
    wire calc_pend = pend_calc_r;

    wire [15:0] calc_n_blk   = pend_n_r >> pend_x.lg_tcN;
    wire [15:0] calc_n_in    = pend_n_r & pend_x.tcn_mask;
    wire [15:0] f_k_word     = pend_k_r >> pend_x.lg_ratio;
    wire [15:0] f_k_blk      = f_k_word >> (pend_x.lg_tcN + 4'd1);
    wire [15:0] b_k_blk      = pend_k_r >> pend_x.lg_bkK;

    wire [15:0] calc_k_blk = pend_x.is_flat ? f_k_blk : b_k_blk;
    wire [31:0] calc_blk_w   = (32'(calc_k_blk) << pend_x.lg_nsteps) + 32'(calc_n_blk);
    // the masked fields are disjoint, so the nested shift-adds are one OR
    wire [31:0] calc_intra_w = (32'(pend_k_r & pend_x.intra_khi_mask) << pend_x.intra_khi_sh)
                             | (32'(calc_n_in) << pend_x.intra_n_sh)
                             | (32'(pend_k_r & pend_x.intra_klo_mask) << pend_x.esize);

    reg [31:0] calc_blk_r, calc_intra_r;
    reg        calc2_valid_r;
    wire [DXA_SMEM_ADDR_W-1:0] calc_dest_addr = pend_x.smem_base
        + DXA_SMEM_ADDR_W'((calc_blk_r << pend_x.calc_sh2) + calc_intra_r);

    wire [15:0] sw_n_in = sw_n_base & cap_tcn_mask;

    wire [GMEM_DATAW-1:0] sw_payload      = sw_oob ? cfill_replicated : sw_data;
    wire [FILL_CAP*8-1:0] sw_padded       = (FILL_CAP*8)'(sw_payload) << (POS_BIAS*8);
    // Capture applies the fine shift only; coarse follows at fb-load.
    wire [FILL_CAP*8-1:0] sw_fine_shifted = (sw_valid_length != 0)
        ? (sw_padded >> {sw_fine_amt, 3'b000}) : '0;

    // ════════════════════════════════════════════════════════════════════
    // Sequential update
    // ════════════════════════════════════════════════════════════════════
    always @(posedge clk) begin
        if (reset) begin
            pend_valid_r     <= 1'b0;
            pend_xfer_r      <= '0;
            pend_calc_r      <= 1'b0;
            pend_addr_rdy_r  <= 1'b0;
            calc2_valid_r    <= 1'b0;
            fb_active_r      <= 1'b0;
            fb_data_r        <= '0;
            fb_level_r       <= '0;
            fb_word_addr_r   <= '0;
            fb_byte_offset_r <= '0;
            fb_start_word_r  <= '0;
            fb_byte_addr_r   <= '0;
            km_rd_off_r      <= '0;
            fb_n_in_r        <= '0;
            fb_n_wrap_r      <= 1'b0;
            fb_tag_r         <= '0;
            fb_xfer_r        <= '0;
            fb_x_r           <= '0;
            fb_last_r        <= 1'b0;
        end else begin
            // ── Drain advance (mid-CL beat) ──
            if (drain_fire && ~drain_will_empty) begin
                if (scatter) begin
                    // Read pointer advances; fb_data_r is NOT shifted. K-major
                    // strides the dest uniformly; tiled advances it by the
                    // in-block/wrap constant selected by the registered flag.
                    km_rd_off_r    <= km_rd_off_r + CL_OFF_BITS'(fb_x_r.elem_bytes);
                    fb_level_r     <= fb_level_r - drain_q_bytes;
                    fb_byte_addr_r <= fb_byte_addr_r
                        + (fb_x_r.tiled ? (fb_n_wrap_r ? fb_x_r.tiled_wrap : fb_x_r.tiled_step)
                                        : fb_x_r.lane_stride);
                    fb_n_in_r      <= fb_n_wrap_r ? 16'd0 : (fb_n_in_r + 16'd1);
                    fb_n_wrap_r    <= ((fb_n_wrap_r ? 16'd0 : (fb_n_in_r + 16'd1)) == fb_x_r.tcn_mask);
                end else begin
                    // Row-major: fixed whole-word shift (free), per-beat slice
                    // is the low SMEM_DATAW bits.
                    fb_data_r      <= fb_data_r >> SMEM_DATAW;
                    fb_level_r     <= fb_level_r - FILL_W'(SMEM_WORD_SIZE);
                    fb_word_addr_r <= fb_word_addr_r + SMEM_ADDR_WIDTH'(1);
                end
            end

            // ── Load fill buffer with the next CL ──
            //   fb_load_data is already fully positioned (fine from pend +
            //   coarse applied above), so this is a plain register load.
            if (fb_load_now) begin
                fb_data_r        <= fb_load_data;
                fb_level_r       <= new_fill_level;
                fb_word_addr_r   <= new_start_word;
                fb_byte_offset_r <= new_smem_byte_off;
                fb_start_word_r  <= new_start_word;
                fb_byte_addr_r   <= pend_x.tiled ? pend_dest_addr_r : fb_load_smem_byte_addr;
                km_rd_off_r      <= '0;
                fb_n_in_r        <= pend_n_in_r;
                fb_n_wrap_r      <= (pend_n_in_r == pend_x.tcn_mask);
                fb_tag_r         <= pend_tag_r;
                fb_xfer_r        <= pend_xfer_r;
                fb_x_r           <= pend_x;
                fb_last_r        <= pend_final;
                fb_active_r      <= 1'b1;
            end else if (drain_emptying_now) begin
                fb_data_r   <= '0;
                fb_level_r  <= '0;
                fb_active_r <= 1'b0;
            end

            // ── Per-CL dest-addr calc pipeline: stage-1 issue, stage-2 land ──
            calc2_valid_r <= calc_pend;
            if (calc_pend) begin
                calc_blk_r   <= calc_blk_w;
                calc_intra_r <= calc_intra_w;
                pend_calc_r  <= 1'b0;
            end
            if (calc2_valid_r) begin
                pend_dest_addr_r <= calc_dest_addr;
                pend_addr_rdy_r  <= 1'b1;
            end

            // ── pend slot management ──
            if (sw_accept) begin
                pend_tag_r            <= sw_tag;
                pend_xfer_r           <= sw_xfer;
                pend_smem_byte_addr_r <= sw_smem_byte_addr;
                pend_valid_length_r   <= sw_valid_length;
                pend_data_r           <= sw_fine_shifted;
                pend_coarse_r         <= sw_coarse_amt;
                pend_k_r              <= sw_k_row;
                pend_n_r              <= sw_n_base;
                pend_n_in_r           <= sw_n_in;
                pend_calc_r           <= cap_tiled;
                pend_addr_rdy_r       <= ~cap_tiled;
                pend_valid_r          <= 1'b1;
            end else if (fb_load_now) begin
                pend_valid_r <= 1'b0;
            end
        end
    end

    // ════════════════════════════════════════════════════════════════════
    // Resource release (per-tag, OOO)
    // ════════════════════════════════════════════════════════════════════
    wire drain_complete = drain_fire && drain_will_empty;

    assign release_en   = drain_complete;
    assign release_tag  = fb_tag_r;
    assign release_xfer = fb_xfer_r;

    // ════════════════════════════════════════════════════════════════════
    // SMEM write output (with optional multicast replay)
    // ════════════════════════════════════════════════════════════════════
    wire                       smem_wr_valid;
    wire [SMEM_ADDR_WIDTH-1:0] smem_wr_addr;
    wire                       smem_wr_last_pkt;

    // LOG2UP (not CLOG2): NUM_WARPS=1 would make CLOG2 0 and the index vector
    // [-1:0]. VX_priority_encoder.index_out is `LOG2UP(N)` wide — match it.
    localparam MC_NW_BITS = `LOG2UP(`VX_CFG_NUM_WARPS);

    // ════════════════════════════════════════════════════════════════════
    // Pipelined multicast replay control
    // ════════════════════════════════════════════════════════════════════
    // cta_mask/smem_stride are stable per CL, so the replay walk is a
    // deterministic iteration over the set bits of a fixed mask. The
    // NUM_WARPS-wide priority encoder + popcount are kept in the
    // register-input path; only registered control reaches the drain-ready
    // feedback, so smem_wr_ready_internal does not traverse the PE.
    reg  [`VX_CFG_NUM_WARPS-1:0] replay_vec_r;       // active remaining mask
    reg  [MC_NW_BITS-1:0]        replay_next_idx_r;  // PE index of replay_vec_r
    reg  [`VX_CFG_NUM_WARPS-1:0] replay_onehot_r;    // PE onehot of replay_vec_r
    reg                          replay_has_rem_r;   // PE valid of replay_vec_r
    reg                          replay_is_last_r;   // replay_vec_r is one bit
    // Per-receiver SMEM word offset = stride_words × (receivers serviced so far
    // in this word). Maintained as a running accumulator instead of a multiply:
    // +stride_words per serviced receiver, reset at each word boundary. Removes
    // both the popcount and the address-path multiply.
    reg  [SMEM_ADDR_WIDTH-1:0]   beat_offset_r;

    wire [MC_NW_BITS-1:0]  replay_next_idx      = replay_next_idx_r;
    wire                   replay_has_remaining = replay_has_rem_r;
    wire                   replay_is_last       = replay_is_last_r;

    wire is_multicast = fb_x_r.is_multicast;
    wire mc_write_valid = drain_valid && (!is_multicast || replay_has_remaining);
    wire mc_write_fire = mc_write_valid && smem_bus_if.req_ready;
    assign smem_wr_ready_internal = is_multicast
        ? (smem_bus_if.req_ready && (!replay_has_remaining || replay_is_last))
        : smem_bus_if.req_ready;

    // Next-cycle replay mask: clear the bit that fires this beat, reload the
    // mask at word boundaries (after the last receiver) and whenever the
    // engine is idle, so the first beat of any word starts from the full mask
    // without a reload bubble. The mask reloaded is that of the CL in fb next
    // cycle: the one loading now, if any.
    wire [`VX_CFG_NUM_WARPS-1:0] cta_mask = fb_load_now ? pend_x.cta_mask : fb_x_r.cta_mask;
    wire replay_beat_fire = mc_write_fire && replay_has_remaining;
    wire [`VX_CFG_NUM_WARPS-1:0] replay_vec_next =
          ~drain_valid      ? cta_mask                          // idle: keep primed
        : ~replay_beat_fire ? replay_vec_r                      // stalled mid-word
        : replay_is_last    ? cta_mask                          // word done: reload
        :                     (replay_vec_r & ~replay_onehot_r); // advance in word

    // The accumulator mirrors the replay_vec_next state machine: 0 when
    // idle/word-boundary, hold on stall, +stride_words per serviced receiver.
    wire [SMEM_ADDR_WIDTH-1:0] stride_words = SMEM_ADDR_WIDTH'(fb_x_r.stride_words);
    wire [SMEM_ADDR_WIDTH-1:0] beat_offset_next =
          ~drain_valid      ? '0                              // idle: prime to 0
        : ~replay_beat_fire ? beat_offset_r                   // stalled mid-word
        : replay_is_last    ? '0                              // word done: reset
        :                     (beat_offset_r + stride_words); // advance one receiver

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

    always @(posedge clk) begin
        if (reset) begin
            replay_vec_r      <= '0;
            replay_next_idx_r <= '0;
            replay_onehot_r   <= '0;
            replay_has_rem_r  <= 1'b0;
            replay_is_last_r  <= 1'b0;
            beat_offset_r     <= '0;
        end else begin
            replay_vec_r      <= replay_vec_next;
            replay_next_idx_r <= replay_idx_next;
            replay_onehot_r   <= replay_onehot_next;
            replay_has_rem_r  <= replay_hasrem_next;
            replay_is_last_r  <= replay_islast_next;
            beat_offset_r     <= beat_offset_next;
        end
    end

    // Base word addr per beat: km_word_addr in K-major (per-elem scatter),
    // fb_word_addr_r in row-major (streamed). Replay adds the per-receiver
    // offset accumulated in beat_offset_r (VX_dxa_setup rounds smem_stride to
    // the CTA slot pitch, a multiple of LMEM_SLOT_ALIGN, itself a multiple of
    // SMEM_WORD_SIZE, so the byte→word shift in stride_words is lossless).
    `STATIC_ASSERT((LMEM_SLOT_ALIGN % SMEM_WORD_SIZE) == 0,
                   ("the CTA slot pitch must be a whole number of SMEM words"))
    wire [SMEM_ADDR_WIDTH-1:0] base_word_addr = scatter ? km_word_addr : fb_word_addr_r;
    wire [SMEM_ADDR_WIDTH-1:0] replay_addr  = base_word_addr + beat_offset_r;

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
    wire smem_wr_attr_last = fb_x_r.notify_smem_done && (
        is_multicast ? (mc_write_valid && is_last_drain) : smem_wr_last_pkt);
    wire [BAR_ADDR_W-1:0] smem_wr_attr_bar = is_multicast
        ? BAR_ADDR_W'(fb_x_r.bar_addr + (BAR_ADDR_W'(replay_next_idx) << NB_BITS))
        : fb_x_r.bar_addr;

    // ════════════════════════════════════════════════════════════════════
    // SMEM bus wiring
    // ════════════════════════════════════════════════════════════════════
    assign smem_bus_if.req_valid       = smem_wr_valid;
    assign smem_bus_if.req_data.rw     = 1'b1;
    assign smem_bus_if.req_data.addr   = smem_wr_addr;
    assign smem_bus_if.req_data.data   = fb_word_data;
    assign smem_bus_if.req_data.byteen = fb_word_byteen;
    assign smem_bus_if.req_data.attr   = {smem_wr_attr_last, smem_wr_attr_bar};
    assign smem_bus_if.req_data.tag.uuid  = fb_x_r.uuid;   // tag DXA write with its issuing uuid
    assign smem_bus_if.req_data.tag.value = SMEM_TAG_VALUE_W'(fb_x_r.core_id) << ENGINE_VALUE_W;
    assign smem_bus_if.rsp_ready       = 1'b0;

    `UNUSED_VAR (smem_bus_if.rsp_valid)
    `UNUSED_VAR (smem_bus_if.rsp_data)

    assign xfer_done     = smem_req_fire && smem_wr_last_pkt;
    assign xfer_done_idx = fb_xfer_r;

    // Each stage reads only the context fields it needs.
    `UNUSED_VAR (pend_x)
    `UNUSED_VAR (fb_x_r)

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (~reset) begin
            if (sw_accept) begin
                $write("DXA_PIPE,%0d,SW_RX,tag=%0d,xfer=%0d,oob=%0d\n",
                    $time, sw_tag, sw_xfer, sw_oob);
            end
            if (smem_req_fire) begin
                $write("DXA_PIPE,%0d,SMEM_WR,addr=0x%0h,byteen=0x%0h,last=%0d\n",
                    $time, smem_wr_addr, fb_word_byteen, smem_wr_last_pkt);
            end
        end
    end
`endif

endmodule
