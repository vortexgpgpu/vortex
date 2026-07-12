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

// VX_om_ingress — turns aperture write beats back into om_bus fragment requests.
//
// One instance per OM steer (i.e. per socket->L2 trunk input). It reconstructs
// what the old dedicated om bus used to carry -- {pos, colour, depth, face} --
// from a plain memory write, and feeds the existing VX_om_bus_arb -> VX_om_core
// unchanged.
//
// APERTURE ENCODING (shift-only, deliberately):
//     offset = ((face << (XBITS+YBITS)) | (y << XBITS) | x) << RECORD_SHIFT
// The framebuffer pitch is padded to a power of two, so decoding an offset back
// into (face, y, x) is pure bit-slicing. A packed `y*fb_width + x` would need a
// DIVIDER here. The aperture is virtual -- nothing is stored there -- so the
// address space the padding wastes costs nothing.
//
// RECORD SHAPE: a shader emits colour only (early-Z owns depth), depth only
// (z-prepass), or both (gl_FragDepth). RECORD_SHIFT is 2 for the one-word modes
// and 3 for the two-word mode; the DCR tells us which, because it is a property
// of the draw, not of the beat.
//
// PAIRING (two-word mode only): the colour beat is held until its depth beat
// arrives, then one om_bus request fires. The hold table is sized so it CANNOT
// fill -- see MAX_OPEN below. That is the deadlock argument: if an allocating
// beat can never be refused, a completing beat can never be stuck behind one.

`include "VX_om_define.vh"

module VX_om_ingress import VX_gpu_pkg::*, VX_om_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter DATA_SIZE  = 1,
    parameter TAG_WIDTH  = 1,
    parameter NUM_LANES  = 1
) (
    input wire clk,
    input wire reset,

    input om_dcrs_t dcrs,

    VX_mem_bus_if.slave  mem_bus_if,   // aperture writes, peeled by VX_om_steer
    VX_om_bus_if.master  om_bus_if,    // fragments -> VX_om_bus_arb -> VX_om_core

    output wire busy
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (TAG_WIDTH)
    // Only the aperture fields are read here; the rest of the OM state belongs to
    // VX_om_core (buffer addresses, blend, stencil, depth func).
    `UNUSED_VAR (dcrs)

    localparam WORD_SIZE      = 4;
    localparam WORDS_PER_LINE = DATA_SIZE / WORD_SIZE;
    localparam WSEL_BITS      = `CLOG2(WORDS_PER_LINE);
    localparam LINE_ADDRW     = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(DATA_SIZE);

    // A colour beat stays open until its depth beat lands. The open set must be
    // bounded, and the table sized to that bound, or the pairing is not
    // deadlock-free: if an allocating beat can never be refused, a completing beat
    // can never be stuck behind one.
    //
    // The bound is per LANE, not per port. A store uop fans out across EVERY lane
    // of the warp, so the colour uop's beats for all lanes reach the ingress
    // before the depth uop emits any -- one open record per lane, not one per
    // dcache port. (An earlier attempt sized this by dcache ports and the overflow
    // assert below caught it immediately.)
    //
    // Per lane the open count is exactly one: fu_lock means a core has at most one
    // export burst in flight, and per-lane program order closes each record before
    // the next one opens. So the bound is every lane of every core that can reach
    // this trunk -- taken conservatively, without assuming lanes spread evenly
    // across the L1 mem ports.
    localparam MAX_OPEN  = `VX_CFG_SOCKET_SIZE * `VX_CFG_NUM_THREADS;
    localparam OPEN_IDXW = `UP(`CLOG2(MAX_OPEN));

    // ── beat decode ────────────────────────────────────────────────────────
    wire [LINE_ADDRW-1:0] line_addr = mem_bus_if.req_data.addr;

    // The NC/bypass path widens a word write into a line write whose byteen
    // selects the word, so recover the word index from the byteen.
    wire [WORDS_PER_LINE-1:0] word_en;
    for (genvar w = 0; w < WORDS_PER_LINE; ++w) begin : g_word_en
        assign word_en[w] = |mem_bus_if.req_data.byteen[w * WORD_SIZE +: WORD_SIZE];
    end

    wire [WSEL_BITS-1:0] wsel;
    VX_priority_encoder #(
        .N (WORDS_PER_LINE)
    ) wsel_enc (
        .data_in    (word_en),
        `UNUSED_PIN (onehot_out),
        .index_out  (wsel),
        `UNUSED_PIN (valid_out)
    );

    wire [`VX_CFG_XLEN-1:0] byte_addr =
        (`VX_CFG_XLEN'(line_addr) << `CLOG2(DATA_SIZE)) | (`VX_CFG_XLEN'(wsel) << 2);

    wire [`VX_CFG_XLEN-1:0] aperture_off = byte_addr - `VX_CFG_XLEN'(`VX_MEM_OM_BASE_ADDR);

    // two-word record (colour + depth) vs one-word (colour only, or depth only)
    wire is_pair = (dcrs.aperture_record_shift == 2'd3);

    // Which word of the record this beat carries. In one-word mode the shader
    // emits exactly one of colour/depth, and the DCR says which.
    wire beat_is_depth = is_pair ? aperture_off[2] : dcrs.aperture_depth_only;

    wire [`VX_CFG_XLEN-1:0] record_idx = aperture_off >> dcrs.aperture_record_shift;

    // Zero-extend before slicing: py's base is a DCR, so the window can run past
    // the top of record_idx on a narrow framebuffer.
    localparam REC_EXT_W = `VX_CFG_XLEN + `VX_OM_DIM_BITS;
    wire [REC_EXT_W-1:0] rec_ext = REC_EXT_W'(record_idx);

    wire [31:0] xb = 32'(dcrs.aperture_xbits);
    wire [31:0] yb = 32'(dcrs.aperture_ybits);

    wire [`VX_OM_DIM_BITS-1:0] px   = rec_ext[0 +: `VX_OM_DIM_BITS];
    wire [`VX_OM_DIM_BITS-1:0] py   = rec_ext[xb +: `VX_OM_DIM_BITS];
    wire                       face = rec_ext[xb + yb];

    wire [31:0] wsel_ext = 32'(wsel);
    wire [31:0] beat_word = mem_bus_if.req_data.data[wsel_ext * 32 +: 32];

    wire beat_fire = mem_bus_if.req_valid && mem_bus_if.req_ready;

    // ── the hold table (two-word mode) ─────────────────────────────────────
    reg [MAX_OPEN-1:0]                       open_valid;
    reg [MAX_OPEN-1:0][`VX_CFG_XLEN-1:0]     open_rec;
    reg [MAX_OPEN-1:0][31:0]                 open_colour;

    // match this beat against an open colour record
    wire [MAX_OPEN-1:0] hit_vec;
    for (genvar i = 0; i < MAX_OPEN; ++i) begin : g_hit
        assign hit_vec[i] = open_valid[i] && (open_rec[i] == record_idx);
    end
    wire hit = |hit_vec;

    wire [OPEN_IDXW-1:0] hit_idx;
    VX_priority_encoder #(
        .N (MAX_OPEN)
    ) hit_enc (
        .data_in    (hit_vec),
        `UNUSED_PIN (onehot_out),
        .index_out  (hit_idx),
        `UNUSED_PIN (valid_out)
    );

    wire [OPEN_IDXW-1:0] free_idx;
    wire                 has_free;
    VX_priority_encoder #(
        .N (MAX_OPEN)
    ) free_enc (
        .data_in    (~open_valid),
        `UNUSED_PIN (onehot_out),
        .index_out  (free_idx),
        .valid_out  (has_free)
    );

    // In two-word mode a colour beat only ALLOCATES; a depth beat COMPLETES the
    // record it matches and fires the fragment.
    wire alloc = is_pair && ~beat_is_depth;
    wire fire  = ~is_pair || (beat_is_depth && hit);

    // ── the om_bus request ─────────────────────────────────────────────────
    // One fragment per beat: the mask is one-hot on lane 0, where the old bus
    // carried NUM_LANES fragments per request. That is NOT a fill-rate loss, and
    // the arithmetic is worth writing down because the width difference looks
    // alarming until you check what the OM can actually drain:
    //
    //   OCACHE_NUM_BANKS = 1        => OCACHE_NUM_REQS = 1
    //   OM_MEM_REQS      = 2 * NUM_SFU_LANES   (a colour and a depth access/lane)
    //
    // so an N-lane request is arbitrated into a single bank and takes >= 2N cycles
    // to drain inside VX_om_core. The OM eats ~0.5 fragments/cycle no matter how
    // wide you feed it -- the old ~2,860-bit bus was over-provisioned by ~64x.
    // L2_SOCKET_REQS ingresses running in parallel supply far more than that.
    //
    // The fill-rate lever is OCACHE_NUM_BANKS, not the transport.
    wire [31:0] frag_colour = beat_is_depth ? open_colour[hit_idx] : beat_word;
    wire [31:0] frag_depth  = beat_is_depth ? beat_word : 32'd0;
    `UNUSED_VAR (frag_depth)   // only the low VX_OM_DEPTH_BITS reach the om_bus

    wire om_ready;
    wire [NUM_LANES-1:0] frag_mask;
    assign frag_mask = NUM_LANES'(1);

    VX_elastic_buffer #(
        .DATAW (UUID_WIDTH + NUM_LANES * (1 + 2 * `VX_OM_DIM_BITS + 32 + `VX_OM_DEPTH_BITS + 1)),
        .SIZE  (2)
    ) req_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_bus_if.req_valid && fire),
        .ready_in  (om_ready),
        .data_in   ({mem_bus_if.req_data.tag.uuid,
                     frag_mask,
                     {NUM_LANES{px}},
                     {NUM_LANES{py}},
                     {NUM_LANES{frag_colour}},
                     {NUM_LANES{frag_depth[`VX_OM_DEPTH_BITS-1:0]}},
                     {NUM_LANES{face}}}),
        .data_out  ({om_bus_if.req_data.uuid, om_bus_if.req_data.mask,
                     om_bus_if.req_data.pos_x, om_bus_if.req_data.pos_y,
                     om_bus_if.req_data.color, om_bus_if.req_data.depth,
                     om_bus_if.req_data.face}),
        .valid_out (om_bus_if.req_valid),
        .ready_out (om_bus_if.req_ready)
    );

    // An allocating beat is never refused (the table cannot fill), so the only
    // back-pressure is the om_bus itself.
    assign mem_bus_if.req_ready = fire ? om_ready : 1'b1;

    always @(posedge clk) begin
        if (reset) begin
            open_valid <= '0;
        end else begin
            if (beat_fire && alloc) begin
                open_valid[free_idx]  <= 1'b1;
                open_rec[free_idx]    <= record_idx;
                open_colour[free_idx] <= beat_word;
            end
            if (beat_fire && is_pair && beat_is_depth && hit) begin
                open_valid[hit_idx] <= 1'b0;
            end
        end
    end

    assign busy = (open_valid != '0) || om_bus_if.req_valid;

    `RUNTIME_ASSERT(~(beat_fire && alloc && ~has_free),
        ("%t: *** %s: OM ingress hold table overflow — MAX_OPEN bound is wrong", $time, INSTANCE_ID))
    `RUNTIME_ASSERT(~(beat_fire && is_pair && beat_is_depth && ~hit),
        ("%t: *** %s: depth beat with no open colour record — the fu_lock pair was split", $time, INSTANCE_ID))

    `UNUSED_VAR (mem_bus_if.req_data.rw)
    `UNUSED_VAR (mem_bus_if.req_data.attr)
    `UNUSED_VAR (mem_bus_if.rsp_ready)
    assign mem_bus_if.rsp_valid = 1'b0;
    assign mem_bus_if.rsp_data  = '0;

endmodule
