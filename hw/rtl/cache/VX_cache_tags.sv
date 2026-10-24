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

`include "VX_cache_define.vh"

// Single-array tag store with per-way write-enable.
//
// All NUM_WAYS tags for a set live in one block-RAM word (read in parallel for
// the hit compare); a per-way write-enable updates a single way on a fill. This
// replaces the previous NUM_WAYS separate tag arrays with one BRAM.
//
// The tag BRAM holds the TAG ONLY. The per-sector valid vector is decoupled into
// a side LUTRAM (like the dirty vector): a sector refill ORs the fetched sector
// into the resident line's valid vector -- a read-modify-write. Done in the tag
// BRAM that RMW puts a tag-read -> compare -> valid-OR -> BRAM-write arc on the
// write-data path; in a per-bit-write LUTRAM the refill is a local per-bit set,
// so the tag BRAM write is just the registered line_tag and never depends on the
// read.

module VX_cache_tags import VX_gpu_pkg::*; #(
    parameter CACHE_SIZE    = 1024,           // Size of cache in bytes
    parameter LINE_SIZE     = 16,             // Size of line inside a bank in bytes
    parameter NUM_BANKS     = 1,              // Number of banks
    parameter NUM_WAYS      = 1,              // Number of associative ways
    parameter WORD_SIZE     = 1,              // Size of a word in bytes
    parameter SECTOR_SIZE   = LINE_SIZE,      // Size of a sector in bytes (fill/eviction granule); = LINE_SIZE => 1 sector
    parameter WRITEBACK     = 0,              // Enable cache writeback
    parameter LAZY_FETCH    = 0,              // Track per-sector partial (allocated-by-store) state
    parameter AMO_ENABLE    = 0               // Enable the AMO-passthrough line invalidate (non-LLC banks only)
) (
    input wire                          clk,
    input wire                          reset,

    // inputs
    input wire                          stall,
    input wire                          init,
    input wire                          flush,
    input wire                          fill,
    input wire                          read,
    input wire                          write,
    input wire                          invalidate, // clear valid on the hit way (qualifies the way's own match)
    // the fill installs no data: the sector becomes partial (cleared otherwise)
    input wire                          fill_nodata,
    // a refetch completes its resident partial sector: clear its partial bit
    input wire                          refetch,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx_n,
    input wire [`CS_TAG_SEL_BITS-1:0]   line_tag,
    input wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx, // requested sector within the line
    input wire [`CS_WAY_SEL_WIDTH-1:0]  evict_way,
    // the way a flush walks: the request's registered way, never a fill's
    // compare-derived target, so the flush's clears stay off the tag read
    input wire [`CS_WAY_SEL_WIDTH-1:0]  flush_way,
    // the way a writeback evicts, for evict_tag: a fill's replacement victim
    // (a refill evicts nothing) or a flush's way -- never the tag compare
    input wire [`CS_WAY_SEL_WIDTH-1:0]  wb_way,
    // fill target way as one-hot, taken from line_present/victim directly:
    // the binary evict_way round trip (encode, mux, per-way compare) put the
    // tag-read cone on this array's own write-enable pins.
    input wire [NUM_WAYS-1:0]           fill_way_oh,

    // outputs
    output wire [NUM_WAYS-1:0]          tag_matches,
    // per-way "line resident" = tag match with any sector valid, ignoring the
    // requested sector. Distinguishes a sector miss (line present, sector
    // invalid -> refill that sector into this way, no eviction) from a line
    // miss (line absent -> allocate a victim way). Equals tag_matches when
    // there is a single sector per line.
    output wire [NUM_WAYS-1:0]          line_present,
    output wire                         evict_dirty,
    // per-sector dirty vector of the evict way (drives the multi-beat per-sector
    // writeback); a single bit == evict_dirty when 1 sector/line.
    output wire [`CS_SECTORS_PER_LINE-1:0] evict_dirty_mask,
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag,
    // the evict way holds a valid sector
    output wire                         evict_valid,
    // the requested sector is resident but partial (a load/AMO to it refetches)
    output wire                         partial_match
);
    //          tag store: tag only   (valid + dirty decoupled into side LUTRAMs)
    localparam SEC = `CS_SECTORS_PER_LINE;
    `UNUSED_VAR (read)

    // one-hot of the requested sector (SEC=1 => constant 1)
    wire [SEC-1:0] sec_oh = SEC'(1) << sector_idx;

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0][SEC-1:0] read_valid;
    wire [NUM_WAYS-1:0][SEC-1:0] read_dirty;
    wire [NUM_WAYS-1:0][SEC-1:0] read_partial;
    wire [NUM_WAYS-1:0]          partial_hit;

    assign evict_valid = (| read_valid[evict_way]);

    if (WRITEBACK) begin : g_evict_tag_wb
        assign evict_dirty = (| read_dirty[evict_way]); // dirty if any sector dirty
        assign evict_dirty_mask = read_dirty[evict_way];
        assign evict_tag = read_tag[wb_way];
    end else begin : g_evict_tag_wt
        `UNUSED_VAR (read_dirty)
        `UNUSED_VAR (read_tag)
        `UNUSED_VAR (wb_way)
        assign evict_dirty = 1'b0;
        assign evict_dirty_mask = '0;
        assign evict_tag = '0;
    end

    // Per-way decoded write strobes and payloads. At most one operation type
    // fires per cycle (input arbitration in the bank). The tag store holds the
    // tag only and is written only on a fill, so its write never depends on
    // the tag compare.
    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] line_wdata;
    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] tag_rdata;

    // Decoupled per-sector valid store: NUM_WAYS*SEC bits/set in LUTRAM, per-bit
    // write-enable. A sector refill sets its bit locally instead of ORing through
    // the tag BRAM. Placed next to the compare so its route collapses.
    wire [NUM_WAYS-1:0][SEC-1:0] valid_wren;
    wire [NUM_WAYS-1:0][SEC-1:0] valid_wdata;
    wire [NUM_WAYS-1:0][SEC-1:0] valid_rdata;

    // Decoupled per-sector dirty store (writeback only): NUM_WAYS*SEC bits/set
    // in LUTRAM, per-bit write-enable. Placed locally next to the compare so its
    // route collapses; keeps the wide tag-compare->dirty-set loop off the tag BRAM.
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_wren;
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_wdata;
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_rdata;

    // Per-sector partial store (lazy fetch only): a fill writes its sector's
    // bit, set by a data-less fill and cleared by a data fill. The bit is only
    // meaningful while the sector is valid, so no other operation touches it.
    wire [NUM_WAYS-1:0][SEC-1:0] partial_wren;
    wire [NUM_WAYS-1:0][SEC-1:0] partial_wdata;
    wire [NUM_WAYS-1:0][SEC-1:0] partial_rdata;

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_way_decode
        wire way_en   = (NUM_WAYS == 1) || (flush_way == i);
        wire do_init  = init; // init all ways
        // The fill way select comes in one-hot (line_present or the registered
        // victim decode); the binary way_en compare stays only for flush, whose
        // way is the registered st0 way_idx and never this cycle's tag read.
        wire do_fill  = fill && ((NUM_WAYS == 1) || fill_way_oh[i]);
        wire do_flush = flush && (!WRITEBACK || way_en); // flush all ways in writethrough mode
        wire do_write = WRITEBACK && write && tag_matches[i]; // only write on tag hit

        // Tag store: the tag changes only on a fill (a refill rewrites the
        // identical tag, harmless). init/flush/invalidate only touch the valid
        // vector (below), never the tag. The fill way follows the tag compare
        // (a refill targets the resident way), so a fill rewrites every way --
        // the filled one with line_tag, the others with their current tag --
        // and the compare reaches the array's data, not its write enables.
        assign line_wdata[i] = do_fill ? line_tag : read_tag[i];

        // Read-First BRAM: a fill committed on the previous cycle isn't yet in
        // the readout. The bypass is keyed on the SET and the FILLED WAY: that
        // way's stale entry is replaced with the filled line's {tag, valid},
        // so a same-set request neither false-hits the just-evicted line nor
        // misses the just-installed one. The bypass must HOLD across a pipe
        // stall (gated by ~stall): when a fill is followed by a dependent
        // replay and the pipe stalls in between (e.g. a multi-beat per-sector
        // writeback), the held tag readout still misses the fill, so a plain
        // 1-cycle buffer would expire mid-stall and the replay would
        // spuriously miss the just-filled line.
        wire rdw_fill_raw;
        wire [SEC-1:0] rdw_sec_oh;
        wire [`CS_TAG_SEL_BITS-1:0]  rdw_tag;
        wire [`CS_LINE_SEL_BITS-1:0] rdw_set;
        wire rdw_refill;
        `BUFFER_EX(rdw_fill_raw, do_fill, ~stall, 1, 1);
        `BUFFER_EX(rdw_sec_oh, sec_oh, ~stall, $bits(rdw_sec_oh), 1);
        `BUFFER_EX(rdw_tag, line_tag, ~stall, $bits(rdw_tag), 1);
        `BUFFER_EX(rdw_set, line_idx, ~stall, $bits(rdw_set), 1);
        // do_fill is way-gated, so rdw_fill_raw identifies the filled way.
        wire way_filled = rdw_fill_raw && (line_idx == rdw_set);

        // Tag equality with the bypass folded in: both compares run in parallel
        // and way_filled selects the result, so the set compare never sits in
        // series with the wide tag compare on the tag->data write-enable arc.
        wire tag_eq = way_filled ? (line_tag == rdw_tag) : (line_tag == tag_rdata[i]);

        // hit: tag match AND the requested sector is valid. read_valid folds in
        // a same-set fill from the previous cycle (see the bypass above), so
        // this compare is exact even on the fill's write edge. Whether a partial
        // sector serves the lookup is decided outside, off the array enables.
        wire sector_valid = read_valid[i][sector_idx] && tag_eq;
        wire sector_partial = (LAZY_FETCH != 0) && read_partial[i][sector_idx];
        wire raw_hit = sector_valid;
        assign partial_hit[i] = sector_valid && sector_partial;
        // AMO passthrough invalidate: clear the requested sector's valid.
        wire do_inval = (AMO_ENABLE != 0) && invalidate && raw_hit;

        // A fill into a way that already holds this line (a sector refill) sets
        // the fetched sector in the existing valid vector; a fill into a fresh
        // victim way installs only the fetched sector (others cleared). With
        // 1 sector/line a fill is always to a fresh way.
        wire fill_refill = do_fill && tag_eq && (| read_valid[i]);
        `BUFFER_EX(rdw_refill, do_fill && fill_refill, ~stall, 1, 1);

        // Per-sector valid update as per-bit set/clear for the LUTRAM (no
        // read-modify-write): set the filled sector; clear the whole line on
        // init/flush, the other sectors on a fresh victim install, the requested
        // sector on invalidate.
        // The fill way follows the tag compare (a refill targets the resident
        // way), so a fill rewrites every way's valid bits -- the filled way
        // with its update, the others with their current bits -- and the
        // compare reaches the store's data, not its write enables. The current
        // bits are read_valid, which already carries last cycle's fill, with
        // last cycle's other same-set clears applied here, off the hit path.
        wire [SEC-1:0] vclr_op = ((do_init || do_flush) ? {SEC{1'b1}} : {SEC{1'b0}})
                               | (do_inval              ? sec_oh      : {SEC{1'b0}});
        wire same_set_v = (line_idx == line_idx_n);
        wire [SEC-1:0] rdw_vclr;
        `BUFFER_EX(rdw_vclr, (same_set_v ? vclr_op : {SEC{1'b0}}), ~stall, SEC, 1);
        wire [SEC-1:0] cur_valid = read_valid[i] & ~rdw_vclr;
        wire [SEC-1:0] fill_valid = do_fill ? (fill_refill ? (cur_valid | sec_oh) : sec_oh) : cur_valid;
        assign valid_wren[i]  = (fill ? {SEC{1'b1}} : {SEC{1'b0}}) | vclr_op;
        assign valid_wdata[i] = fill ? fill_valid : {SEC{1'b0}};

        // Tag from the tag BRAM, valid from the decoupled LUTRAM. Both share the
        // same look-ahead read / read-first access pattern, so the way_filled
        // bypass corrects the just-filled line for both. A fresh fill REPLACED
        // this way's entry; a refill set its sector in the resident line's vector.
        wire [SEC-1:0] rdata_valid = valid_rdata[i];
        assign read_tag[i]   = way_filled ? rdw_tag : tag_rdata[i];
        assign read_valid[i] = way_filled ? (rdw_refill ? (rdata_valid | rdw_sec_oh) : rdw_sec_oh)
                                          : rdata_valid;

        // ---- decoupled per-sector dirty (writeback only) ----
        if (WRITEBACK) begin : g_dirty
            // set the written sector on a write hit; clear sectors on
            // fill/inval; clear all on init/flush or a fresh (evicting) fill.
            // The hit way of a write and the target way of a fill follow the
            // tag compare, so those operations rewrite every way -- the target
            // with its update, the others with their current bits (read_dirty
            // is exact: it forwards last cycle's update) -- and the compare
            // reaches the store's data, not its write enables.
            wire           write_op = WRITEBACK && write;
            wire [SEC-1:0] dset_oh  = do_write ? sec_oh : {SEC{1'b0}};
            wire           dclr_all = do_init || do_flush || (do_fill && ~fill_refill);
            wire [SEC-1:0] dclr_oh  = (do_inval ? sec_oh : {SEC{1'b0}})
                                    | ((do_fill && fill_refill) ? sec_oh : {SEC{1'b0}});
            wire [SEC-1:0] fill_keep = do_fill ? (fill_refill ? (read_dirty[i] & ~sec_oh) : {SEC{1'b0}})
                                               : read_dirty[i];
            assign dirty_wren[i]  = (write_op ? sec_oh : {SEC{1'b0}})
                                  | {SEC{fill || do_init || do_flush}}
                                  | (do_inval ? sec_oh : {SEC{1'b0}});
            assign dirty_wdata[i] = fill ? fill_keep
                                  : ((write_op ? read_dirty[i] : {SEC{1'b0}}) | dset_oh); // 0 on any clear

            // Read-first forwarding for the dirty LUTRAM (same-set, 1 cycle, held
            // across stall): a dirty update to this set last cycle isn't in the
            // readout when the next request reads the same set. Clear wins over
            // the stale readout; set wins over clear (mutually exclusive ops).
            wire same_set = (line_idx == line_idx_n);
            wire [SEC-1:0] rdw_dset, rdw_dclr;
            wire rdw_dclr_all;
            `BUFFER_EX(rdw_dset, (same_set ? dset_oh : {SEC{1'b0}}), ~stall, SEC, 1);
            `BUFFER_EX(rdw_dclr, (same_set ? dclr_oh : {SEC{1'b0}}), ~stall, SEC, 1);
            `BUFFER_EX(rdw_dclr_all, same_set && dclr_all, ~stall, 1, 1);
            assign read_dirty[i] = (dirty_rdata[i] & ~rdw_dclr & ~{SEC{rdw_dclr_all}}) | rdw_dset;
        end else begin : g_no_dirty
            `UNUSED_VAR (do_write)
            assign dirty_wren[i]  = '0;
            assign dirty_wdata[i] = '0;
            assign read_dirty[i]  = {SEC{1'b0}};
        end

        if (LAZY_FETCH != 0) begin : g_partial
            // Same access pattern as the valid store, so the way_filled bypass
            // corrects a same-set fill from the previous cycle the same way. A
            // refetch's clear needs none: nothing is looked up right behind it.
            wire rdw_nodata;
            `BUFFER_EX(rdw_nodata, fill_nodata, ~stall, 1, 1);
            wire do_refetch = refetch && sector_valid;
            assign partial_wren[i]  = (do_fill || do_refetch) ? sec_oh : {SEC{1'b0}};
            assign partial_wdata[i] = {SEC{do_fill && fill_nodata}};
            assign read_partial[i]  = way_filled ? ((partial_rdata[i] & ~rdw_sec_oh) | (rdw_sec_oh & {SEC{rdw_nodata}}))
                                                 : partial_rdata[i];
        end else begin : g_no_partial
            assign partial_wren[i]  = '0;
            assign partial_wdata[i] = '0;
            assign read_partial[i]  = {SEC{1'b0}};
        end

        assign tag_matches[i] = raw_hit;
        // line resident in this way: tag matches and at least one sector valid
        // (a same-cycle-prior fill is folded into tag_eq/read_valid).
        assign line_present[i] = tag_eq && (| read_valid[i]);
    end

    // Single tag array: one BRAM word holds all ways' tags; per-way write-enable
    // updates a single way on a fill. Read at line_idx_n (one cycle ahead),
    // written at line_idx, read-first to match the fill/replay ordering.
    VX_dp_ram #(
        .DATAW (NUM_WAYS * `CS_TAG_SEL_BITS),
        .SIZE  (`CS_LINES_PER_BANK),
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (~stall),
        // a fill writes the whole set's tags (see line_wdata)
        .write (fill),
        .wren  (1'b1),
        .waddr (line_idx),
        .raddr (line_idx_n),
        .wdata (line_wdata),
        .rdata (tag_rdata)
    );

    // The side stores' array enables are the operations that may write them,
    // never the OR of their per-bit enables: those carry the tag compare, which
    // would otherwise fan into every LUTRAM write enable. A bit still changes
    // only where its own enable is set.
    wire valid_write   = init || flush || fill || ((AMO_ENABLE != 0) && invalidate);
    wire dirty_write   = init || flush || fill || write || ((AMO_ENABLE != 0) && invalidate);
    wire partial_write = fill || refetch;

    // Decoupled per-sector valid store: mirrors the tag BRAM's access pattern
    // (look-ahead read, read-first) so pipeline alignment is identical, with
    // per-bit write-enable so a sector refill sets its bit without reading back.
    VX_dp_ram #(
        .DATAW    (NUM_WAYS * SEC),
        .WRENW    (NUM_WAYS * SEC),
        .SIZE     (`CS_LINES_PER_BANK),
        .OUT_REG  (1),
        .LUTRAM   (1),
        .RDW_MODE ("R")
    ) valid_store (
        .clk   (clk),
        .reset (reset),
        .read  (~stall),
        .write (valid_write),
        .wren  (valid_wren),
        .waddr (line_idx),
        .raddr (line_idx_n),
        .wdata (valid_wdata),
        .rdata (valid_rdata)
    );

    // Decoupled per-sector dirty store (writeback only). Mirrors the tag store's
    // access pattern (look-ahead read, read-first) so pipeline alignment is
    // identical; 1-bit/way/sector LUTRAM keeps it off the tag BRAM write path.
    if (WRITEBACK) begin : g_dirty_store
        VX_dp_ram #(
            .DATAW    (NUM_WAYS * SEC),
            .WRENW    (NUM_WAYS * SEC),
            .SIZE     (`CS_LINES_PER_BANK),
            .OUT_REG  (1),
            .LUTRAM   (1),
            .RDW_MODE ("R")
        ) dirty_store (
            .clk   (clk),
            .reset (reset),
            .read  (~stall),
            .write (dirty_write),
            .wren  (dirty_wren),
            .waddr (line_idx),
            .raddr (line_idx_n),
            .wdata (dirty_wdata),
            .rdata (dirty_rdata)
        );
    end else begin : g_no_dirty_store
        assign dirty_rdata = '0;
        `UNUSED_VAR ({dirty_wren, dirty_wdata, dirty_rdata, dirty_write})
    end

    if (LAZY_FETCH != 0) begin : g_partial_store
        VX_dp_ram #(
            .DATAW    (NUM_WAYS * SEC),
            .WRENW    (NUM_WAYS * SEC),
            .SIZE     (`CS_LINES_PER_BANK),
            .OUT_REG  (1),
            .LUTRAM   (1),
            .RDW_MODE ("R")
        ) partial_store (
            .clk   (clk),
            .reset (reset),
            .read  (~stall),
            .write (partial_write),
            .wren  (partial_wren),
            .waddr (line_idx),
            .raddr (line_idx_n),
            .wdata (partial_wdata),
            .rdata (partial_rdata)
        );
        assign partial_match = (| partial_hit);
    end else begin : g_no_partial_store
        assign partial_rdata = '0;
        assign partial_match = 1'b0;
        `UNUSED_VAR ({partial_wren, partial_wdata, partial_rdata, partial_hit, read_partial, fill_nodata, refetch, partial_write})
    end

endmodule
