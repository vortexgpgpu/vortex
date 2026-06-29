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
// the hit compare); a per-way write-enable updates a single way on a
// fill/write/invalidate without a read-modify-write. This replaces the
// previous NUM_WAYS separate tag arrays with one BRAM.

module VX_cache_tags import VX_gpu_pkg::*; #(
    // Size of cache in bytes
    parameter CACHE_SIZE    = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE     = 16,
    // Number of banks
    parameter NUM_BANKS     = 1,
    // Number of associative ways
    parameter NUM_WAYS      = 1,
    // Size of a word in bytes
    parameter WORD_SIZE     = 1,
    // Size of a sector in bytes (fill/eviction granule); = LINE_SIZE => 1 sector
    parameter SECTOR_SIZE   = LINE_SIZE,
    // Enable cache writeback
    parameter WRITEBACK     = 0,
    // Enable the AMO-passthrough line invalidate (non-LLC banks only)
    parameter AMO_ENABLE    = 0
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
    input wire                          invalidate, // clear valid on the hit way
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx_n,
    input wire [`CS_TAG_SEL_BITS-1:0]   line_tag,
    input wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx, // requested sector within the line
    input wire [`CS_WAY_SEL_WIDTH-1:0]  evict_way,

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
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag
);
    //          tag store: valid[SEC], tag   (dirty decoupled into a side LUTRAM)
    localparam SEC        = `CS_SECTORS_PER_LINE;
    localparam TAG_ENTRYW = SEC + `CS_TAG_SEL_BITS;
    `UNUSED_VAR (read)

    // one-hot of the requested sector (SEC=1 => constant 1)
    wire [SEC-1:0] sec_oh = SEC'(1) << sector_idx;

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0][SEC-1:0] read_valid;
    wire [NUM_WAYS-1:0][SEC-1:0] read_dirty;

    if (WRITEBACK) begin : g_evict_tag_wb
        assign evict_dirty = (| read_dirty[evict_way]); // dirty if any sector dirty
        assign evict_dirty_mask = read_dirty[evict_way];
        assign evict_tag = read_tag[evict_way];
    end else begin : g_evict_tag_wt
        `UNUSED_VAR (read_dirty)
        assign evict_dirty = 1'b0;
        assign evict_dirty_mask = '0;
        assign evict_tag = '0;
    end

    // Per-way decoded write strobes and write payloads. At most one operation
    // type fires per cycle (input arbitration in the bank). The tag store holds
    // valid[SEC]+tag only; per-sector dirty lives in a decoupled LUTRAM so the
    // tag-store write-enable does not depend on the write-hit tag compare.
    wire [NUM_WAYS-1:0] line_write;
    wire [NUM_WAYS-1:0][TAG_ENTRYW-1:0] line_wdata;
    wire [NUM_WAYS-1:0][TAG_ENTRYW-1:0] tag_rdata;

    // Decoupled per-sector dirty store (writeback only): NUM_WAYS*SEC bits/set
    // in LUTRAM, per-bit write-enable. Placed locally next to the compare so its
    // route collapses; keeps the wide tag-compare->dirty-set loop off the tag BRAM.
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_wren;
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_wdata;
    wire [NUM_WAYS-1:0][SEC-1:0] dirty_rdata;

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_way_decode
        // raw hit: tag match AND the requested sector is valid (a just-filled
        // line is caught by rdw_fill in tag_matches below since its tag readout
        // is still stale this cycle).
        wire raw_hit = read_valid[i][sector_idx] && (line_tag == read_tag[i]);

        wire way_en   = (NUM_WAYS == 1) || (evict_way == i);
        wire do_init  = init; // init all ways
        wire do_fill  = fill && way_en;
        wire do_flush = flush && (!WRITEBACK || way_en); // flush all ways in writethrough mode
        wire do_write = WRITEBACK && write && tag_matches[i]; // only write on tag hit
        // AMO passthrough invalidate: clear the requested sector's valid.
        wire do_inval = (AMO_ENABLE != 0) && invalidate && raw_hit;

        // A write hit changes neither the tag nor the valid vector, so it does
        // NOT write the tag store (only the dirty LUTRAM). The tag wren therefore
        // depends only on init/fill/flush + the AMO-only invalidate.
        assign line_write[i] = do_init || do_fill || do_flush || do_inval;

        // A fill into a way that already holds this line (a sector refill) ORs
        // the fetched sector into the existing valid vector; a fill into a fresh
        // victim way installs only the fetched sector. With 1 sector/line a fill
        // is always to a fresh way, so this is the legacy reset behavior.
        wire fill_refill = do_fill && (line_tag == read_tag[i]) && (| read_valid[i]);

        // Per-sector valid merge. read_valid is the current line's vector
        // (includes a same-cycle-prior fill via rdw_fill below), so fill/inval
        // preserve other sectors.
        wire [SEC-1:0] valid_wr =
              (do_init || do_flush) ? {SEC{1'b0}}
            :  do_inval             ? (read_valid[i] & ~sec_oh)
            :  do_fill              ? (fill_refill ? (read_valid[i] | sec_oh) : sec_oh)
            :                          read_valid[i];
        assign line_wdata[i] = {valid_wr, line_tag};

        // Read-First BRAM: a same-line fill issued last cycle isn't yet in the
        // readout — OR its (buffered) sector effect back in. The bypass must
        // HOLD across a pipe stall (gated by ~stall): when a fill is followed by
        // a dependent replay and the pipe stalls in between (e.g. a multi-beat
        // per-sector writeback), the held tag readout still misses the fill, so
        // a plain 1-cycle buffer would expire mid-stall and the replay would
        // spuriously miss the just-filled line.
        wire rdw_fill_raw;
        wire [SEC-1:0] rdw_sec_oh;
        wire [`CS_TAG_SEL_BITS-1:0]  rdw_tag;
        wire [`CS_LINE_SEL_BITS-1:0] rdw_set;
        wire rdw_refill;
        `BUFFER_EX(rdw_fill_raw, do_fill, ~stall, 1, 1);
        `BUFFER_EX(rdw_refill, do_fill && fill_refill, ~stall, 1, 1);
        `BUFFER_EX(rdw_sec_oh, sec_oh, ~stall, $bits(rdw_sec_oh), 1);
        `BUFFER_EX(rdw_tag, line_tag, ~stall, $bits(rdw_tag), 1);
        `BUFFER_EX(rdw_set, line_idx, ~stall, $bits(rdw_set), 1);
        // The just-filled way only aliases the CURRENT request when both the set
        // and tag match the line that was filled; otherwise the stale readout
        // belongs to a different line and must NOT be bypassed. Without this gate
        // a back-to-back fill to the same set would be mistaken for a sector-
        // refill into this way, skipping the resident line's dirty eviction.
        wire rdw_fill = rdw_fill_raw && (line_tag == rdw_tag) && (line_idx == rdw_set);
        // A fresh (non-refill) fill last cycle REPLACED the way's valid vector
        // with just its sector; the stale read-first readout still shows the
        // evicted line. Override valid with exactly its sector on a fresh fill;
        // for a refill, OR the new sector in.
        wire rdw_fresh = rdw_fill && ~rdw_refill;

        wire [TAG_ENTRYW-1:0] rdata_i = tag_rdata[i];
        wire [SEC-1:0] rdata_valid = rdata_i[`CS_TAG_SEL_BITS +: SEC];
        assign read_tag[i]   = rdata_i[0 +: `CS_TAG_SEL_BITS];
        assign read_valid[i] = rdw_fresh ? rdw_sec_oh
                             : (rdata_valid | (rdw_fill ? rdw_sec_oh : {SEC{1'b0}}));

        // ---- decoupled per-sector dirty (writeback only) ----
        if (WRITEBACK) begin : g_dirty
            // set the written sector on a write hit; clear sectors on
            // fill/inval; clear all on init/flush or a fresh (evicting) fill.
            wire [SEC-1:0] dset_oh  = do_write ? sec_oh : {SEC{1'b0}};
            wire           dclr_all = do_init || do_flush || (do_fill && ~fill_refill);
            wire [SEC-1:0] dclr_oh  = (do_inval ? sec_oh : {SEC{1'b0}})
                                    | ((do_fill && fill_refill) ? sec_oh : {SEC{1'b0}});
            assign dirty_wren[i]  = dset_oh | dclr_oh | {SEC{dclr_all}};
            assign dirty_wdata[i] = dset_oh; // 1 only where setting; 0 on any clear

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

        assign tag_matches[i] = raw_hit || rdw_fill;
        // line resident in this way: tag matches and at least one sector valid
        // (a same-cycle-prior fill is folded in via rdw_fill).
        assign line_present[i] = ((line_tag == read_tag[i]) && (| read_valid[i])) || rdw_fill;
    end

    // Single tag array: one BRAM word holds all ways' {valid[SEC], tag}; per-way
    // write-enable updates a single way. Read at line_idx_n (one cycle ahead),
    // written at line_idx, read-first to match the fill/replay ordering.
    VX_dp_ram #(
        .DATAW (NUM_WAYS * TAG_ENTRYW),
        .WRENW (NUM_WAYS),
        .SIZE  (`CS_LINES_PER_BANK),
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (~stall),
        .write (| line_write),
        .wren  (line_write),
        .waddr (line_idx),
        .raddr (line_idx_n),
        .wdata (line_wdata),
        .rdata (tag_rdata)
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
            .write (| dirty_wren),
            .wren  (dirty_wren),
            .waddr (line_idx),
            .raddr (line_idx_n),
            .wdata (dirty_wdata),
            .rdata (dirty_rdata)
        );
    end else begin : g_no_dirty_store
        assign dirty_rdata = '0;
        `UNUSED_VAR ({dirty_wren, dirty_wdata, dirty_rdata})
    end

endmodule
