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

module VX_cache_bank import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0,
    parameter NUM_REQS          = 1,
    parameter CACHE_SIZE        = 1024,     // cache size in bytes
    parameter LINE_SIZE         = 16,       // line size in bytes
    parameter NUM_BANKS         = 1,
    parameter NUM_WAYS          = 1,
    parameter WORD_SIZE         = 4,        // word size in bytes
    parameter SECTOR_SIZE       = LINE_SIZE,// sector (fill/eviction granule); = LINE_SIZE => 1 sector
    parameter CRSQ_SIZE         = 1,        // core response queue size
    parameter MSHR_SIZE         = 1,        // miss reservation queue size
    parameter MRSQ_SIZE         = 1,        // memory response queue size (sized at wrapper)
    parameter MREQ_SIZE         = 1,        // memory request queue size
    parameter WRITE_ENABLE      = 1,
    parameter WRITEBACK         = 0,
    parameter DIRTY_BYTES       = 0,
    parameter REPL_POLICY       = `CS_REPL_FIFO,
    parameter TAG_WIDTH         = UUID_WIDTH + 1,
    parameter CORE_OUT_BUF      = 0,
    parameter MEM_OUT_BUF       = 0,
    parameter IS_LLC            = 0,        // last-level cache: AMOs commit locally here
    parameter AMO_ENABLE        = 0,        // synthesize atomic-op logic
    // Bank pipeline depth (register stages from request-select to commit). 2 is
    // the classic lookup(S0)+commit(S1) pipeline; larger values defer the data
    // array by (LATENCY-2) stages to break the tag->data critical path on large
    // caches (tags/replacement/MSHR stay at S0/S1).
    parameter LATENCY           = 2,
    parameter MSHR_ADDR_WIDTH   = `LOG2UP(MSHR_SIZE),
    parameter MEM_TAG_WIDTH     = UUID_WIDTH + MSHR_ADDR_WIDTH,
    parameter REQ_SEL_WIDTH     = `UP(`CS_REQ_SEL_BITS),
    parameter WORD_SEL_WIDTH    = `UP(`CS_WORD_SEL_BITS)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output wire perf_read_miss,
    output wire perf_write_miss,
    output wire perf_evictions,
    output wire perf_mshr_stall,
`endif

    // Core request
    input wire                          core_req_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] core_req_addr,
    input wire                          core_req_rw,
    input wire [WORD_SEL_WIDTH-1:0]     core_req_wsel,
    input wire [WORD_SIZE-1:0]          core_req_byteen,
    input wire [`CS_WORD_WIDTH-1:0]     core_req_data,
    input wire [TAG_WIDTH-1:0]          core_req_tag,
    input wire [REQ_SEL_WIDTH-1:0]      core_req_idx,
    input wire [`UP(MEM_ATTR_WIDTH)-1:0] core_req_attr,
    output wire                         core_req_ready,

    // Core response
    output wire                         core_rsp_valid,
    output wire [`CS_WORD_WIDTH-1:0]    core_rsp_data,
    output wire [TAG_WIDTH-1:0]         core_rsp_tag,
    output wire [REQ_SEL_WIDTH-1:0]     core_rsp_idx,
    input  wire                         core_rsp_ready,

    // Memory request (sector-granular; = line when 1 sector/line)
    output wire                         mem_req_valid,
    output wire [`CS_LINE_SECTOR_ADDR_WIDTH-1:0] mem_req_addr,
    output wire                         mem_req_rw,
    output wire [SECTOR_SIZE-1:0]       mem_req_byteen,
    output wire [`CS_SECTOR_WIDTH-1:0]  mem_req_data,
    output wire [MEM_TAG_WIDTH-1:0]     mem_req_tag,
    output wire [`UP(MEM_ATTR_WIDTH)-1:0] mem_req_attr,
    input  wire                         mem_req_ready,

    // Memory response
    input wire                          mem_rsp_valid,
    input wire [`CS_SECTOR_WIDTH-1:0]   mem_rsp_data,
    input wire [MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                         mem_rsp_ready,

    // Flush
    input wire                          flush_begin,
    input wire [`UP(UUID_WIDTH)-1:0]    flush_uuid,
    output wire                         flush_end
);
    localparam PIPELINE_STAGES = LATENCY;
    localparam PIPE_EX = LATENCY - 2;       // extra data-deferral stages (0 = classic 2-stage)
    `STATIC_ASSERT(LATENCY >= 2, ("invalid parameter: cache bank LATENCY must be >= 2"))
    `UNUSED_PARAM (MRSQ_SIZE)

    // ========================================================================
    // Pipeline payload types
    //
    // The request travels as a struct and the S0-computed lookup results are a
    // separate `lookup_t` delta, composed into `commit_t` for the response /
    // memory-request stage. The wide fill `data` line and `tag_matches` ride
    // only the data-array path (`data_t`), never the commit path, so the deeper
    // commit pipeline stays narrow.
    //   sel -> S0     : data_t  (st0)            -- request + fill line
    //   S0  -> stD    : data_t  (stD)            -- drives the data array
    //   S0  -> S1->stC: commit_t (st1, stC)      -- request + lookup delta
    // `way_idx` and `mshr_id` are reused across stages (flush_way/replay_id at
    // select; resolved way / allocated id at commit). PIPE_EX=0 collapses
    // stD->S0 and stC->S1: the classic 2-stage bank.
    // ========================================================================
    typedef struct packed {
        logic                           valid, is_init, is_fill, is_flush, is_creq, is_replay, is_passthru_fill, rw;
        logic [`UP(MEM_ATTR_WIDTH)-1:0] attr;
        logic [`CS_WAY_SEL_WIDTH-1:0]   way_idx;     // flush_way @sel, resolved way @S1
        logic [`CS_LINE_ADDR_WIDTH-1:0] addr;
        logic [WORD_SIZE-1:0]           byteen;
        logic [WORD_SEL_WIDTH-1:0]      word_idx;
        logic [REQ_SEL_WIDTH-1:0]       req_idx;
        logic [TAG_WIDTH-1:0]           tag;
        logic [MSHR_ADDR_WIDTH-1:0]     mshr_id;     // replay_id @sel, alloc/replay id @S1
        amo_req_t                       amo;
    } req_t;

    typedef struct packed {            // S0-computed lookup delta (commit side)
        logic                          is_hit, is_dirty, mshr_pending;
        logic                          is_refill; // fill into an already-resident line (sector refill)
        logic [`CS_SECTORS_PER_LINE-1:0] evict_dirty_mask; // per-sector dirty of the evict way
        logic [`CS_TAG_SEL_BITS-1:0]   evict_tag;
        logic [`CS_WORD_WIDTH-1:0]      write_word;
        logic [MSHR_ADDR_WIDTH-1:0]    mshr_previd;
    } lookup_t;

    typedef struct packed {            // data-array drive (S0 -> stD)
        req_t                          req;
        logic [`CS_LINE_WIDTH-1:0]     data;
        logic [NUM_WAYS-1:0]           tag_matches;
    } data_t;

    typedef struct packed {            // response + memory request (S0 -> S1 -> stC)
        req_t                          req;
        lookup_t                       lk;
    } commit_t;

    data_t   sel_req, st0, dat_in, stD;   // request + fill line: sel -> S0 -> stD
    commit_t cmt_in, st1, stC;            // request + lookup delta: S0 -> S1 -> stC
    lookup_t lk_st0;                      // S0 lookup results

    // ------------------------------------------------------------------------
    // Shared signals
    // ------------------------------------------------------------------------
    wire crsp_queue_stall, mshr_alm_full, mshr_empty;
    wire mshr_probe_pending_ld, mshr_probe_pending_amo;
    wire mreq_queue_empty, mreq_queue_alm_full;
    wire [`CS_LINE_ADDR_WIDTH-1:0] mem_rsp_addr;
    wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] mem_rsp_sector; // sector this fill installs
    wire [MSHR_ADDR_WIDTH-1:0] mshr_alloc_id, mshr_previd;
    wire mshr_pending_raw;

    // MSHR replay (dequeue) sideband
    wire                           replay_valid, replay_ready, replay_rw;
    wire [`CS_LINE_ADDR_WIDTH-1:0] replay_addr;
    wire [WORD_SEL_WIDTH-1:0]      replay_wsel;
    wire [WORD_SIZE-1:0]           replay_byteen;
    wire [`CS_WORD_WIDTH-1:0]      replay_data;
    wire [TAG_WIDTH-1:0]           replay_tag;
    wire [REQ_SEL_WIDTH-1:0]       replay_idx;
    wire [MSHR_ADDR_WIDTH-1:0]     replay_id;
    amo_req_t                      replay_amo;

    // AMO engine interconnect (tied to 0 when the bank carries no AMO logic).
    wire                          amo_hit_st1, amo_commit_busy, amo_chain_stall, amo_wb_pending;
    wire [`CS_WORD_WIDTH-1:0]     amo_rsp_data;
    wire [`CS_LINE_ADDR_WIDTH-1:0] amo_wb_addr;
    wire [WORD_SEL_WIDTH-1:0]     amo_wb_word_idx;
    wire [WORD_SIZE-1:0]          amo_wb_byteen;
    wire [`CS_WORD_WIDTH-1:0]     amo_wb_data;
    wire [TAG_WIDTH-1:0]          amo_wb_tag;
    wire [REQ_SEL_WIDTH-1:0]      amo_wb_idx;
    wire [`UP(MEM_ATTR_WIDTH)-1:0] amo_wb_attr;
    wire                          is_amo_fwd_st0, is_amo_fwd_st1, is_amo_replay_st1;
    wire                          is_passthru_fill_sel, req_input_defer;
    wire [`CS_WORD_WIDTH-1:0]     amo_ptw_word_st1;

    wire flush_valid, flush_ready, init_valid;
    wire [`CS_LINE_SEL_BITS-1:0] flush_sel;
    wire [`CS_WAY_SEL_WIDTH-1:0] flush_way;

    // AMO sideband, extracted from the attr field (gated by AMO_ENABLE).
    amo_req_t core_req_amo;
    assign core_req_amo = AMO_ENABLE ? amo_req_t'(core_req_attr[MEM_ATTR_AMO_OFFS +: AMO_REQ_BITS])
                                     : amo_req_t'('0);

    // ------------------------------------------------------------------------
    // Per-stage decoded operations
    // ------------------------------------------------------------------------
    wire do_init_st0  = st0.req.valid && st0.req.is_init;
    wire do_flush_st0 = st0.req.valid && st0.req.is_flush;
    wire do_read_st0  = st0.req.valid && st0.req.is_creq && ~st0.req.rw;
    wire do_write_st0 = st0.req.valid && st0.req.is_creq && st0.req.rw;
    wire do_fill_st0  = st0.req.valid && st0.req.is_fill;
    wire do_lookup_st0 = do_read_st0 || do_write_st0;

    wire do_read_st1  = st1.req.valid && st1.req.is_creq && ~st1.req.rw;
    wire do_write_st1 = st1.req.valid && st1.req.is_creq && st1.req.rw;
    wire do_lookup_st1 = do_read_st1 || do_write_st1;

    wire do_read_stc  = stC.req.valid && stC.req.is_creq && ~stC.req.rw;
    wire do_write_stc = stC.req.valid && stC.req.is_creq && stC.req.rw;

    wire do_init_std  = stD.req.valid && stD.req.is_init;
    wire do_fill_std  = stD.req.valid && stD.req.is_fill;
    wire do_flush_std = stD.req.valid && stD.req.is_flush;
    wire do_read_std  = stD.req.valid && stD.req.is_creq && ~stD.req.rw;
    wire do_write_std = stD.req.valid && stD.req.is_creq && stD.req.rw;

    wire [`CS_LINE_SEL_BITS-1:0] line_idx_st0 = st0.req.addr[`CS_LINE_SEL_BITS-1:0];
    wire [`CS_TAG_SEL_BITS-1:0]  line_tag_st0 = `CS_LINE_ADDR_TAG(st0.req.addr);
    // Requested sector = top CS_SECTOR_SEL_BITS of the in-line word offset.
    wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx_st0;
    if (`CS_SECTOR_SEL_BITS != 0) begin : g_sector_idx
        assign sector_idx_st0 = st0.req.word_idx[`CS_WORD_SEL_BITS-1 -: `CS_SECTOR_SEL_BITS];
    end else begin : g_sector_idx0
        assign sector_idx_st0 = '0;
    end
    wire [`CS_WORD_WIDTH-1:0]    write_word_st0 = st0.data[`CS_WORD_WIDTH-1:0];
    wire [`CS_LINE_ADDR_WIDTH-1:0] addr_stc = stC.req.addr;

    // ------------------------------------------------------------------------
    // Bank-empty detection (gates flush). A request occupies S0, S1 and the
    // PIPE_EX commit-bubble stages (valid_st1 delayed 1..PIPE_EX); the parallel
    // data bubble S0->stD is subsumed by this window.
    // ------------------------------------------------------------------------
    wire pipe_inflight;
    if (PIPE_EX == 0) begin : g_no_bubble_occ
        assign pipe_inflight = st0.req.valid || st1.req.valid;
    end else begin : g_bubble_occ
        reg [PIPE_EX-1:0] commit_valid;
        always @(posedge clk) begin
            if (reset) begin
                commit_valid <= '0;
            end else if (~pipe_stall) begin
                commit_valid[0] <= st1.req.valid;
                for (int i = 1; i < PIPE_EX; ++i) begin
                    commit_valid[i] <= commit_valid[i-1];
                end
            end
        end
        assign pipe_inflight = st0.req.valid || st1.req.valid || (| commit_valid);
    end
    wire no_pending_req = ~pipe_inflight && mreq_queue_empty;

    VX_cache_flush #(
        .BANK_ID    (BANK_ID),
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WRITEBACK  (WRITEBACK)
    ) cache_flush (
        .clk         (clk),
        .reset       (reset),
        .flush_begin (flush_begin),
        .flush_end   (flush_end),
        .flush_init  (init_valid),
        .flush_valid (flush_valid),
        .flush_line  (flush_sel),
        .flush_way   (flush_way),
        .flush_ready (flush_ready),
        .mshr_empty  (mshr_empty),
        .bank_empty  (no_pending_req)
    );

    // wb_hold pauses the commit while a multi-beat per-sector writeback drains
    // (0 for single-sector lines, so the baseline pipe is unaffected).
    wire wb_hold;
    // amo_chain_stall paces a same-line AMO behind an in-flight commit by one
    // cycle; it is 0 for non-AMO traffic, so the baseline pipe is unaffected.
    wire pipe_stall = crsp_queue_stall || amo_chain_stall || wb_hold;

    // ========================================================================
    // Input arbitration
    //   priority: init > replay > fill(mem_rsp) > flush > core-req
    // replay maximizes utilization (guaranteed hit); fill precedes flush/creq to
    // avoid deadlock on a miss; flush precedes creq for consistency.
    // ========================================================================
    wire replay_grant  = ~init_valid;
    wire replay_enable = replay_grant && replay_valid;
    wire fill_grant    = replay_grant && ~replay_enable;
    wire fill_enable   = fill_grant && mem_rsp_valid;
    wire flush_grant   = fill_grant && ~fill_enable;
    wire flush_enable  = flush_grant && flush_valid;
    wire creq_grant    = flush_grant && ~flush_enable;

    // A core-request slot fires from a real core_req or a pending LLC AMO
    // writeback (synthetic write injected after a commit); mutually exclusive.
    // amo_commit_busy/req_input_defer enforce AMO ordering (0 for non-AMO banks).
    wire amo_creq_path = core_req_valid && ~amo_commit_busy && ~req_input_defer;
    wire amo_wb_path   = amo_wb_pending && ~amo_hit_st1;
    wire creq_enable   = creq_grant && (amo_creq_path || amo_wb_path);

    assign replay_ready   = replay_grant && ~(!WRITEBACK && replay_rw && mreq_queue_alm_full) && ~pipe_stall;
    assign mem_rsp_ready  = fill_grant && ~(WRITEBACK && mreq_queue_alm_full) && ~pipe_stall;
    assign flush_ready    = flush_grant && ~(WRITEBACK && mreq_queue_alm_full) && ~pipe_stall;
    assign core_req_ready = creq_grant && ~mreq_queue_alm_full && ~mshr_alm_full && ~pipe_stall
                         && ~amo_commit_busy && ~req_input_defer;

    wire init_fire    = init_valid;
    wire replay_fire  = replay_valid && replay_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;
    wire flush_fire   = flush_valid && flush_ready;
    // amo_wb_path already excludes the cycle a fresh AMO commits at S1, so the
    // writeback never races the chain update.
    wire amo_wb_fire   = amo_wb_path && creq_grant && ~mreq_queue_alm_full && ~mshr_alm_full && ~pipe_stall;
    wire core_req_fire = (amo_creq_path || amo_wb_path) && creq_grant
                       && ~mreq_queue_alm_full && ~mshr_alm_full && ~pipe_stall;

    wire [MSHR_ADDR_WIDTH-1:0] mem_rsp_id = mem_rsp_tag[MSHR_ADDR_WIDTH-1:0];

    // Fill word_idx: place the installed sector in the high (sector) bits of the
    // in-line word offset; low (word-in-sector) bits are don't-care for a fill.
    wire [WORD_SEL_WIDTH-1:0] fill_word_idx;
    if (`CS_SECTOR_SEL_BITS != 0) begin : g_fill_word_idx
        assign fill_word_idx = WORD_SEL_WIDTH'(mem_rsp_sector) << (`CS_WORD_SEL_BITS - `CS_SECTOR_SEL_BITS);
    end else begin : g_fill_word_idx0
        `UNUSED_VAR (mem_rsp_sector)
        assign fill_word_idx = '0;
    end

    // generate-guarded width selects (the dead branch must not elaborate an
    // out-of-range slice when the other width path is taken).
    wire [TAG_WIDTH-1:0] mem_rsp_tag_s;
    if (TAG_WIDTH > MEM_TAG_WIDTH) begin : g_mem_rsp_tag_s_pad
        assign mem_rsp_tag_s = {mem_rsp_tag, (TAG_WIDTH-MEM_TAG_WIDTH)'(1'b0)};
    end else begin : g_mem_rsp_tag_s_cut
        assign mem_rsp_tag_s = mem_rsp_tag[MEM_TAG_WIDTH-1 -: TAG_WIDTH];
        `UNUSED_VAR (mem_rsp_tag)
    end

    wire [TAG_WIDTH-1:0] flush_tag;
    if (UUID_WIDTH != 0) begin : g_flush_tag_uuid
        if (TAG_WIDTH > UUID_WIDTH) begin : g_pad
            assign flush_tag = {flush_uuid, (TAG_WIDTH-UUID_WIDTH)'(1'b0)};
        end else begin : g_cut
            assign flush_tag = flush_uuid[UUID_WIDTH-1 -: TAG_WIDTH];
        end
    end else begin : g_flush_tag_0
        `UNUSED_VAR (flush_uuid)
        assign flush_tag = '0;
    end

    // Per-bit fill/write data mux. AMO writeback fields tie to 0 for non-AMO
    // banks, so the wb arms prune away.
    // The fill response carries one sector (CS_SECTOR_WIDTH). Replicate it across
    // the line so each sector's slices see their words; the data array writes
    // only the installed sector (per-slice gate). i % CS_SECTOR_WIDTH == i when
    // 1 sector/line, so this is the legacy full-line fill there.
    wire [`CS_LINE_WIDTH-1:0] data_sel;
    if (WRITE_ENABLE) begin : g_data_sel
        for (genvar i = 0; i < `CS_LINE_WIDTH; ++i) begin : g_i
            if (i < `CS_WORD_WIDTH) begin : g_lo
                assign data_sel[i] = replay_valid ? replay_data[i]
                                   : (mem_rsp_valid ? mem_rsp_data[i % `CS_SECTOR_WIDTH]
                                   : (amo_wb_pending ? amo_wb_data[i] : core_req_data[i]));
            end else begin : g_hi
                assign data_sel[i] = mem_rsp_data[i % `CS_SECTOR_WIDTH]; // fill (sector-replicated)
            end
        end
    end else begin : g_data_sel_ro
        assign data_sel = {`CS_SECTORS_PER_LINE{mem_rsp_data}};
        `UNUSED_VAR ({core_req_data, replay_data, amo_wb_data})
    end

    // Input mux -> arbitrated request (whole-struct populate). AMO priority
    // matches the mux (replay > wb > core_req): a replay can fire during a
    // pending wb (chained AMO replays from MSHR after a fill) and must not be
    // cleared by amo_wb_pending; the synthetic writeback carries amo.valid=0 so
    // it never re-commits at S1.
    always @(*) begin
        sel_req = '0;
        sel_req.req.valid    = init_fire || replay_fire || mem_rsp_fire || flush_fire || core_req_fire;
        sel_req.req.is_init  = init_valid;
        sel_req.req.is_fill  = fill_enable;
        sel_req.req.is_flush = flush_enable;
        sel_req.req.is_creq  = creq_enable || replay_enable;
        sel_req.req.is_replay = replay_enable;
        sel_req.req.is_passthru_fill = is_passthru_fill_sel;
        sel_req.req.rw       = replay_valid ? replay_rw : (amo_wb_pending ? 1'b1 : core_req_rw);
        sel_req.req.attr     = amo_wb_pending ? amo_wb_attr : (core_req_valid ? core_req_attr : '0);
        sel_req.req.way_idx  = flush_way;
        sel_req.req.addr     = (init_valid | flush_valid) ? `CS_LINE_ADDR_WIDTH'(flush_sel)
                             : (replay_valid ? replay_addr : (mem_rsp_valid ? mem_rsp_addr
                             : (amo_wb_pending ? amo_wb_addr : core_req_addr)));
        sel_req.req.byteen   = replay_valid ? replay_byteen : (amo_wb_pending ? amo_wb_byteen : core_req_byteen);
        // a fill carries the installed sector in its word_idx high bits so the
        // tag/data stages mark/write the right sector (0 when 1 sector/line).
        sel_req.req.word_idx = replay_valid ? replay_wsel
                             : (mem_rsp_valid ? fill_word_idx
                             : (amo_wb_pending ? amo_wb_word_idx : core_req_wsel));
        sel_req.req.req_idx  = replay_valid ? replay_idx : (amo_wb_pending ? amo_wb_idx : core_req_idx);
        sel_req.req.tag      = (init_valid | flush_valid) ? (flush_valid ? flush_tag : '0)
                             : (replay_valid ? replay_tag : (mem_rsp_valid ? mem_rsp_tag_s
                             : (amo_wb_pending ? amo_wb_tag : core_req_tag)));
        sel_req.req.mshr_id  = replay_id;
        sel_req.req.amo      = replay_valid ? replay_amo : (amo_wb_pending ? amo_req_t'('0)
                             : (core_req_valid ? core_req_amo : amo_req_t'('0)));
        sel_req.data         = data_sel;
        // tag_matches is computed at S0; left 0 here (overridden at the data bubble).
    end

    // UUID extraction (debug + MSHR ordering): per stage, from the carried tag.
    wire [`UP(UUID_WIDTH)-1:0] req_uuid_sel, req_uuid_st0, req_uuid_st1, req_uuid_stc;
    if (UUID_WIDTH != 0) begin : g_req_uuid
        assign req_uuid_sel = sel_req.req.tag[TAG_WIDTH-1 -: UUID_WIDTH];
        assign req_uuid_st0 = st0.req.tag[TAG_WIDTH-1 -: UUID_WIDTH];
        assign req_uuid_st1 = st1.req.tag[TAG_WIDTH-1 -: UUID_WIDTH];
        assign req_uuid_stc = stC.req.tag[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_req_uuid_0
        assign {req_uuid_sel, req_uuid_st0, req_uuid_st1, req_uuid_stc} = '0;
    end
    `UNUSED_VAR ({req_uuid_st0, req_uuid_st1})

    // S0 register
    VX_pipe_register #(
        .DATAW  ($bits(data_t)),
        .RESETW (1)
    ) reg_s0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  (sel_req),
        .data_out (st0)
    );

    // ========================================================================
    // S0 lookup: replacement + tags + way-encode + MSHR allocate
    // ========================================================================
    wire [`CS_WAY_SEL_WIDTH-1:0] victim_way;
    wire [NUM_WAYS-1:0] tag_matches_st0;
    wire [NUM_WAYS-1:0] line_present_st0;
    wire [`CS_WAY_SEL_WIDTH-1:0] hit_idx_st0;
    wire evict_dirty_st0;
    wire [`CS_SECTORS_PER_LINE-1:0] evict_dirty_mask_st0;
    wire [`CS_TAG_SEL_BITS-1:0] evict_tag_st0;

    // A fill into a line that is already resident (a sector refill) must target
    // the resident way, not a fresh victim, so the new sector lands in the same
    // line copy. With 1 sector/line a fill's line is never already resident, so
    // this is gated off and the victim way is always used (legacy behavior).
    wire line_present_any_st0 = (`CS_SECTORS_PER_LINE > 1) && (| line_present_st0);
    wire [`CS_WAY_SEL_WIDTH-1:0] present_way_st0;
    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) present_way_enc (
        .data_in  (line_present_st0),
        .data_out (present_way_st0),
        `UNUSED_PIN (valid_out)
    );
    wire [`CS_WAY_SEL_WIDTH-1:0] fill_way_st0 = line_present_any_st0 ? present_way_st0 : victim_way;
    wire [`CS_WAY_SEL_WIDTH-1:0] evict_way_st0 = st0.req.is_fill ? fill_way_st0 : st0.req.way_idx;

    VX_cache_repl #(
        .CACHE_SIZE  (CACHE_SIZE),
        .LINE_SIZE   (LINE_SIZE),
        .NUM_BANKS   (NUM_BANKS),
        .NUM_WAYS    (NUM_WAYS),
        .REPL_POLICY (REPL_POLICY)
    ) cache_repl (
        .clk          (clk),
        .reset        (reset),
        .stall        (pipe_stall),
        .init         (do_init_st0),
        .lookup_valid (do_lookup_st1 && ~pipe_stall),
        .lookup_hit   (st1.lk.is_hit),
        .lookup_line  (st1.req.addr[`CS_LINE_SEL_BITS-1:0]),
        .lookup_way   (st1.req.way_idx),
        .repl_valid   (do_fill_st0 && ~st0.req.is_passthru_fill && ~pipe_stall),
        .repl_line    (line_idx_st0),
        .repl_line_n  (sel_req.req.addr[`CS_LINE_SEL_BITS-1:0]),
        .repl_way     (victim_way)
    );

    VX_cache_tags #(
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WORD_SIZE  (WORD_SIZE),
        .SECTOR_SIZE(SECTOR_SIZE),
        .WRITEBACK  (WRITEBACK),
        .AMO_ENABLE ((AMO_ENABLE != 0) && (IS_LLC == 0))
    ) cache_tags (
        .clk         (clk),
        .reset       (reset),
        .stall       (pipe_stall),
        .init        (do_init_st0),
        .flush       (do_flush_st0 && ~pipe_stall),
        .fill        (do_fill_st0 && ~st0.req.is_passthru_fill && ~pipe_stall),
        .read        (do_read_st0 && ~pipe_stall),
        .write       (do_write_st0 && ~pipe_stall),
        // non-LLC AMO forwards downstream and invalidates its own copy so the
        // issuer's later plain load refetches the new value.
        .invalidate  (is_amo_fwd_st0 && lk_st0.is_hit && ~pipe_stall),
        .line_idx    (line_idx_st0),
        .line_idx_n  (sel_req.req.addr[`CS_LINE_SEL_BITS-1:0]),
        .line_tag    (line_tag_st0),
        .sector_idx  (sector_idx_st0),
        .evict_way   (evict_way_st0),
        .tag_matches (tag_matches_st0),
        .line_present (line_present_st0),
        .evict_dirty (evict_dirty_st0),
        .evict_dirty_mask (evict_dirty_mask_st0),
        .evict_tag   (evict_tag_st0)
    );

    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) way_idx_enc (
        .data_in  (tag_matches_st0),
        .data_out (hit_idx_st0),
        `UNUSED_PIN (valid_out)
    );

    // S0 lookup delta (single combinational driver). The AMO requester is forced
    // non-pending so it never coalesces onto a prior same-line entry.
    always @(*) begin
        lk_st0 = '0;
        lk_st0.is_hit       = (| tag_matches_st0);
        lk_st0.is_dirty     = evict_dirty_st0;
        lk_st0.is_refill    = st0.req.is_fill && line_present_any_st0;
        lk_st0.evict_dirty_mask = evict_dirty_mask_st0;
        lk_st0.evict_tag    = evict_tag_st0;
        lk_st0.write_word   = write_word_st0;
        lk_st0.mshr_previd  = mshr_previd;
        lk_st0.mshr_pending = mshr_pending_raw && ~is_amo_fwd_st0;
    end

    // ========================================================================
    // Pipeline registration
    //
    // Tags / replacement / MSHR (allocate AND finalize) stay at S0/S1: the MSHR
    // coalescing chain requires allocate(S0)->finalize(S1) exactly one cycle
    // apart (deferring it orphans coalesced same-line entries -> deadlock). Only
    // the data array (stD) and the commit consumers (stC) defer by PIPE_EX, so
    // the array is driven by *registered* tag-compare results — breaking the
    // tag->data critical path. Read and write both move to the same deferred
    // stage, so pipeline order is preserved (no store->load hazard logic).
    // ========================================================================

    // data path: carry the request + fill line + tag compare, resolving the way
    // for the data array (victim way for fill/flush, hit way otherwise).
    always @(*) begin
        dat_in = st0;
        dat_in.req.way_idx = evict_way_st0;
        dat_in.tag_matches = tag_matches_st0;
    end

    // commit path: the request (with the resolved hit/victim way and MSHR id)
    // plus the lookup delta. The wide fill line is dropped here.
    always @(*) begin
        cmt_in.req = st0.req;
        cmt_in.req.way_idx = st0.req.is_creq ? hit_idx_st0 : evict_way_st0;
        cmt_in.req.mshr_id = st0.req.is_replay ? st0.req.mshr_id : mshr_alloc_id;
        cmt_in.lk = lk_st0;
    end

    VX_pipe_register #(
        .DATAW  ($bits(data_t)),
        .RESETW (1),
        .DEPTH  (PIPE_EX)
    ) reg_dat (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  (dat_in),
        .data_out (stD)
    );

    VX_pipe_register #(
        .DATAW  ($bits(commit_t)),
        .RESETW (1)
    ) reg_s1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  (cmt_in),
        .data_out (st1)
    );

    VX_pipe_register #(
        .DATAW  ($bits(commit_t)),
        .RESETW (1),
        .DEPTH  (PIPE_EX)
    ) reg_cmt (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  (st1),
        .data_out (stC)
    );

    // a passthru-AMO replay carries its result word instead of an installed
    // line, so it counts as a hit at the commit stage.
    wire eff_hit_st1 = st1.lk.is_hit || is_amo_replay_st1;
    wire eff_hit_stc = stC.lk.is_hit || is_amo_replay_st1;
    `RUNTIME_ASSERT (~(st1.req.valid && st1.req.is_replay && ~eff_hit_st1), ("missed mshr replay"))

    // ========================================================================
    // Data array (driven at stD; outputs land at stC)
    // ========================================================================
    wire[`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] read_data_stc;
    wire [LINE_SIZE-1:0] evict_byteen_stc;
    wire [`CS_WORD_WIDTH-1:0] read_word_stc = read_data_stc[stC.req.word_idx];

    // Sector being accessed at the data-array stage = top bits of word_idx.
    wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx_std;
    if (`CS_SECTOR_SEL_BITS != 0) begin : g_sector_idx_std
        assign sector_idx_std = stD.req.word_idx[`CS_WORD_SEL_BITS-1 -: `CS_SECTOR_SEL_BITS];
    end else begin : g_sector_idx_std0
        assign sector_idx_std = '0;
    end

    VX_cache_data #(
        .CACHE_SIZE   (CACHE_SIZE),
        .LINE_SIZE    (LINE_SIZE),
        .NUM_BANKS    (NUM_BANKS),
        .NUM_WAYS     (NUM_WAYS),
        .WORD_SIZE    (WORD_SIZE),
        .SECTOR_SIZE  (SECTOR_SIZE),
        .WRITE_ENABLE (WRITE_ENABLE),
        .WRITEBACK    (WRITEBACK),
        .DIRTY_BYTES  (DIRTY_BYTES)
    ) cache_data (
        .clk          (clk),
        .reset        (reset),
        .init         (do_init_std),
        .fill         (do_fill_std && ~stD.req.is_passthru_fill && ~pipe_stall),
        .flush        (do_flush_std && ~pipe_stall),
        .read         (do_read_std && ~pipe_stall),
        .write        (do_write_std && ~pipe_stall),
        .evict_way    (stD.req.way_idx),
        .tag_matches  (stD.tag_matches),
        .line_idx     (stD.req.addr[`CS_LINE_SEL_BITS-1:0]),
        .fill_data    (stD.data),
        .write_word   (stD.data[`CS_WORD_WIDTH-1:0]),
        .word_idx     (stD.req.word_idx),
        .sector_idx   (sector_idx_std),
        .way_idx_r    (stC.req.way_idx),
        .write_byteen (stD.req.byteen),
        .read_data    (read_data_stc),
        .evict_byteen (evict_byteen_stc)
    );

    // ========================================================================
    // MSHR (allocate at S0, finalize at S1)
    // ========================================================================
    wire mshr_allocate_st0 = st0.req.valid && st0.req.is_creq && ~st0.req.is_replay;
    wire mshr_finalize_st1 = st1.req.valid && st1.req.is_creq && ~st1.req.is_replay;

    // release the entry on a hit. A forwarded AMO keeps its entry until its
    // downstream response returns (fill/dequeue frees it), so never release it.
    wire mshr_release_st1;
    if (WRITEBACK) begin : g_mshr_release
        assign mshr_release_st1 = st1.lk.is_hit && ~is_amo_fwd_st1;
    end else begin : g_mshr_release_ro
        // keep missed writes in MSHR if a pending entry exists for the line, so a
        // pending fill arriving without the write content replays them locally.
        assign mshr_release_st1 = (st1.lk.is_hit || (st1.req.rw && ~st1.lk.mshr_pending)) && ~is_amo_fwd_st1;
    end
    wire mshr_release_fire = mshr_finalize_st1 && mshr_release_st1 && ~pipe_stall;

    wire [1:0] mshr_dequeue;
    `POP_COUNT(mshr_dequeue, {replay_fire, mshr_release_fire});

    VX_pending_size #(
        .SIZE  (MSHR_SIZE),
        .DECRW (2)
    ) mshr_pending_size (
        .clk   (clk),
        .reset (reset),
        .incr  (core_req_fire),
        .decr  (mshr_dequeue),
        .empty (mshr_empty),
        `UNUSED_PIN (alm_empty),
        .full  (mshr_alm_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    VX_cache_mshr #(
        .INSTANCE_ID (`SFORMATF(("%s-mshr", INSTANCE_ID))),
        .BANK_ID     (BANK_ID),
        .LINE_SIZE   (LINE_SIZE),
        .SECTOR_SIZE (SECTOR_SIZE),
        .NUM_BANKS   (NUM_BANKS),
        .MSHR_SIZE   (MSHR_SIZE),
        .WRITEBACK   (WRITEBACK),
        .AMO_ENABLE  ((AMO_ENABLE != 0) && (IS_LLC == 0)),
        .DATA_WIDTH  (WORD_SEL_WIDTH + WORD_SIZE + `CS_WORD_WIDTH + TAG_WIDTH + REQ_SEL_WIDTH + AMO_REQ_BITS)
    ) cache_mshr (
        .clk                 (clk),
        .reset               (reset),
        .deq_req_uuid        (req_uuid_sel),
        .alc_req_uuid        (req_uuid_st0),
        .fin_req_uuid        (req_uuid_st1),
        .fill_valid          (mem_rsp_fire),
        .fill_id             (mem_rsp_id),
        .fill_addr           (mem_rsp_addr),
        .fill_sector         (mem_rsp_sector),
        .probe_addr          (core_req_addr),
        .probe_pending_ld    (mshr_probe_pending_ld),
        .probe_pending_amo   (mshr_probe_pending_amo),
        .dequeue_valid       (replay_valid),
        .dequeue_addr        (replay_addr),
        .dequeue_rw          (replay_rw),
        .dequeue_data        ({replay_wsel, replay_byteen, replay_data, replay_tag, replay_idx, replay_amo}),
        .dequeue_id          (replay_id),
        .dequeue_ready       (replay_ready),
        .allocate_valid      (mshr_allocate_st0 && ~pipe_stall),
        .allocate_addr       (st0.req.addr),
        .allocate_sector     (sector_idx_st0),
        .allocate_rw         (st0.req.rw),
        // Only non-LLC AMOs must not coalesce; at the LLC same-line AMOs coalesce
        // and serialize their commits on the single filled line.
        .allocate_is_amo     ((AMO_ENABLE && !IS_LLC) ? st0.req.amo.amo_valid : 1'b0),
        .allocate_data       ({st0.req.word_idx, st0.req.byteen, write_word_st0, st0.req.tag, st0.req.req_idx, st0.req.amo}),
        .allocate_id         (mshr_alloc_id),
        .allocate_pending    (mshr_pending_raw),
        .allocate_previd     (mshr_previd),
        `UNUSED_PIN (allocate_ready),
        .finalize_valid      (mshr_finalize_st1 && ~pipe_stall),
        .finalize_is_release (mshr_release_st1),
        // Only link an entry into the pending chain if it is KEPT (a miss). A
        // released (hit) entry must never become a chain member: otherwise the
        // prev's fill would later dequeue and replay the already-released slot
        // (double free -> MSHR pending-size underflow). This case arises when a
        // request hits a line that is still draining its fill chain — common
        // with sectoring, where a hot line accumulates a long same-line chain.
        .finalize_is_pending (st1.lk.mshr_pending && ~mshr_release_st1),
        .finalize_id         (st1.req.mshr_id),
        .finalize_previd     (st1.lk.mshr_previd)
    );

    // ========================================================================
    // AMO engine
    //
    // The read word lands at the deferred commit stage stC; the engine consumes
    // it at S1 (== stC when PIPE_EX=0, the validated case).
    // ========================================================================
    if (AMO_ENABLE) begin : g_amo
        // Look-ahead line address for the reservation cache's sync-BRAM read:
        // the line entering the commit stage (stC) next cycle, so the registered
        // read lands at stC. stC = st1 delayed by PIPE_EX; one stage earlier is
        // st0 (PIPE_EX=0) or st1 delayed by PIPE_EX-1 (PIPE_EX>0).
        wire [`CS_LINE_ADDR_WIDTH-1:0] amo_res_addr_n;
        if (PIPE_EX == 0) begin : g_resn0
            assign amo_res_addr_n = st0.req.addr;
        end else begin : g_resn
            VX_pipe_register #(
                .DATAW (`CS_LINE_ADDR_WIDTH),
                .DEPTH (PIPE_EX - 1)
            ) reg_resn (
                .clk      (clk),
                .reset    (reset),
                .enable   (~pipe_stall),
                .data_in  (st1.req.addr),
                .data_out (amo_res_addr_n)
            );
        end

        VX_cache_amo #(
            .IS_LLC          (IS_LLC),
            .NUM_RES_ENTRIES (`VX_CFG_AMO_RS_SIZE),
            .LINE_ADDR_BITS  (`CS_LINE_ADDR_WIDTH),
            .WORD_WIDTH      (`CS_WORD_WIDTH),
            .WORD_SIZE       (WORD_SIZE),
            .WORD_SEL_WIDTH  (WORD_SEL_WIDTH),
            .TAG_WIDTH       (TAG_WIDTH),
            .REQ_SEL_WIDTH   (REQ_SEL_WIDTH),
            .ATTR_WIDTH      (`UP(MEM_ATTR_WIDTH)),
            .MSHR_SIZE       (MSHR_SIZE),
            .MSHR_ADDR_WIDTH (MSHR_ADDR_WIDTH),
            .WORDS_PER_LINE  (`CS_WORDS_PER_LINE),
            .WORDS_PER_SECTOR(`CS_WORDS_PER_SECTOR),
            .PIPE_EX         (PIPE_EX)
        ) amo (
            .clk                    (clk),
            .reset                  (reset),
            .pipe_stall             (pipe_stall),
            .amo_st0                (st0.req.amo),
            .valid_st0              (st0.req.valid),
            .is_creq_st0            (st0.req.is_creq),
            .is_hit_st0             (lk_st0.is_hit),
            .is_replay_st0          (st0.req.is_replay),
            // Commit ports are fed from stC (the deferred data-output stage), so
            // the AMO RMW operands and the read word align at PIPE_EX>0. At
            // PIPE_EX=0, stC == S1 and this is identical to the classic bank.
            .amo_st1                (stC.req.amo),
            .valid_st1              (stC.req.valid),
            .is_creq_st1            (stC.req.is_creq),
            .is_hit_st1             (stC.lk.is_hit),
            .is_replay_st1          (stC.req.is_replay),
            .do_write_st1           (do_write_stc),
            .read_word_st1          (read_word_stc),
            .byteen_st1             (stC.req.byteen),
            .write_word_st1         (stC.lk.write_word),
            .word_idx_st0           (st0.req.word_idx),
            .word_idx_st1           (stC.req.word_idx),
            .addr_st0               (st0.req.addr),
            .addr_st1               (addr_stc),
            .res_addr_n             (amo_res_addr_n),
            .tag_st1                (stC.req.tag),
            .req_idx_st1            (stC.req.req_idx),
            .attr_st1               (stC.req.attr),
            .wb_fire                (amo_wb_fire),
            .mshr_allocate_st0      (mshr_allocate_st0),
            .mshr_alloc_id_st0      (mshr_alloc_id),
            .mshr_id_st1            (stC.req.mshr_id),
            .mem_rsp_fire           (mem_rsp_fire),
            .mem_rsp_id             (mem_rsp_id),
            .mem_rsp_data           (mem_rsp_data),
            .is_fill_sel            (fill_enable),
            .core_req_valid         (core_req_valid),
            .core_req_is_amo        (core_req_amo.amo_valid),
            .core_req_rw            (core_req_rw),
            .core_req_addr          (core_req_addr),
            .rw_st0                 (st0.req.rw),
            .mshr_probe_pending_ld  (mshr_probe_pending_ld),
            .mshr_probe_pending_amo (mshr_probe_pending_amo),
            .amo_hit_st1            (amo_hit_st1),
            .commit_busy            (amo_commit_busy),
            .chain_stall            (amo_chain_stall),
            .wb_pending             (amo_wb_pending),
            .rsp_data               (amo_rsp_data),
            .wb_addr                (amo_wb_addr),
            .wb_word_idx            (amo_wb_word_idx),
            .wb_byteen              (amo_wb_byteen),
            .wb_data                (amo_wb_data),
            .wb_tag                 (amo_wb_tag),
            .wb_idx                 (amo_wb_idx),
            .wb_attr                (amo_wb_attr),
            .is_amo_fwd_st0         (is_amo_fwd_st0),
            .is_amo_fwd_st1         (is_amo_fwd_st1),
            .is_amo_replay_st1      (is_amo_replay_st1),
            .is_passthru_fill_sel   (is_passthru_fill_sel),
            .amo_ptw_word_st1       (amo_ptw_word_st1),
            .req_input_defer        (req_input_defer)
        );
    end else begin : g_no_amo
        assign {amo_hit_st1, amo_commit_busy, amo_wb_pending, amo_chain_stall} = '0;
        assign {amo_rsp_data, amo_wb_addr, amo_wb_word_idx, amo_wb_byteen} = '0;
        assign {amo_wb_data, amo_wb_tag, amo_wb_idx, amo_wb_attr} = '0;
        assign {is_amo_fwd_st0, is_amo_fwd_st1, is_amo_replay_st1} = '0;
        assign {is_passthru_fill_sel, amo_ptw_word_st1, req_input_defer} = '0;
        // S1-only signals consumed solely by the AMO engine.
        `UNUSED_VAR ({amo_wb_fire, mshr_probe_pending_ld, mshr_probe_pending_amo, st1.req.amo, st1.req.attr, st1.req.req_idx, st1.req.word_idx, st1.req.byteen, st1.lk.write_word})
    end

    // ========================================================================
    // Core response (stC)
    //
    // Fires for reads (and LLC AMO commits) on hit, never for the synthetic
    // writeback (rw=1). A non-LLC AMO's first pass forwards downstream and must
    // not respond locally (its result returns via the passthru replay). Suppress
    // while a same-line AMO is chain-stalled so a held read enqueues once.
    // ========================================================================
    wire crsp_queue_valid = do_read_stc && eff_hit_stc && ~is_amo_fwd_st1 && ~amo_chain_stall;
    wire crsp_queue_ready;
    wire [`CS_WORD_WIDTH-1:0] crsp_queue_data = is_amo_replay_st1 ? amo_ptw_word_st1
                                              : (amo_hit_st1 ? amo_rsp_data : read_word_stc);

    VX_elastic_buffer #(
        .DATAW   (TAG_WIDTH + `CS_WORD_WIDTH + REQ_SEL_WIDTH),
        .SIZE    (CRSQ_SIZE),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) core_rsp_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_queue_valid),
        .ready_in  (crsp_queue_ready),
        .data_in   ({stC.req.tag, crsp_queue_data, stC.req.req_idx}),
        .data_out  ({core_rsp_tag, core_rsp_data, core_rsp_idx}),
        .valid_out (core_rsp_valid),
        .ready_out (core_rsp_ready)
    );
    assign crsp_queue_stall = crsp_queue_valid && ~crsp_queue_ready;

    // ========================================================================
    // Memory request (stC) — sector-granular
    //
    // A read/write miss issues a fill request for the missed sector. A dirty
    // eviction writes back each dirty sector as its own sector-sized beat: the
    // sequencer below drains one dirty sector per cycle, holding the commit at
    // stC until the last beat is accepted. With 1 sector/line this is a single
    // beat (wb_hold never asserts) — byte-identical to the legacy path.
    // ========================================================================
    localparam SEC = `CS_SECTORS_PER_LINE;
    wire mreq_queue_push, mreq_queue_pop;
    wire [`CS_SECTOR_WIDTH-1:0] mreq_queue_data;
    wire [SECTOR_SIZE-1:0] mreq_queue_byteen;
    wire [`CS_LINE_SECTOR_ADDR_WIDTH-1:0] mreq_queue_addr;
    wire [MEM_TAG_WIDTH-1:0] mreq_queue_tag;
    wire mreq_queue_rw;

    wire is_fill_or_flush_stc = stC.req.is_fill || (stC.req.is_flush && WRITEBACK);
    wire do_fill_or_flush_stc = stC.req.valid && is_fill_or_flush_stc;
    // a sector refill keeps the resident line (no eviction), so never writes back.
    wire do_writeback_stc = do_fill_or_flush_stc && stC.lk.is_dirty && ~stC.lk.is_refill;
    wire [`CS_LINE_ADDR_WIDTH-1:0] evict_addr_stc = {stC.lk.evict_tag, stC.req.addr[`CS_LINE_SEL_BITS-1:0]};

    // sector of the request at commit (the missed sector of a fill request).
    wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx_stc;
    if (`CS_SECTOR_SEL_BITS != 0) begin : g_sector_idx_stc
        assign sector_idx_stc = stC.req.word_idx[`CS_WORD_SEL_BITS-1 -: `CS_SECTOR_SEL_BITS];
    end else begin : g_sector_idx_stc0
        assign sector_idx_stc = '0;
    end

    // Per-sector writeback sequencer. wb_mask_cur is the set of dirty sectors
    // still to write back; one is drained per cycle (lowest first). wb_done_r
    // latches once the current commit's writeback fully drains so an unrelated
    // hold of stC (crsp/amo) cannot re-inject it; it clears when the commit
    // finally advances.
    reg [SEC-1:0] wb_mask_r;
    reg wb_done_r;
    wire wb_active = (| wb_mask_r);
    wire [SEC-1:0] wb_mask_cur = wb_active ? wb_mask_r
                               : ((do_writeback_stc && ~wb_done_r) ? stC.lk.evict_dirty_mask : {SEC{1'b0}});
    wire is_wb_beat = (| wb_mask_cur);
    wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] wb_sector;
    VX_priority_encoder #(
        .N (SEC)
    ) wb_sector_sel (
        .data_in   (wb_mask_cur),
        .index_out (wb_sector),
        `UNUSED_PIN (valid_out),
        `UNUSED_PIN (onehot_out)
    );
    wire [SEC-1:0] wb_sec_oh = SEC'(1) << wb_sector;
    wire wb_beat_accept = is_wb_beat && ~mreq_queue_alm_full;
    wire [SEC-1:0] wb_mask_nxt = wb_beat_accept ? (wb_mask_cur & ~wb_sec_oh) : wb_mask_cur;
    assign wb_hold = (| wb_mask_nxt); // beats remain after this cycle -> hold stC
    always @(posedge clk) begin
        if (reset) begin
            wb_mask_r <= '0;
            wb_done_r <= 1'b0;
        end else begin
            wb_mask_r <= wb_mask_nxt;
            // latch done on the last accepted beat while stC is still held;
            // clear once the commit advances (so the next commit starts fresh).
            if (is_wb_beat && ~wb_hold) begin
                wb_done_r <= 1'b1;
            end
            if (~pipe_stall) begin
                wb_done_r <= 1'b0;
            end
        end
    end

    // sector-granular addresses: {line, sector} with sector in the low bits
    // (cache.sv re-inserts the bank id above the sector).
    wire [`CS_LINE_SECTOR_ADDR_WIDTH-1:0] wb_mreq_addr, rd_mreq_addr;
    if (`CS_SECTOR_SEL_BITS != 0) begin : g_sec_addr
        assign wb_mreq_addr = {evict_addr_stc, wb_sector};
        assign rd_mreq_addr = {addr_stc, sector_idx_stc};
    end else begin : g_no_sec_addr
        `UNUSED_VAR (sector_idx_stc)
        assign wb_mreq_addr = evict_addr_stc;
        assign rd_mreq_addr = addr_stc;
    end

    // selected writeback sector: data slice + per-byte dirty mask of that sector.
    // flatten the packed word-array before bit-slicing the sector out.
    wire [`CS_LINE_WIDTH-1:0]   read_data_flat_stc = read_data_stc;
    wire [`CS_SECTOR_WIDTH-1:0] wb_data_sec = read_data_flat_stc[wb_sector*`CS_SECTOR_WIDTH +: `CS_SECTOR_WIDTH];
    wire [SECTOR_SIZE-1:0]      wb_byteen_sec = evict_byteen_stc[wb_sector*SECTOR_SIZE +: SECTOR_SIZE];

    if (WRITE_ENABLE) begin : g_mreq_queue
        if (WRITEBACK) begin : g_wb
            if (DIRTY_BYTES) begin : g_dirty_bytes
                wire has_dirty_bytes = (| evict_byteen_stc);
                `RUNTIME_ASSERT (~do_fill_or_flush_stc || (stC.lk.is_dirty == has_dirty_bytes), ("missmatch dirty bytes: dirty_line=%b, dirty_bytes=%b, addr=0x%0h", stC.lk.is_dirty, has_dirty_bytes, `CS_BANK_TO_FULL_ADDR(addr_stc, BANK_ID)))
            end
            // fill request on a read/write miss (one sector); multi-beat writeback
            // on a dirty eviction. The two are mutually exclusive (a writeback is a
            // fill/flush commit; a fill request is a creq commit).
            wire fill_req_push = (do_read_stc || do_write_stc) && ~stC.lk.is_hit && ~stC.lk.mshr_pending && ~pipe_stall;
            assign mreq_queue_push   = fill_req_push || wb_beat_accept;
            assign mreq_queue_addr   = is_wb_beat ? wb_mreq_addr : rd_mreq_addr;
            assign mreq_queue_rw     = is_wb_beat;
            assign mreq_queue_data   = wb_data_sec; // read fill request: data unused
            assign mreq_queue_byteen = is_wb_beat ? wb_byteen_sec : {SECTOR_SIZE{1'b1}};
            `UNUSED_VAR ({stC.lk.write_word, stC.req.byteen, stC.req.is_replay})
        end else begin : g_wt
            // word byte-enable demuxed over the line, then sliced to the word's sector.
            wire [LINE_SIZE-1:0] full_byteen;
            VX_demux #(
                .DATAW (WORD_SIZE),
                .N     (`CS_WORDS_PER_LINE)
            ) byteen_demux (
                .sel_in   (stC.req.word_idx),
                .data_in  (stC.req.byteen),
                .data_out (full_byteen)
            );
            wire [SECTOR_SIZE-1:0] sec_byteen = full_byteen[sector_idx_stc*SECTOR_SIZE +: SECTOR_SIZE];
            // fill on a read miss; memory write on a write (don't resend replays);
            // forward a non-LLC AMO downstream (its passthru replay must not refill).
            assign mreq_queue_push = ((do_read_stc && ~eff_hit_stc && ~stC.lk.mshr_pending)
                                  || (do_write_stc && ~stC.req.is_replay)
                                  || is_amo_fwd_st1) && ~pipe_stall;
            assign mreq_queue_addr = rd_mreq_addr;
            assign mreq_queue_rw = stC.req.rw;
            assign mreq_queue_data = {`CS_WORDS_PER_SECTOR{stC.lk.write_word}};
            assign mreq_queue_byteen = (stC.req.rw || is_amo_fwd_st1) ? sec_byteen : {SECTOR_SIZE{1'b1}};
            `UNUSED_VAR ({is_wb_beat, wb_beat_accept, wb_mreq_addr, wb_data_sec, wb_byteen_sec, wb_sector, evict_addr_stc, stC.lk.evict_tag, stC.lk.is_dirty, stC.lk.evict_dirty_mask})
        end
    end else begin : g_mreq_queue_ro
        assign mreq_queue_push = (do_read_stc && ~stC.lk.is_hit && ~stC.lk.mshr_pending) && ~pipe_stall;
        assign mreq_queue_addr = rd_mreq_addr;
        assign mreq_queue_rw = 0;
        assign mreq_queue_data = '0;
        assign mreq_queue_byteen = {SECTOR_SIZE{1'b1}};
        `UNUSED_VAR ({is_wb_beat, wb_beat_accept, wb_mreq_addr, wb_data_sec, wb_byteen_sec, wb_sector, do_writeback_stc, evict_addr_stc, evict_byteen_stc, stC.lk.write_word, stC.lk.evict_tag, stC.lk.is_dirty, stC.lk.evict_dirty_mask, stC.req.byteen, stC.req.word_idx, stC.req.is_replay, do_write_stc})
    end

    if (UUID_WIDTH != 0) begin : g_mreq_queue_tag_uuid
        assign mreq_queue_tag = {req_uuid_stc, stC.req.mshr_id};
    end else begin : g_mreq_queue_tag
        assign mreq_queue_tag = stC.req.mshr_id;
    end

    assign mreq_queue_pop = mem_req_valid && mem_req_ready;

    VX_fifo_queue #(
        .DATAW    (1 + `CS_LINE_SECTOR_ADDR_WIDTH + SECTOR_SIZE + `CS_SECTOR_WIDTH + MEM_TAG_WIDTH + `UP(MEM_ATTR_WIDTH)),
        .DEPTH    (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE - PIPELINE_STAGES),
        .OUT_REG  (`TO_OUT_BUF_REG(MEM_OUT_BUF))
    ) mem_req_queue (
        .clk      (clk),
        .reset    (reset),
        .push     (mreq_queue_push),
        .pop      (mreq_queue_pop),
        .data_in  ({mreq_queue_rw, mreq_queue_addr, mreq_queue_byteen, mreq_queue_data, mreq_queue_tag, stC.req.attr}),
        .data_out ({mem_req_rw, mem_req_addr, mem_req_byteen, mem_req_data, mem_req_tag, mem_req_attr}),
        .empty    (mreq_queue_empty),
        .alm_full (mreq_queue_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );
    assign mem_req_valid = ~mreq_queue_empty;

    `UNUSED_VAR (do_lookup_st0)

///////////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    assign perf_read_miss  = do_read_st1 && ~st1.lk.is_hit;
    assign perf_write_miss = do_write_st1 && ~st1.lk.is_hit;
    assign perf_evictions  = do_writeback_stc;
    assign perf_mshr_stall = mshr_alm_full;
`endif

`ifdef DBG_TRACE_CACHE
    wire crsp_queue_fire = crsp_queue_valid && crsp_queue_ready;
    wire input_stall = (replay_valid || mem_rsp_valid || core_req_valid || flush_valid)
                   && ~(replay_fire || mem_rsp_fire || core_req_fire || flush_fire);

    wire [`VX_CFG_XLEN-1:0] mem_rsp_full_addr = `CS_BANK_TO_FULL_ADDR(mem_rsp_addr, BANK_ID);
    wire [`VX_CFG_XLEN-1:0] replay_full_addr = `CS_BANK_TO_FULL_ADDR(replay_addr, BANK_ID);
    wire [`VX_CFG_XLEN-1:0] core_req_full_addr = `CS_BANK_TO_FULL_ADDR(core_req_addr, BANK_ID);
    wire [`VX_CFG_XLEN-1:0] full_addr_st0 = `CS_BANK_TO_FULL_ADDR(st0.req.addr, BANK_ID);
    wire [`VX_CFG_XLEN-1:0] full_addr_st1 = `CS_BANK_TO_FULL_ADDR(st1.req.addr, BANK_ID);
    wire [`VX_CFG_XLEN-1:0] mreq_queue_full_addr = `CS_BANK_TO_FULL_ADDR(mreq_queue_addr, BANK_ID);

    always @(posedge clk) begin
        if (input_stall || pipe_stall) begin
            `TRACE(4, ("%t: *** %s stall: crsq=%b, mreq=%b, mshr=%b\n", $time, INSTANCE_ID,
                crsp_queue_stall, mreq_queue_alm_full, mshr_alm_full))
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%t: %s fill-rsp: addr=0x%0h, mshr_id=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                mem_rsp_full_addr, mem_rsp_id, mem_rsp_data, req_uuid_sel))
        end
        if (replay_fire) begin
            `TRACE(2, ("%t: %s mshr-pop: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID,
                replay_full_addr, replay_tag, replay_idx, req_uuid_sel))
        end
        if (core_req_fire) begin
            if (core_req_rw) begin
                `TRACE(2, ("%t: %s core-wr-req: addr=0x%0h, tag=0x%0h, req_idx=%0d, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                    core_req_full_addr, core_req_tag, core_req_idx, core_req_byteen, core_req_data, req_uuid_sel))
            end else begin
                `TRACE(2, ("%t: %s core-rd-req: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID,
                    core_req_full_addr, core_req_tag, core_req_idx, req_uuid_sel))
            end
        end
        if (do_init_st0) begin
            `TRACE(3, ("%t: %s tags-init: addr=0x%0h, line=%0d\n", $time, INSTANCE_ID, full_addr_st0, line_idx_st0))
        end
        if (do_fill_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s tags-fill: addr=0x%0h, way=%0d, line=%0d, dirty=%b (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, evict_way_st0, line_idx_st0, lk_st0.is_dirty, req_uuid_st0))
        end
        if (do_flush_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s tags-flush: addr=0x%0h, way=%0d, line=%0d, dirty=%b (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, evict_way_st0, line_idx_st0, lk_st0.is_dirty, req_uuid_st0))
        end
        if (do_lookup_st0 && ~pipe_stall) begin
            if (lk_st0.is_hit) begin
                `TRACE(3, ("%t: %s tags-hit: addr=0x%0h, rw=%b, way=%0d, line=%0d, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID,
                    full_addr_st0, st0.req.rw, hit_idx_st0, line_idx_st0, line_tag_st0, req_uuid_st0))
            end else begin
                `TRACE(3, ("%t: %s tags-miss: addr=0x%0h, rw=%b, way=%0d, line=%0d, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID,
                    full_addr_st0, st0.req.rw, hit_idx_st0, line_idx_st0, line_tag_st0, req_uuid_st0))
            end
        end
        if (do_fill_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-fill: addr=0x%0h, way=%0d, line=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, evict_way_st0, line_idx_st0, st0.data, req_uuid_st0))
        end
        if (do_flush_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-flush: addr=0x%0h, way=%0d, line=%0d (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, evict_way_st0, line_idx_st0, req_uuid_st0))
        end
        if (do_read_st1 && st1.lk.is_hit && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-read: addr=0x%0h, way=%0d, line=%0d, wsel=%0d (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st1, st1.req.way_idx, st1.req.addr[`CS_LINE_SEL_BITS-1:0], st1.req.word_idx, req_uuid_st1))
        end
        if (do_write_st1 && st1.lk.is_hit && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-write: addr=0x%0h, way=%0d, line=%0d, wsel=%0d, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st1, st1.req.way_idx, st1.req.addr[`CS_LINE_SEL_BITS-1:0], st1.req.word_idx, st1.req.byteen, st1.lk.write_word, req_uuid_st1))
        end
        if (crsp_queue_fire) begin
            `TRACE(2, ("%t: %s core-rd-rsp: addr=0x%0h, tag=0x%0h, req_idx=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                addr_stc, stC.req.tag, stC.req.req_idx, crsp_queue_data, req_uuid_stc))
        end
        if (mreq_queue_push) begin
            if (!WRITEBACK && do_write_stc) begin
                `TRACE(2, ("%t: %s writethrough: addr=0x%0h, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, mreq_queue_byteen, mreq_queue_data, req_uuid_stc))
            end else if (WRITEBACK && do_writeback_stc) begin
                `TRACE(2, ("%t: %s writeback: addr=0x%0h, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, mreq_queue_byteen, mreq_queue_data, req_uuid_stc))
            end else begin
                `TRACE(2, ("%t: %s fill-req: addr=0x%0h, mshr_id=%0d (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, stC.req.mshr_id, req_uuid_stc))
            end
        end
    end
`endif

endmodule
