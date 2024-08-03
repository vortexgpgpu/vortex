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

module VX_cache_bank #(
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0,

    // Number of Word requests per cycle
    parameter NUM_REQS          = 1,

    // Size of cache in bytes
    parameter CACHE_SIZE        = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 16,
    // Number of banks
    parameter NUM_BANKS         = 1,
    // Number of associative ways
    parameter NUM_WAYS          = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 4,

    // Core Response Queue Size
    parameter CRSQ_SIZE         = 1,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE         = 1,
    // Memory Request Queue Size
    parameter MREQ_SIZE         = 1,

    // Enable cache writeable
    parameter WRITE_ENABLE      = 1,

    // Enable cache writeback
    parameter WRITEBACK         = 0,

    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES       = 0,

    // Request debug identifier
    parameter UUID_WIDTH        = 0,

    // core request tag size
    parameter TAG_WIDTH         = UUID_WIDTH + 1,

    // Core response output buffer
    parameter CORE_OUT_BUF      = 0,

    // Memory request output buffer
    parameter MEM_OUT_BUF       = 0,

    parameter MSHR_ADDR_WIDTH   = `LOG2UP(MSHR_SIZE),
    parameter REQ_SEL_WIDTH     = `UP(`CS_REQ_SEL_BITS),
    parameter WORD_SEL_WIDTH    = `UP(`CS_WORD_SEL_BITS)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output wire perf_read_misses,
    output wire perf_write_misses,
    output wire perf_mshr_stalls,
`endif

    // Core Request
    input wire                          core_req_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] core_req_addr,
    input wire                          core_req_rw,    // write enable
    input wire [WORD_SEL_WIDTH-1:0]     core_req_wsel,  // select the word in a cacheline, e.g. word size = 4 bytes, cacheline size = 64 bytes, it should have log(64/4)= 4 bits
    input wire [WORD_SIZE-1:0]          core_req_byteen,// which bytes in data to write
    input wire [`CS_WORD_WIDTH-1:0]     core_req_data,  // data to be written
    input wire [TAG_WIDTH-1:0]          core_req_tag,   // identifier of the request (request id)
    input wire [REQ_SEL_WIDTH-1:0]      core_req_idx,   // index of the request in the core request array
    input wire                          core_req_flush, // flush enable
    output wire                         core_req_ready,

    // Core Response
    output wire                         core_rsp_valid,
    output wire [`CS_WORD_WIDTH-1:0]    core_rsp_data,
    output wire [TAG_WIDTH-1:0]         core_rsp_tag,
    output wire [REQ_SEL_WIDTH-1:0]     core_rsp_idx,
    input  wire                         core_rsp_ready,

    // Memory request
    output wire                         mem_req_valid,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] mem_req_addr,
    output wire                         mem_req_rw,
    output wire [LINE_SIZE-1:0]         mem_req_byteen,
    output wire [`CS_LINE_WIDTH-1:0]    mem_req_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   mem_req_id,     // index of the head entry in the mshr
    output wire                         mem_req_flush,
    input  wire                         mem_req_ready,

    // Memory response
    input wire                          mem_rsp_valid,
    input wire [`CS_LINE_WIDTH-1:0]     mem_rsp_data,
    input wire [MSHR_ADDR_WIDTH-1:0]    mem_rsp_id,
    output wire                         mem_rsp_ready,

    // flush
    input wire                          flush_begin,
    output wire                         flush_end
);

    localparam PIPELINE_STAGES = 2;

`IGNORE_UNUSED_BEGIN
    wire [`UP(UUID_WIDTH)-1:0] req_uuid_sel, req_uuid_st0, req_uuid_st1;
`IGNORE_UNUSED_END

    wire                            crsp_queue_stall;
    wire                            mshr_alm_full;
    wire                            mreq_queue_empty;
    wire                            mreq_queue_alm_full;

    wire [`CS_LINE_ADDR_WIDTH-1:0]  mem_rsp_addr;

    wire                            replay_valid;
    wire [`CS_LINE_ADDR_WIDTH-1:0]  replay_addr;
    wire                            replay_rw;
    wire [WORD_SEL_WIDTH-1:0]       replay_wsel;
    wire [WORD_SIZE-1:0]            replay_byteen;
    wire [`CS_WORD_WIDTH-1:0]       replay_data;
    wire [TAG_WIDTH-1:0]            replay_tag;
    wire [REQ_SEL_WIDTH-1:0]        replay_idx;
    wire [MSHR_ADDR_WIDTH-1:0]      replay_id;
    wire                            replay_ready;

    wire                            is_init_st0, is_init_st1;
    wire                            is_flush_st0, is_flush_st1;
    wire [NUM_WAYS-1:0]             flush_way_st0;

    wire [`CS_LINE_ADDR_WIDTH-1:0]  addr_sel, addr_st0, addr_st1;
    wire [`CS_LINE_SEL_BITS-1:0]    line_sel_st0, line_sel_st1;
    wire                            rw_sel, rw_st0, rw_st1;
    wire [WORD_SEL_WIDTH-1:0]       wsel_sel, wsel_st0, wsel_st1;
    wire [WORD_SIZE-1:0]            byteen_sel, byteen_st0, byteen_st1;
    wire [REQ_SEL_WIDTH-1:0]        req_idx_sel, req_idx_st0, req_idx_st1;
    wire [TAG_WIDTH-1:0]            tag_sel, tag_st0, tag_st1;
    wire [`CS_WORD_WIDTH-1:0]       read_data_st1;
    wire [`CS_LINE_WIDTH-1:0]       data_sel, data_st0, data_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      replay_id_st0, mshr_id_st0, mshr_id_st1;
    wire                            valid_sel, valid_st0, valid_st1;
    wire                            is_creq_st0, is_creq_st1;
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_replay_st0, is_replay_st1;
    wire                            creq_flush_sel, creq_flush_st0, creq_flush_st1;
    wire                            evict_dirty_st0, evict_dirty_st1;
    wire [NUM_WAYS-1:0]             way_sel_st0, way_sel_st1;
    wire [NUM_WAYS-1:0]             tag_matches_st0;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_alloc_id_st0;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_prev_st0, mshr_prev_st1;
    wire                            mshr_pending_st0, mshr_pending_st1;
    wire                            mshr_empty;

    wire flush_valid;
    wire init_valid;
    wire [`CS_LINE_SEL_BITS-1:0] flush_sel;
    wire [NUM_WAYS-1:0] flush_way;
    wire flush_ready;

    // ensure we have no pending memory request in the bank
    wire no_pending_req = ~valid_st0 && ~valid_st1 && mreq_queue_empty;

    // this reset relay should match pipeline during tags initialization
    `RESET_RELAY (flush_reset, reset);

    // flush unit
    VX_bank_flush #(
        .BANK_ID    (BANK_ID),
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WRITEBACK  (WRITEBACK)
    ) flush_unit (
        .clk         (clk),
        .reset       (flush_reset),
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

    wire rdw_hazard1_sel;
    wire rdw_hazard2_sel;
    reg rdw_hazard3_st1;

    wire pipe_stall = crsp_queue_stall || rdw_hazard3_st1;

    // inputs arbitration:
    // mshr replay has highest priority to maximize utilization since there is no miss.
    // handle memory responses next to prevent deadlock with potential memory request from a miss.
    // flush has precedence over core requests to ensure that the cache is in a consistent state.
    wire replay_grant = ~init_valid;
    wire replay_enable = replay_grant && replay_valid;

    wire fill_grant  = ~init_valid && ~replay_enable;
    wire fill_enable = fill_grant && mem_rsp_valid;

    wire flush_grant  = ~init_valid && ~replay_enable && ~fill_enable;
    wire flush_enable = flush_grant && flush_valid;

    wire creq_grant  = ~init_valid && ~replay_enable && ~fill_enable && ~flush_enable;
    wire creq_enable = creq_grant && core_req_valid;

    assign replay_ready = replay_grant
                       && ~rdw_hazard1_sel
                       && ~pipe_stall;

    assign mem_rsp_ready = fill_grant
                        && (!WRITEBACK || ~mreq_queue_alm_full) // needed for evictions
                        && ~rdw_hazard2_sel
                        && ~pipe_stall;

    assign flush_ready = flush_grant
                      && (!WRITEBACK || ~mreq_queue_alm_full) // needed for evictions
                      && ~rdw_hazard2_sel
                      && ~pipe_stall;

    assign core_req_ready = creq_grant
                         && ~mreq_queue_alm_full
                         && ~mshr_alm_full
                         && ~pipe_stall;

    wire init_fire     = init_valid;
    wire replay_fire   = replay_valid && replay_ready;
    wire mem_rsp_fire  = mem_rsp_valid && mem_rsp_ready;
    wire flush_fire = flush_valid && flush_ready;
    wire core_req_fire = core_req_valid && core_req_ready;

    assign valid_sel   = init_fire || replay_fire || mem_rsp_fire || flush_fire || core_req_fire;
    assign rw_sel      = replay_valid ? replay_rw : core_req_rw;
    assign byteen_sel  = replay_valid ? replay_byteen : core_req_byteen;
    assign wsel_sel    = replay_valid ? replay_wsel : core_req_wsel;
    assign req_idx_sel = replay_valid ? replay_idx : core_req_idx;
    assign tag_sel     = replay_valid ? replay_tag : core_req_tag;
    assign creq_flush_sel = core_req_valid && core_req_flush;

    assign addr_sel    = (init_valid | flush_valid) ? `CS_LINE_ADDR_WIDTH'(flush_sel) :
                            (replay_valid ? replay_addr : (mem_rsp_valid ? mem_rsp_addr : core_req_addr));

    if (WRITE_ENABLE) begin
        assign data_sel[`CS_WORD_WIDTH-1:0] = replay_valid ? replay_data : (mem_rsp_valid ? mem_rsp_data[`CS_WORD_WIDTH-1:0] : core_req_data);
    end else begin
        assign data_sel[`CS_WORD_WIDTH-1:0] = mem_rsp_data[`CS_WORD_WIDTH-1:0];
        `UNUSED_VAR (core_req_data)
        `UNUSED_VAR (replay_data)
    end
    for (genvar i = `CS_WORD_WIDTH; i < `CS_LINE_WIDTH; ++i) begin
        assign data_sel[i] = mem_rsp_data[i]; // only the memory response fills the upper words of data_sel
    end

    if (UUID_WIDTH != 0) begin
        assign req_uuid_sel = tag_sel[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_sel = 0;
    end

    `RESET_RELAY (pipe0_reset, reset);

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + NUM_WAYS + `CS_LINE_ADDR_WIDTH + `CS_LINE_WIDTH + 1 + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (pipe0_reset),
        .enable   (~pipe_stall),
        .data_in  ({valid_sel, init_valid,  replay_enable, fill_enable, flush_enable, creq_enable, creq_flush_sel, flush_way,      addr_sel, data_sel, rw_sel, byteen_sel, wsel_sel, req_idx_sel, tag_sel, replay_id}),
        .data_out ({valid_st0, is_init_st0, is_replay_st0, is_fill_st0, is_flush_st0, is_creq_st0, creq_flush_st0, flush_way_st0,  addr_st0, data_st0, rw_st0, byteen_st0, wsel_st0, req_idx_st0, tag_st0, replay_id_st0})
    );

    if (UUID_WIDTH != 0) begin
        assign req_uuid_st0 = tag_st0[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_st0 = 0;
    end

    wire do_init_st0    = valid_st0 && is_init_st0;
    wire do_flush_st0   = valid_st0 && is_flush_st0;
    wire do_creq_rd_st0 = valid_st0 && is_creq_st0 && ~rw_st0;
    wire do_creq_wr_st0 = valid_st0 && is_creq_st0 && rw_st0;
    wire do_replay_rd_st0 = valid_st0 && is_replay_st0 && ~rw_st0;
    wire do_replay_wr_st0 = valid_st0 && is_replay_st0 && rw_st0;
    wire do_fill_st0    = valid_st0 && is_fill_st0;
    wire do_cache_rd_st0 = do_creq_rd_st0 || do_replay_rd_st0;
    wire do_cache_wr_st0 = do_creq_wr_st0 || do_replay_wr_st0;
    wire do_lookup_st0  = do_cache_rd_st0 || do_cache_wr_st0;

    wire [`CS_WORD_WIDTH-1:0] write_data_st0 = data_st0[`CS_WORD_WIDTH-1:0];

    assign line_sel_st0 = addr_st0[`CS_LINE_SEL_BITS-1:0];

    wire [NUM_WAYS-1:0] evict_way_st0;
    wire [`CS_TAG_SEL_BITS-1:0] evict_tag_st0;

    `RESET_RELAY (tags_reset, reset);

    VX_cache_tags #(
        .INSTANCE_ID($sformatf("%s-tags", INSTANCE_ID)),
        .BANK_ID    (BANK_ID),
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WORD_SIZE  (WORD_SIZE),
        .WRITEBACK  (WRITEBACK),
        .UUID_WIDTH (UUID_WIDTH)
    ) cache_tags (
        .clk        (clk),
        .reset      (tags_reset),

        .req_uuid   (req_uuid_st0),

        .stall      (pipe_stall),

        // init/flush/fill/write/lookup
        .init       (do_init_st0),
        .flush      (do_flush_st0),
        .fill       (do_fill_st0),
        .write      (do_cache_wr_st0),
        .lookup     (do_lookup_st0),
        .line_addr  (addr_st0),
        .way_sel    (flush_way_st0),
        .tag_matches(tag_matches_st0),

        // replacement
        .evict_dirty(evict_dirty_st0),
        .evict_way  (evict_way_st0),
        .evict_tag  (evict_tag_st0)
    );

    wire [`CS_LINE_ADDR_WIDTH-1:0] addr2_st0;

    wire is_flush2_st0 = WRITEBACK && is_flush_st0;

    assign mshr_id_st0 = is_creq_st0 ? mshr_alloc_id_st0 : replay_id_st0;

    assign way_sel_st0 = (is_fill_st0 || is_flush2_st0) ? evict_way_st0 : tag_matches_st0;

    assign addr2_st0 = (is_fill_st0 || is_flush2_st0) ? {evict_tag_st0, line_sel_st0} : addr_st0;

    `RESET_RELAY (pipe1_reset, reset);

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + `CS_LINE_ADDR_WIDTH + `CS_LINE_WIDTH + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH + MSHR_ADDR_WIDTH + NUM_WAYS + 1 + 1),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (pipe1_reset),
        .enable   (~pipe_stall),
        .data_in  ({valid_st0, is_init_st0, is_replay_st0, is_fill_st0, is_flush2_st0, is_creq_st0, creq_flush_st0, rw_st0, addr2_st0, data_st0, byteen_st0, wsel_st0, req_idx_st0, tag_st0, mshr_id_st0, mshr_prev_st0, way_sel_st0, evict_dirty_st0, mshr_pending_st0}),
        .data_out ({valid_st1, is_init_st1, is_replay_st1, is_fill_st1, is_flush_st1,  is_creq_st1, creq_flush_st1, rw_st1, addr_st1,  data_st1, byteen_st1, wsel_st1, req_idx_st1, tag_st1, mshr_id_st1, mshr_prev_st1, way_sel_st1, evict_dirty_st1, mshr_pending_st1})
    );

    // we have a tag hit
    wire is_hit_st1 = (| way_sel_st1);

    if (UUID_WIDTH != 0) begin
        assign req_uuid_st1 = tag_st1[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_st1 = 0;
    end

    wire is_read_st1      = is_creq_st1 && ~rw_st1;
    wire is_write_st1     = is_creq_st1 && rw_st1;

    wire do_init_st1      = valid_st1 && is_init_st1;
    wire do_fill_st1      = valid_st1 && is_fill_st1;
    wire do_flush_st1     = valid_st1 && is_flush_st1;

    wire do_creq_rd_st1   = valid_st1 && is_read_st1;
    wire do_creq_wr_st1   = valid_st1 && is_write_st1;
    wire do_replay_rd_st1 = valid_st1 && is_replay_st1 && ~rw_st1;
    wire do_replay_wr_st1 = valid_st1 && is_replay_st1 && rw_st1;

    wire do_read_hit_st1  = do_creq_rd_st1 && is_hit_st1;
    wire do_read_miss_st1 = do_creq_rd_st1 && ~is_hit_st1;

    wire do_write_hit_st1 = do_creq_wr_st1 && is_hit_st1;
    wire do_write_miss_st1= do_creq_wr_st1 && ~is_hit_st1;

    wire do_cache_rd_st1  = do_read_hit_st1 || do_replay_rd_st1;
    wire do_cache_wr_st1  = do_write_hit_st1 || do_replay_wr_st1;

    assign line_sel_st1 = addr_st1[`CS_LINE_SEL_BITS-1:0];

    `UNUSED_VAR (do_write_miss_st1)

    // ensure mshr replay always get a hit
    `RUNTIME_ASSERT (~(valid_st1 && is_replay_st1) || is_hit_st1, ("missed mshr replay"));

    // both tag and data stores use BRAM with no read-during-write protection.
    // we ned to stall the pipeline to prevent read-after-write hazards.
    assign rdw_hazard1_sel = do_fill_st0; // stall first replay following a fill
    assign rdw_hazard2_sel = WRITEBACK && do_cache_wr_st0; // a writeback can evict any preceeding write
    always @(posedge clk) begin
        // stall reads following writes to same line address
        rdw_hazard3_st1 <= do_cache_rd_st0 && do_cache_wr_st1 && (line_sel_st0 == line_sel_st1)
                       && ~rdw_hazard3_st1; // release pipeline stall
    end

    wire [`CS_LINE_WIDTH-1:0] write_data_st1 = {`CS_WORDS_PER_LINE{data_st1[`CS_WORD_WIDTH-1:0]}};
    wire [`CS_LINE_WIDTH-1:0] fill_data_st1 = data_st1;
    wire [LINE_SIZE-1:0] write_byteen_st1;

    wire [`CS_LINE_WIDTH-1:0] dirty_data_st1;
    wire [LINE_SIZE-1:0] dirty_byteen_st1;

     if (`CS_WORDS_PER_LINE > 1) begin
        reg [LINE_SIZE-1:0] write_byteen_r;
        always @(*) begin
            write_byteen_r = '0;
            write_byteen_r[wsel_st1 * WORD_SIZE +: WORD_SIZE] = byteen_st1;
        end
        assign write_byteen_st1 = write_byteen_r;
    end else begin
        assign write_byteen_st1 = byteen_st1;
    end

    `RESET_RELAY (data_reset, reset);

    VX_cache_data #(
        .INSTANCE_ID  ($sformatf("%s-data", INSTANCE_ID)),
        .BANK_ID      (BANK_ID),
        .CACHE_SIZE   (CACHE_SIZE),
        .LINE_SIZE    (LINE_SIZE),
        .NUM_BANKS    (NUM_BANKS),
        .NUM_WAYS     (NUM_WAYS),
        .WORD_SIZE    (WORD_SIZE),
        .WRITE_ENABLE (WRITE_ENABLE),
        .WRITEBACK    (WRITEBACK),
        .DIRTY_BYTES  (DIRTY_BYTES),
        .UUID_WIDTH   (UUID_WIDTH)
    ) cache_data (
        .clk        (clk),
        .reset      (data_reset),

        .req_uuid   (req_uuid_st1),

        .stall      (pipe_stall),

        .init       (do_init_st1),
        .read       (do_cache_rd_st1),
        .fill       (do_fill_st1),
        .flush      (do_flush_st1),
        .write      (do_cache_wr_st1),
        .way_sel    (way_sel_st1),
        .line_addr  (addr_st1),
        .wsel       (wsel_st1),
        .fill_data  (fill_data_st1),
        .write_data (write_data_st1),
        .write_byteen(write_byteen_st1),
        .read_data  (read_data_st1),
        .dirty_data (dirty_data_st1),
        .dirty_byteen(dirty_byteen_st1)
    );

    wire [MSHR_SIZE-1:0] mshr_lookup_pending_st0;
    wire [MSHR_SIZE-1:0] mshr_lookup_rw_st0;
    wire mshr_allocate_st0 = valid_st0 && is_creq_st0 && ~pipe_stall;
    wire mshr_lookup_st0   = mshr_allocate_st0;
    wire mshr_finalize_st1 = valid_st1 && is_creq_st1 && ~pipe_stall;

    // release allocated mshr entry if we had a hit
    wire mshr_release_st1;
    if (WRITEBACK) begin
        assign mshr_release_st1 = is_hit_st1;
    end else begin
        // we need to keep missed write requests in MSHR if there is already a pending entry to the same address
        // this ensures that missed write requests are replayed locally in case a pending fill arrives without the write content
        // this can happen when writes are sent late, when the fill was already in flight.
        assign mshr_release_st1 = is_hit_st1 || (rw_st1 && ~mshr_pending_st1);
    end

    VX_pending_size #(
        .SIZE (MSHR_SIZE)
    ) mshr_pending_size (
        .clk   (clk),
        .reset (reset),
        .incr  (core_req_fire),
        .decr  (replay_fire || (mshr_finalize_st1 && mshr_release_st1)),
        .empty (mshr_empty),
        `UNUSED_PIN (alm_empty),
        .full  (mshr_alm_full),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    `RESET_RELAY (mshr_reset, reset);

    VX_cache_mshr #(
        .INSTANCE_ID ($sformatf("%s-mshr", INSTANCE_ID)),
        .BANK_ID     (BANK_ID),
        .LINE_SIZE   (LINE_SIZE),
        .NUM_BANKS   (NUM_BANKS),
        .MSHR_SIZE   (MSHR_SIZE),
        .UUID_WIDTH  (UUID_WIDTH),
        .DATA_WIDTH  (WORD_SEL_WIDTH + WORD_SIZE + `CS_WORD_WIDTH + TAG_WIDTH + REQ_SEL_WIDTH)
    ) cache_mshr (
        .clk            (clk),
        .reset          (mshr_reset),

        .deq_req_uuid   (req_uuid_sel),
        .lkp_req_uuid   (req_uuid_st0),
        .fin_req_uuid   (req_uuid_st1),

        // memory fill
        .fill_valid     (mem_rsp_fire),
        .fill_id        (mem_rsp_id),
        .fill_addr      (mem_rsp_addr),

        // dequeue
        .dequeue_valid  (replay_valid),
        .dequeue_addr   (replay_addr),
        .dequeue_rw     (replay_rw),
        .dequeue_data   ({replay_wsel, replay_byteen, replay_data, replay_tag, replay_idx}),
        .dequeue_id     (replay_id),
        .dequeue_ready  (replay_ready),

        // allocate
        .allocate_valid (mshr_allocate_st0),
        .allocate_addr  (addr_st0),
        .allocate_rw    (rw_st0),
        .allocate_data  ({wsel_st0, byteen_st0, write_data_st0, tag_st0, req_idx_st0}),
        .allocate_id    (mshr_alloc_id_st0),
        .allocate_prev  (mshr_prev_st0),
        `UNUSED_PIN     (allocate_ready),

        // lookup
        .lookup_valid   (mshr_lookup_st0),
        .lookup_addr    (addr_st0),
        .lookup_pending (mshr_lookup_pending_st0),
        .lookup_rw      (mshr_lookup_rw_st0),

        // finalize
        .finalize_valid (mshr_finalize_st1),
        .finalize_release(mshr_release_st1),
        .finalize_pending(mshr_pending_st1),
        .finalize_id    (mshr_id_st1),
        .finalize_prev  (mshr_prev_st1)
    );

    // check if there are pending requests to same line in the MSHR
    wire [MSHR_SIZE-1:0] lookup_matches;
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin
        assign lookup_matches[i] = mshr_lookup_pending_st0[i]
                                && (i != mshr_alloc_id_st0) // exclude current mshr id
                                && (WRITEBACK || ~mshr_lookup_rw_st0[i]);  // exclude write requests if writethrough
    end
    assign mshr_pending_st0 = (| lookup_matches);

    // schedule core response

    wire crsp_queue_valid, crsp_queue_ready;
    wire [`CS_WORD_WIDTH-1:0] crsp_queue_data;
    wire [REQ_SEL_WIDTH-1:0] crsp_queue_idx;
    wire [TAG_WIDTH-1:0] crsp_queue_tag;

    assign crsp_queue_valid = do_cache_rd_st1;
    assign crsp_queue_idx   = req_idx_st1;
    assign crsp_queue_data  = read_data_st1;
    assign crsp_queue_tag   = tag_st1;

    `RESET_RELAY (crsp_queue_reset, reset);

    VX_elastic_buffer #(
        .DATAW   (TAG_WIDTH + `CS_WORD_WIDTH + REQ_SEL_WIDTH),
        .SIZE    (CRSQ_SIZE),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) core_rsp_queue (
        .clk       (clk),
        .reset     (crsp_queue_reset),
        .valid_in  (crsp_queue_valid && ~rdw_hazard3_st1),
        .ready_in  (crsp_queue_ready),
        .data_in   ({crsp_queue_tag, crsp_queue_data, crsp_queue_idx}),
        .data_out  ({core_rsp_tag, core_rsp_data, core_rsp_idx}),
        .valid_out (core_rsp_valid),
        .ready_out (core_rsp_ready)
    );

    assign crsp_queue_stall = crsp_queue_valid && ~crsp_queue_ready;

    // schedule memory request

    wire mreq_queue_push, mreq_queue_pop;
    wire [`CS_LINE_WIDTH-1:0] mreq_queue_data;
    wire [LINE_SIZE-1:0] mreq_queue_byteen;
    wire [`CS_LINE_ADDR_WIDTH-1:0] mreq_queue_addr;
    wire [MSHR_ADDR_WIDTH-1:0] mreq_queue_id;
    wire mreq_queue_rw;
    wire mreq_queue_flush;

    wire is_fill_or_flush_st1 = is_fill_st1 || is_flush_st1;
    wire do_fill_or_flush_st1 = valid_st1 && is_fill_or_flush_st1;
    wire do_writeback_st1 = do_fill_or_flush_st1 && evict_dirty_st1;

    if (WRITEBACK) begin
        if (DIRTY_BYTES) begin
            // ensure dirty bytes match the tag info
            wire has_dirty_bytes = (| dirty_byteen_st1);
            `RUNTIME_ASSERT (~do_fill_or_flush_st1 || (evict_dirty_st1 == has_dirty_bytes), ("missmatch dirty bytes: dirty_line=%b, dirty_bytes=%b, addr=0x%0h", evict_dirty_st1, has_dirty_bytes, `CS_LINE_TO_FULL_ADDR(addr_st1, BANK_ID)));
        end
        assign mreq_queue_push = (((do_read_miss_st1 || do_write_miss_st1) && ~mshr_pending_st1)
                               || do_writeback_st1)
                              && ~rdw_hazard3_st1;
    end else begin
        `UNUSED_VAR (do_writeback_st1)
        assign mreq_queue_push = ((do_read_miss_st1 && ~mshr_pending_st1)
                               || do_creq_wr_st1)
                              && ~rdw_hazard3_st1;
    end

    assign mreq_queue_pop = mem_req_valid && mem_req_ready;
    assign mreq_queue_addr = addr_st1;
    assign mreq_queue_id = mshr_id_st1;
    assign mreq_queue_flush = creq_flush_st1;

    if (WRITE_ENABLE) begin
        assign mreq_queue_rw = WRITEBACK ? is_fill_or_flush_st1 : rw_st1;
        assign mreq_queue_data = WRITEBACK ? dirty_data_st1 : write_data_st1;
        assign mreq_queue_byteen = WRITEBACK ? dirty_byteen_st1 : write_byteen_st1;
    end else begin
        assign mreq_queue_rw = 0;
        assign mreq_queue_data = 0;
        assign mreq_queue_byteen = 0;
        `UNUSED_VAR (dirty_data_st1)
        `UNUSED_VAR (dirty_byteen_st1)
    end

    `RESET_RELAY (mreq_queue_reset, reset);

    VX_fifo_queue #(
        .DATAW    (1 + `CS_LINE_ADDR_WIDTH + MSHR_ADDR_WIDTH + LINE_SIZE + `CS_LINE_WIDTH + 1),
        .DEPTH    (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE-PIPELINE_STAGES),
        .OUT_REG  (`TO_OUT_BUF_REG(MEM_OUT_BUF))
    ) mem_req_queue (
        .clk        (clk),
        .reset      (mreq_queue_reset),
        .push       (mreq_queue_push),
        .pop        (mreq_queue_pop),
        .data_in    ({mreq_queue_rw, mreq_queue_addr, mreq_queue_id, mreq_queue_byteen, mreq_queue_data, mreq_queue_flush}),
        .data_out   ({mem_req_rw, mem_req_addr, mem_req_id, mem_req_byteen, mem_req_data, mem_req_flush}),
        .empty      (mreq_queue_empty),
        .alm_full   (mreq_queue_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign mem_req_valid = ~mreq_queue_empty;

///////////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    assign perf_read_misses  = do_read_miss_st1;
    assign perf_write_misses = do_write_miss_st1;
    assign perf_mshr_stalls  = mshr_alm_full;
`endif

`ifdef DBG_TRACE_CACHE
    wire crsp_queue_fire = crsp_queue_valid && crsp_queue_ready;
    wire input_stall = (replay_valid || mem_rsp_valid || core_req_valid || flush_valid)
                   && ~(replay_fire || mem_rsp_fire || core_req_fire || flush_fire);
    always @(posedge clk) begin
        if (input_stall || pipe_stall) begin
            `TRACE(3, ("%d: *** %s stall: crsq=%b, mreq=%b, mshr=%b, rdw1=%b, rdw2=%b, rdw3=%b\n", $time, INSTANCE_ID, crsp_queue_stall, mreq_queue_alm_full, mshr_alm_full, rdw_hazard1_sel, rdw_hazard2_sel, rdw_hazard3_st1));
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%d: %s fill-rsp: addr=0x%0h, mshr_id=%0d, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(mem_rsp_addr, BANK_ID), mem_rsp_id, mem_rsp_data));
        end
        if (replay_fire) begin
            `TRACE(2, ("%d: %s mshr-pop: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(replay_addr, BANK_ID), replay_tag, replay_idx, req_uuid_sel));
        end
        if (core_req_fire) begin
            if (core_req_rw)
                `TRACE(2, ("%d: %s core-wr-req: addr=0x%0h, tag=0x%0h, req_idx=%0d, byteen=%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(core_req_addr, BANK_ID), core_req_tag, core_req_idx, core_req_byteen, core_req_data, req_uuid_sel));
            else
                `TRACE(2, ("%d: %s core-rd-req: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(core_req_addr, BANK_ID), core_req_tag, core_req_idx, req_uuid_sel));
        end
        if (crsp_queue_fire) begin
            `TRACE(2, ("%d: %s core-rd-rsp: addr=0x%0h, tag=0x%0h, req_idx=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(addr_st1, BANK_ID), crsp_queue_tag, crsp_queue_idx, crsp_queue_data, req_uuid_st1));
        end
        if (mreq_queue_push) begin
            if (do_creq_wr_st1 && !WRITEBACK)
                `TRACE(2, ("%d: %s writethrough: addr=0x%0h, byteen=%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(mreq_queue_addr, BANK_ID), mreq_queue_byteen, mreq_queue_data, req_uuid_st1));
            else if (do_writeback_st1)
                `TRACE(2, ("%d: %s writeback: addr=0x%0h, byteen=%h, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(mreq_queue_addr, BANK_ID), mreq_queue_byteen, mreq_queue_data));
            else
                `TRACE(2, ("%d: %s fill-req: addr=0x%0h, mshr_id=%0d (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(mreq_queue_addr, BANK_ID), mreq_queue_id, req_uuid_st1));
        end
    end
`endif

endmodule
