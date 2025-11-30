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

    // Replacement policy
    parameter REPL_POLICY       = `CS_REPL_FIFO,

    // Request debug identifier
    parameter UUID_WIDTH        = 0,

    // core request tag size
    parameter TAG_WIDTH         = UUID_WIDTH + 1,

    // core request flags
    parameter FLAGS_WIDTH       = 0,

    // Core response output register
    parameter CORE_OUT_REG      = 0,

    // Memory request output register
    parameter MEM_OUT_REG       = 0,

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
    output wire perf_mshr_stall,
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
    input wire [`UP(FLAGS_WIDTH)-1:0]   core_req_flags,
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
    output wire [MEM_TAG_WIDTH-1:0]     mem_req_tag,
    output wire [`UP(FLAGS_WIDTH)-1:0]  mem_req_flags,
    input  wire                         mem_req_ready,

    // Memory response
    input wire                          mem_rsp_valid,
    input wire [`CS_LINE_WIDTH-1:0]     mem_rsp_data,
    input wire [MEM_TAG_WIDTH-1:0]      mem_rsp_tag,
    output wire                         mem_rsp_ready,

    // flush
    input wire                          flush_begin,
    input wire [`UP(UUID_WIDTH)-1:0]    flush_uuid,
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


    wire                            valid_sel, valid_st0, valid_st1;
    wire                            is_init_st0;
    wire                            is_creq_st0, is_creq_st1;
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_flush_st0, is_flush_st1;
    wire [`CS_WAY_SEL_WIDTH-1:0]    flush_way_st0, evict_way_st0;
    wire [`CS_WAY_SEL_WIDTH-1:0]    way_idx_st0, way_idx_st1;

    wire [`CS_LINE_ADDR_WIDTH-1:0]  addr_sel, addr_st0, addr_st1;
    wire [`CS_LINE_SEL_BITS-1:0]    line_idx_st0, line_idx_st1;
    wire [`CS_TAG_SEL_BITS-1:0]     line_tag_st0, line_tag_st1;
    wire [`CS_TAG_SEL_BITS-1:0]     evict_tag_st0, evict_tag_st1;
    wire                            rw_sel, rw_st0, rw_st1;
    wire [WORD_SEL_WIDTH-1:0]       word_idx_sel, word_idx_st0, word_idx_st1;
    wire [WORD_SIZE-1:0]            byteen_sel, byteen_st0, byteen_st1;
    wire [REQ_SEL_WIDTH-1:0]        req_idx_sel, req_idx_st0, req_idx_st1;
    wire [TAG_WIDTH-1:0]            tag_sel, tag_st0, tag_st1;
    wire [`CS_WORD_WIDTH-1:0]       write_word_st0, write_word_st1;
    wire [`CS_LINE_WIDTH-1:0]       data_sel, data_st0, data_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_id_st0, mshr_id_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      replay_id_st0;
    wire                            is_dirty_st0, is_dirty_st1;
    wire                            is_replay_st0, is_replay_st1;
    wire                            is_hit_st0, is_hit_st1;
    wire [`UP(FLAGS_WIDTH)-1:0]     flags_sel, flags_st0, flags_st1;
    wire                            mshr_pending_st0, mshr_pending_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_previd_st0, mshr_previd_st1;
    wire                            mshr_empty;

    wire flush_valid;
    wire init_valid;
    wire [`CS_LINE_SEL_BITS-1:0] flush_sel;
    wire [`CS_WAY_SEL_WIDTH-1:0] flush_way;
    wire flush_ready;

    // ensure we have no pending memory request in the bank
    wire no_pending_req = ~valid_st0 && ~valid_st1 && mreq_queue_empty;

    VX_bank_flush #(
        .BANK_ID    (BANK_ID),
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WRITEBACK  (WRITEBACK)
    ) flush_unit (
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

    wire pipe_stall = crsp_queue_stall;

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
                       && ~(!WRITEBACK && replay_rw && mreq_queue_alm_full) // needed for writethrough
                       && ~pipe_stall;

    assign mem_rsp_ready = fill_grant
                        && ~(WRITEBACK && mreq_queue_alm_full) // needed for writeback
                        && ~pipe_stall;

    assign flush_ready = flush_grant
                      && ~(WRITEBACK && mreq_queue_alm_full) // needed for writeback
                      && ~pipe_stall;

    assign core_req_ready = creq_grant
                         && ~mreq_queue_alm_full // needed for fill requests
                         && ~mshr_alm_full // needed for mshr allocation
                         && ~pipe_stall;

    wire init_fire     = init_valid;
    wire replay_fire   = replay_valid && replay_ready;
    wire mem_rsp_fire  = mem_rsp_valid && mem_rsp_ready;
    wire flush_fire    = flush_valid && flush_ready;
    wire core_req_fire = core_req_valid && core_req_ready;

    wire [MSHR_ADDR_WIDTH-1:0] mem_rsp_id = mem_rsp_tag[MSHR_ADDR_WIDTH-1:0];

    wire [TAG_WIDTH-1:0] mem_rsp_tag_s;
    if (TAG_WIDTH > MEM_TAG_WIDTH) begin : g_mem_rsp_tag_s_pad
        assign mem_rsp_tag_s = {mem_rsp_tag, (TAG_WIDTH-MEM_TAG_WIDTH)'(1'b0)};
    end else begin : g_mem_rsp_tag_s_cut
        assign mem_rsp_tag_s = mem_rsp_tag[MEM_TAG_WIDTH-1 -: TAG_WIDTH];
        `UNUSED_VAR (mem_rsp_tag)
    end

    wire [TAG_WIDTH-1:0] flush_tag;
    if (UUID_WIDTH != 0) begin : g_flush_tag_uuid
        assign flush_tag = {flush_uuid, (TAG_WIDTH-UUID_WIDTH)'(1'b0)};
    end else begin : g_flush_tag_0
        `UNUSED_VAR (flush_uuid)
        assign flush_tag = '0;
    end

    assign valid_sel   = init_fire || replay_fire || mem_rsp_fire || flush_fire || core_req_fire;
    assign rw_sel      = replay_valid ? replay_rw : core_req_rw;
    assign byteen_sel  = replay_valid ? replay_byteen : core_req_byteen;
    assign addr_sel    = (init_valid | flush_valid) ? `CS_LINE_ADDR_WIDTH'(flush_sel) :
                            (replay_valid ? replay_addr : (mem_rsp_valid ? mem_rsp_addr : core_req_addr));
    assign word_idx_sel= replay_valid ? replay_wsel : core_req_wsel;
    assign req_idx_sel = replay_valid ? replay_idx : core_req_idx;
    assign tag_sel     = (init_valid | flush_valid) ? (flush_valid ? flush_tag : '0) :
                            (replay_valid ? replay_tag : (mem_rsp_valid ? mem_rsp_tag_s : core_req_tag));
    assign flags_sel   = core_req_valid ? core_req_flags : '0;

    if (WRITE_ENABLE) begin : g_data_sel
        for (genvar i = 0; i < `CS_LINE_WIDTH; ++i) begin : g_i
            if (i < `CS_WORD_WIDTH) begin : g_lo
                assign data_sel[i] = replay_valid ? replay_data[i] : (mem_rsp_valid ? mem_rsp_data[i] : core_req_data[i]);
            end else begin : g_hi
                assign data_sel[i] = mem_rsp_data[i]; // only the memory response fills the upper words of data_sel
            end
        end
    end else begin : g_data_sel_ro
        assign data_sel = mem_rsp_data;
        `UNUSED_VAR (core_req_data)
        `UNUSED_VAR (replay_data)
    end

    if (UUID_WIDTH != 0) begin : g_req_uuid_sel
        assign req_uuid_sel = tag_sel[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_req_uuid_sel_0
        assign req_uuid_sel = '0;
    end

    wire is_init_sel   = init_valid;
    wire is_creq_sel   = creq_enable || replay_enable;
    wire is_fill_sel   = fill_enable;
    wire is_flush_sel  = flush_enable;
    wire is_replay_sel = replay_enable;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + `UP(FLAGS_WIDTH) + `CS_WAY_SEL_WIDTH + `CS_LINE_ADDR_WIDTH + `CS_LINE_WIDTH + 1 + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  ({valid_sel, is_init_sel, is_fill_sel, is_flush_sel, is_creq_sel, is_replay_sel, flags_sel, flush_way,     addr_sel, data_sel, rw_sel, byteen_sel, word_idx_sel, req_idx_sel, tag_sel, replay_id}),
        .data_out ({valid_st0, is_init_st0, is_fill_st0, is_flush_st0, is_creq_st0, is_replay_st0, flags_st0, flush_way_st0, addr_st0, data_st0, rw_st0, byteen_st0, word_idx_st0, req_idx_st0, tag_st0, replay_id_st0})
    );

    if (UUID_WIDTH != 0) begin : g_req_uuid_st0
        assign req_uuid_st0 = tag_st0[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_req_uuid_st0_0
        assign req_uuid_st0 = '0;
    end

    wire is_read_st0  = is_creq_st0 && ~rw_st0;
    wire is_write_st0 = is_creq_st0 && rw_st0;

    wire do_init_st0  = valid_st0 && is_init_st0;
    wire do_flush_st0 = valid_st0 && is_flush_st0;
    wire do_read_st0  = valid_st0 && is_read_st0;
    wire do_write_st0 = valid_st0 && is_write_st0;
    wire do_fill_st0  = valid_st0 && is_fill_st0;

    wire is_read_st1  = is_creq_st1 && ~rw_st1;
    wire is_write_st1 = is_creq_st1 && rw_st1;

    wire do_read_st1  = valid_st1 && is_read_st1;
    wire do_write_st1 = valid_st1 && is_write_st1;

    assign line_idx_st0 = addr_st0[`CS_LINE_SEL_BITS-1:0];
    assign line_tag_st0 = `CS_LINE_ADDR_TAG(addr_st0);

    assign write_word_st0 = data_st0[`CS_WORD_WIDTH-1:0];

    wire do_lookup_st0 = do_read_st0 || do_write_st0;
    wire do_lookup_st1 = do_read_st1 || do_write_st1;

    wire [`CS_WAY_SEL_WIDTH-1:0] victim_way_st0;
    wire [NUM_WAYS-1:0] tag_matches_st0;

    VX_cache_repl #(
        .CACHE_SIZE  (CACHE_SIZE),
        .LINE_SIZE   (LINE_SIZE),
        .NUM_BANKS   (NUM_BANKS),
        .NUM_WAYS    (NUM_WAYS),
        .REPL_POLICY (REPL_POLICY)
    ) cache_repl (
        .clk        (clk),
        .reset      (reset),
        .stall      (pipe_stall),
        .init       (do_init_st0),
        .lookup_valid(do_lookup_st1 && ~pipe_stall),
        .lookup_hit (is_hit_st1),
        .lookup_line(line_idx_st1),
        .lookup_way (way_idx_st1),
        .repl_valid (do_fill_st0 && ~pipe_stall),
        .repl_line  (line_idx_st0),
        .repl_way   (victim_way_st0)
    );

    assign evict_way_st0 = is_fill_st0 ? victim_way_st0 : flush_way_st0;

    VX_cache_tags #(
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WORD_SIZE  (WORD_SIZE),
        .WRITEBACK  (WRITEBACK)
    ) cache_tags (
        .clk        (clk),
        .reset      (reset),
        // inputs
        .init       (do_init_st0),
        .flush      (do_flush_st0 && ~pipe_stall),
        .fill       (do_fill_st0 && ~pipe_stall),
        .read       (do_read_st0 && ~pipe_stall),
        .write      (do_write_st0 && ~pipe_stall),
        .line_idx   (line_idx_st0),
        .line_tag   (line_tag_st0),
        .evict_way  (evict_way_st0),
        // outputs
        .tag_matches(tag_matches_st0),
        .evict_dirty(is_dirty_st0),
        .evict_tag  (evict_tag_st0)
    );

    wire [`CS_WAY_SEL_WIDTH-1:0] hit_idx_st0;
    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) way_idx_enc (
        .data_in  (tag_matches_st0),
        .data_out (hit_idx_st0),
        `UNUSED_PIN (valid_out)
    );

    assign way_idx_st0 = is_creq_st0 ? hit_idx_st0 : evict_way_st0;
    assign is_hit_st0 = (| tag_matches_st0);

    wire [MSHR_ADDR_WIDTH-1:0] mshr_alloc_id_st0;
    assign mshr_id_st0 = is_replay_st0 ? replay_id_st0 : mshr_alloc_id_st0;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + `UP(FLAGS_WIDTH) + `CS_WAY_SEL_WIDTH + `CS_TAG_SEL_BITS + `CS_TAG_SEL_BITS + `CS_LINE_SEL_BITS + `CS_LINE_WIDTH + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH + MSHR_ADDR_WIDTH + 1),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  ({valid_st0, is_fill_st0, is_flush_st0, is_creq_st0, is_replay_st0, is_dirty_st0, is_hit_st0, rw_st0, flags_st0, way_idx_st0, evict_tag_st0, line_tag_st0, line_idx_st0, data_st0, byteen_st0, word_idx_st0, req_idx_st0, tag_st0, mshr_id_st0, mshr_previd_st0, mshr_pending_st0}),
        .data_out ({valid_st1, is_fill_st1, is_flush_st1, is_creq_st1, is_replay_st1, is_dirty_st1, is_hit_st1, rw_st1, flags_st1, way_idx_st1, evict_tag_st1, line_tag_st1, line_idx_st1, data_st1, byteen_st1, word_idx_st1, req_idx_st1, tag_st1, mshr_id_st1, mshr_previd_st1, mshr_pending_st1})
    );

    if (UUID_WIDTH != 0) begin : g_req_uuid_st1
        assign req_uuid_st1 = tag_st1[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin : g_req_uuid_st1_0
        assign req_uuid_st1 = '0;
    end

    assign addr_st1 = {line_tag_st1, line_idx_st1};

    // ensure mshr replay always get a hit
    `RUNTIME_ASSERT (~(valid_st1 && is_replay_st1 && ~is_hit_st1), ("%t: missed mshr replay", $time))

    assign write_word_st1 = data_st1[`CS_WORD_WIDTH-1:0];
    `UNUSED_VAR (data_st1)

    wire[`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] read_data_st1;
    wire [LINE_SIZE-1:0] evict_byteen_st1;

    VX_cache_data #(
        .CACHE_SIZE   (CACHE_SIZE),
        .LINE_SIZE    (LINE_SIZE),
        .NUM_BANKS    (NUM_BANKS),
        .NUM_WAYS     (NUM_WAYS),
        .WORD_SIZE    (WORD_SIZE),
        .WRITE_ENABLE (WRITE_ENABLE),
        .WRITEBACK    (WRITEBACK),
        .DIRTY_BYTES  (DIRTY_BYTES)
    ) cache_data (
        .clk        (clk),
        .reset      (reset),
        // inputs
        .init       (do_init_st0),
        .fill       (do_fill_st0 && ~pipe_stall),
        .flush      (do_flush_st0 && ~pipe_stall),
        .read       (do_read_st0 && ~pipe_stall),
        .write      (do_write_st0 && ~pipe_stall),
        .evict_way  (evict_way_st0),
        .tag_matches(tag_matches_st0),
        .line_idx   (line_idx_st0),
        .fill_data  (data_st0),
        .write_word (write_word_st0),
        .word_idx   (word_idx_st0),
        .write_byteen(byteen_st0),
        .way_idx_r  (way_idx_st1),
        // outputs
        .read_data  (read_data_st1),
        .evict_byteen(evict_byteen_st1)
    );

    // only allocate MSHR entries for non-replay core requests
    wire mshr_allocate_st0 = valid_st0 && is_creq_st0 && ~is_replay_st0;
    wire mshr_finalize_st1 = valid_st1 && is_creq_st1 && ~is_replay_st1;

    // release allocated mshr entry if we had a hit
    wire mshr_release_st1;
    if (WRITEBACK) begin : g_mshr_release
        assign mshr_release_st1 = is_hit_st1;
    end else begin : g_mshr_release_ro
        // we need to keep missed write requests in MSHR if there is already a pending entry to the same address.
        // this ensures that missed write requests are replayed locally in case a pending fill arrives without the write content.
        // this can happen when writes are sent to memory late, when a related fill was already in flight.
        assign mshr_release_st1 = is_hit_st1 || (rw_st1 && ~mshr_pending_st1);
    end

    wire mshr_release_fire = mshr_finalize_st1 && mshr_release_st1 && ~pipe_stall;

    wire [1:0] mshr_dequeue;
    `POP_COUNT(mshr_dequeue, {replay_fire, mshr_release_fire});

    VX_pending_size #(
        .SIZE (MSHR_SIZE),
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
        .NUM_BANKS   (NUM_BANKS),
        .MSHR_SIZE   (MSHR_SIZE),
        .WRITEBACK   (WRITEBACK),
        .UUID_WIDTH  (UUID_WIDTH),
        .DATA_WIDTH  (WORD_SEL_WIDTH + WORD_SIZE + `CS_WORD_WIDTH + TAG_WIDTH + REQ_SEL_WIDTH)
    ) cache_mshr (
        .clk            (clk),
        .reset          (reset),

        .deq_req_uuid   (req_uuid_sel),
        .alc_req_uuid   (req_uuid_st0),
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
        .allocate_valid (mshr_allocate_st0 && ~pipe_stall),
        .allocate_addr  (addr_st0),
        .allocate_rw    (rw_st0),
        .allocate_data  ({word_idx_st0, byteen_st0, write_word_st0, tag_st0, req_idx_st0}),
        .allocate_id    (mshr_alloc_id_st0),
        .allocate_pending(mshr_pending_st0),
        .allocate_previd(mshr_previd_st0),
        `UNUSED_PIN     (allocate_ready),

        // finalize
        .finalize_valid (mshr_finalize_st1 && ~pipe_stall),
        .finalize_is_release(mshr_release_st1),
        .finalize_is_pending(mshr_pending_st1),
        .finalize_id    (mshr_id_st1),
        .finalize_previd(mshr_previd_st1)
    );

    // schedule core response

    wire crsp_queue_valid, crsp_queue_ready;
    wire [`CS_WORD_WIDTH-1:0] crsp_queue_data;
    wire [REQ_SEL_WIDTH-1:0] crsp_queue_idx;
    wire [TAG_WIDTH-1:0] crsp_queue_tag;

    assign crsp_queue_valid = do_read_st1 && is_hit_st1;
    assign crsp_queue_idx   = req_idx_st1;
    assign crsp_queue_data  = read_data_st1[word_idx_st1];
    assign crsp_queue_tag   = tag_st1;

    VX_elastic_buffer #(
        .DATAW   (TAG_WIDTH + `CS_WORD_WIDTH + REQ_SEL_WIDTH),
        .SIZE    (CRSQ_SIZE),
        .OUT_REG (CORE_OUT_REG)
    ) core_rsp_queue (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (crsp_queue_valid),
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
    wire [MEM_TAG_WIDTH-1:0] mreq_queue_tag;
    wire mreq_queue_rw;
    wire [`UP(FLAGS_WIDTH)-1:0] mreq_queue_flags;

    wire is_fill_or_flush_st1 = is_fill_st1 || (is_flush_st1 && WRITEBACK);
    wire do_fill_or_flush_st1 = valid_st1 && is_fill_or_flush_st1;
    wire do_writeback_st1 = do_fill_or_flush_st1 && is_dirty_st1;
    wire [`CS_LINE_ADDR_WIDTH-1:0] evict_addr_st1 = {evict_tag_st1, line_idx_st1};

    if (WRITE_ENABLE) begin : g_mreq_queue
        if (WRITEBACK) begin : g_wb
            if (DIRTY_BYTES) begin : g_dirty_bytes
                // ensure dirty bytes match the tag info
                wire has_dirty_bytes = (| evict_byteen_st1);
                `RUNTIME_ASSERT (~do_fill_or_flush_st1 || (is_dirty_st1 == has_dirty_bytes), ("%t: missmatch dirty bytes: dirty_line=%b, dirty_bytes=%b, addr=0x%0h", $time, is_dirty_st1, has_dirty_bytes, `CS_BANK_TO_FULL_ADDR(addr_st1, BANK_ID)))
            end
            // issue a fill request on a read/write miss
            // issue a writeback on a dirty line eviction
            assign mreq_queue_push = ((do_lookup_st1 && ~is_hit_st1 && ~mshr_pending_st1)
                                   || do_writeback_st1)
                                  && ~pipe_stall;
            assign mreq_queue_addr = is_fill_or_flush_st1 ? evict_addr_st1 : addr_st1;
            assign mreq_queue_rw = is_fill_or_flush_st1;
            assign mreq_queue_data = read_data_st1;
            assign mreq_queue_byteen = is_fill_or_flush_st1 ? evict_byteen_st1 : '1;
            `UNUSED_VAR (write_word_st1)
            `UNUSED_VAR (byteen_st1)
        end else begin : g_wt
            wire [LINE_SIZE-1:0] line_byteen;
            VX_demux #(
                .DATAW (WORD_SIZE),
                .N (`CS_WORDS_PER_LINE)
            ) byteen_demux (
                .sel_in   (word_idx_st1),
                .data_in  (byteen_st1),
                .data_out (line_byteen)
            );
            // issue a fill request on a read miss
            // issue a memory write on a write request
            assign mreq_queue_push = ((do_read_st1 && ~is_hit_st1 && ~mshr_pending_st1)
                                  || do_write_st1)
                                  && ~pipe_stall;
            assign mreq_queue_addr = addr_st1;
            assign mreq_queue_rw = rw_st1;
            assign mreq_queue_data = {`CS_WORDS_PER_LINE{write_word_st1}};
            assign mreq_queue_byteen = rw_st1 ? line_byteen : '1;
            `UNUSED_VAR (is_fill_or_flush_st1)
            `UNUSED_VAR (do_writeback_st1)
            `UNUSED_VAR (evict_addr_st1)
            `UNUSED_VAR (evict_byteen_st1)
        end
    end else begin : g_mreq_queue_ro
        // issue a fill request on a read miss
        assign mreq_queue_push = (do_read_st1 && ~is_hit_st1 && ~mshr_pending_st1)
                              && ~pipe_stall;
        assign mreq_queue_addr = addr_st1;
        assign mreq_queue_rw = 0;
        assign mreq_queue_data = '0;
        assign mreq_queue_byteen = '1;
        `UNUSED_VAR (do_writeback_st1)
        `UNUSED_VAR (evict_addr_st1)
        `UNUSED_VAR (evict_byteen_st1)
        `UNUSED_VAR (write_word_st1)
        `UNUSED_VAR (byteen_st1)
    end

    if (UUID_WIDTH != 0) begin : g_mreq_queue_tag_uuid
        assign mreq_queue_tag = {req_uuid_st1, mshr_id_st1};
    end else begin : g_mreq_queue_tag
        assign mreq_queue_tag = mshr_id_st1;
    end

    assign mreq_queue_pop = mem_req_valid && mem_req_ready;
    assign mreq_queue_flags = flags_st1;

    VX_fifo_queue #(
        .DATAW    (1 + `CS_LINE_ADDR_WIDTH + LINE_SIZE + `CS_LINE_WIDTH + MEM_TAG_WIDTH + `UP(FLAGS_WIDTH)),
        .DEPTH    (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE - PIPELINE_STAGES),
        .OUT_REG  (MEM_OUT_REG)
    ) mem_req_queue (
        .clk        (clk),
        .reset      (reset),
        .push       (mreq_queue_push),
        .pop        (mreq_queue_pop),
        .data_in    ({mreq_queue_rw, mreq_queue_addr, mreq_queue_byteen, mreq_queue_data, mreq_queue_tag, mreq_queue_flags}),
        .data_out   ({mem_req_rw,    mem_req_addr,    mem_req_byteen,    mem_req_data,    mem_req_tag,    mem_req_flags}),
        .empty      (mreq_queue_empty),
        .alm_full   (mreq_queue_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign mem_req_valid = ~mreq_queue_empty;

    `UNUSED_VAR (do_lookup_st0)

///////////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    assign perf_read_miss  = do_read_st1 && ~is_hit_st1;
    assign perf_write_miss = do_write_st1 && ~is_hit_st1;
    assign perf_mshr_stall = mshr_alm_full;
`endif

`ifdef DBG_TRACE_CACHE
    wire crsp_queue_fire = crsp_queue_valid && crsp_queue_ready;
    wire input_stall = (replay_valid || mem_rsp_valid || core_req_valid || flush_valid)
                   && ~(replay_fire || mem_rsp_fire || core_req_fire || flush_fire);

    wire [`XLEN-1:0] mem_rsp_full_addr = `CS_BANK_TO_FULL_ADDR(mem_rsp_addr, BANK_ID);
    wire [`XLEN-1:0] replay_full_addr = `CS_BANK_TO_FULL_ADDR(replay_addr, BANK_ID);
    wire [`XLEN-1:0] core_req_full_addr = `CS_BANK_TO_FULL_ADDR(core_req_addr, BANK_ID);
    wire [`XLEN-1:0] full_addr_st0 = `CS_BANK_TO_FULL_ADDR(addr_st0, BANK_ID);
    wire [`XLEN-1:0] full_addr_st1 = `CS_BANK_TO_FULL_ADDR(addr_st1, BANK_ID);
    wire [`XLEN-1:0] mreq_queue_full_addr = `CS_BANK_TO_FULL_ADDR(mreq_queue_addr, BANK_ID);

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
                full_addr_st0, evict_way_st0, line_idx_st0, is_dirty_st0, req_uuid_st0))
        end
        if (do_flush_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s tags-flush: addr=0x%0h, way=%0d, line=%0d, dirty=%b (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, evict_way_st0, line_idx_st0, is_dirty_st0, req_uuid_st0))
        end
        if (do_lookup_st0 && ~pipe_stall) begin
            if (is_hit_st0) begin
                `TRACE(3, ("%t: %s tags-hit: addr=0x%0h, rw=%b, way=%0d, line=%0d, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID,
                    full_addr_st0, rw_st0, way_idx_st0, line_idx_st0, line_tag_st0, req_uuid_st0))
            end else begin
                `TRACE(3, ("%t: %s tags-miss: addr=0x%0h, rw=%b, way=%0d, line=%0d, tag=0x%0h (#%0d)\n", $time, INSTANCE_ID,
                    full_addr_st0, rw_st0, way_idx_st0, line_idx_st0, line_tag_st0, req_uuid_st0))
            end
        end
        if (do_fill_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-fill: addr=0x%0h, way=%0d, line=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, way_idx_st0, line_idx_st0, data_st0, req_uuid_st0))
        end
        if (do_flush_st0 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-flush: addr=0x%0h, way=%0d, line=%0d (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st0, way_idx_st0, line_idx_st0, req_uuid_st0))
        end
        if (do_read_st1 && is_hit_st1 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-read: addr=0x%0h, way=%0d, line=%0d, wsel=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st1, way_idx_st1, line_idx_st1, word_idx_st1, crsp_queue_data, req_uuid_st1))
        end
        if (do_write_st1 && is_hit_st1 && ~pipe_stall) begin
            `TRACE(3, ("%t: %s data-write: addr=0x%0h, way=%0d, line=%0d, wsel=%0d, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st1, way_idx_st1, line_idx_st1, word_idx_st1, byteen_st1, write_word_st1, req_uuid_st1))
        end
        if (crsp_queue_fire) begin
            `TRACE(2, ("%t: %s core-rd-rsp: addr=0x%0h, tag=0x%0h, req_idx=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                full_addr_st1, crsp_queue_tag, crsp_queue_idx, crsp_queue_data, req_uuid_st1))
        end
        if (mreq_queue_push) begin
            if (!WRITEBACK && do_write_st1) begin
                `TRACE(2, ("%t: %s writethrough: addr=0x%0h, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, mreq_queue_byteen, mreq_queue_data, req_uuid_st1))
            end else if (WRITEBACK && do_writeback_st1) begin
                `TRACE(2, ("%t: %s writeback: addr=0x%0h, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, mreq_queue_byteen, mreq_queue_data, req_uuid_st1))
            end else begin
                `TRACE(2, ("%t: %s fill-req: addr=0x%0h, mshr_id=%0d (#%0d)\n", $time, INSTANCE_ID,
                    mreq_queue_full_addr, mshr_id_st1, req_uuid_st1))
            end
        end
    end
`endif

endmodule
