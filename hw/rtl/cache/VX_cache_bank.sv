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
    input wire                          core_req_rw,
    input wire [WORD_SEL_WIDTH-1:0]     core_req_wsel,
    input wire [WORD_SIZE-1:0]          core_req_byteen,
    input wire [`CS_WORD_WIDTH-1:0]     core_req_data,
    input wire [TAG_WIDTH-1:0]          core_req_tag,
    input wire [REQ_SEL_WIDTH-1:0]      core_req_idx,
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
    output wire [WORD_SEL_WIDTH-1:0]    mem_req_wsel,
    output wire [WORD_SIZE-1:0]         mem_req_byteen,
    output wire [`CS_WORD_WIDTH-1:0]    mem_req_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   mem_req_id,
    input  wire                         mem_req_ready,

    // Memory response
    input wire                          mem_rsp_valid,
    input wire [`CS_LINE_WIDTH-1:0]     mem_rsp_data,
    input wire [MSHR_ADDR_WIDTH-1:0]    mem_rsp_id,
    output wire                         mem_rsp_ready,

    // initialization
    input wire                          init_enable,
    input wire [`CS_LINE_SEL_BITS-1:0]  init_line_sel
);

`IGNORE_UNUSED_BEGIN
    wire [`UP(UUID_WIDTH)-1:0] req_uuid_sel, req_uuid_st0, req_uuid_st1;
`IGNORE_UNUSED_END

    wire                            crsq_stall;
    wire                            mshr_alm_full;
    wire                            mreq_alm_full;

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

    wire [`CS_LINE_ADDR_WIDTH-1:0]  addr_sel, addr_st0, addr_st1;
    wire                            rw_st0, rw_st1;
    wire [WORD_SEL_WIDTH-1:0]       wsel_st0, wsel_st1;
    wire [WORD_SIZE-1:0]            byteen_st0, byteen_st1;
    wire [REQ_SEL_WIDTH-1:0]        req_idx_st0, req_idx_st1;
    wire [TAG_WIDTH-1:0]            tag_st0, tag_st1;
    wire [`CS_WORD_WIDTH-1:0]       read_data_st1;
    wire [`CS_LINE_WIDTH-1:0]       data_sel, data_st0, data_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      replay_id_st0, mshr_id_st0, mshr_id_st1;
    wire                            valid_sel, valid_st0, valid_st1;
    wire                            is_init_st0;
    wire                            is_creq_st0, is_creq_st1;
    wire                            is_fill_st0, is_fill_st1;
    wire                            is_replay_st0, is_replay_st1;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_alloc_id_st0;
    wire [MSHR_ADDR_WIDTH-1:0]      mshr_prev_st0, mshr_prev_st1;
    wire                            mshr_pending_st0, mshr_pending_st1;

    wire rdw_hazard_st0;
    reg rdw_hazard_st1;

    wire pipe_stall = crsq_stall || rdw_hazard_st1;

    // inputs arbitration:
    // mshr replay has highest priority to maximize utilization since there is no miss.
    // handle memory responses next to prevent deadlock with potential memory request from a miss.
    wire replay_grant = ~init_enable;
    wire replay_enable = replay_grant && replay_valid;

    wire fill_grant  = ~init_enable && ~replay_enable;
    wire fill_enable = fill_grant && mem_rsp_valid;

    wire creq_grant  = ~init_enable && ~replay_enable && ~fill_enable;
    wire creq_enable = creq_grant && core_req_valid;

    assign replay_ready = replay_grant
                         && ~rdw_hazard_st0
                         && ~pipe_stall;

    assign mem_rsp_ready = fill_grant
                        && ~pipe_stall;

    assign core_req_ready = creq_grant
                        && ~mreq_alm_full
                        && ~mshr_alm_full
                        && ~pipe_stall;

    wire init_fire     = init_enable;
    wire replay_fire = replay_valid && replay_ready;
    wire mem_rsp_fire  = mem_rsp_valid && mem_rsp_ready;
    wire core_req_fire = core_req_valid && core_req_ready;

    wire [TAG_WIDTH-1:0] mshr_creq_tag = replay_enable ? replay_tag : core_req_tag;

    if (UUID_WIDTH != 0) begin
        assign req_uuid_sel = mshr_creq_tag[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_sel = 0;
    end

    `UNUSED_VAR (mshr_creq_tag)

    assign valid_sel = init_fire || replay_fire || mem_rsp_fire || core_req_fire;

    assign addr_sel = init_enable ? `CS_LINE_ADDR_WIDTH'(init_line_sel) :
                        (replay_valid ? replay_addr :
                            (mem_rsp_valid ? mem_rsp_addr : core_req_addr));

    assign data_sel[`CS_WORD_WIDTH-1:0] = (mem_rsp_valid || !WRITE_ENABLE) ? mem_rsp_data[`CS_WORD_WIDTH-1:0] : (replay_valid ? replay_data : core_req_data);
    for (genvar i = `CS_WORD_WIDTH; i < `CS_LINE_WIDTH; ++i) begin
        assign data_sel[i] = mem_rsp_data[i];
    end

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + `CS_LINE_ADDR_WIDTH + `CS_LINE_WIDTH + 1 + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH),
        .RESETW (1)
    ) pipe_reg0 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  ({
            valid_sel,
            init_enable,
            replay_enable,
            fill_enable,
            creq_enable,
            addr_sel,
            data_sel,
            replay_valid ? replay_rw : core_req_rw,
            replay_valid ? replay_byteen : core_req_byteen,
            replay_valid ? replay_wsel : core_req_wsel,
            replay_valid ? replay_idx : core_req_idx,
            replay_valid ? replay_tag : core_req_tag,
            replay_id
        }),
        .data_out ({valid_st0, is_init_st0, is_replay_st0, is_fill_st0, is_creq_st0, addr_st0, data_st0, rw_st0, byteen_st0, wsel_st0, req_idx_st0, tag_st0, replay_id_st0})
    );

    if (UUID_WIDTH != 0) begin
        assign req_uuid_st0 = tag_st0[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_st0 = 0;
    end

    wire do_creq_rd_st0 = valid_st0 && is_creq_st0 && ~rw_st0;
    wire do_fill_st0    = valid_st0 && is_fill_st0;
    wire do_init_st0    = valid_st0 && is_init_st0;
    wire do_lookup_st0  = valid_st0 && ~(is_fill_st0 || is_init_st0);

    wire [`CS_WORD_WIDTH-1:0] write_data_st0 = data_st0[`CS_WORD_WIDTH-1:0];

    wire [NUM_WAYS-1:0] tag_matches_st0, tag_matches_st1;
    wire [NUM_WAYS-1:0] way_sel_st0, way_sel_st1;

    `RESET_RELAY (tag_reset, reset);

    VX_cache_tags #(
        .INSTANCE_ID(INSTANCE_ID),
        .BANK_ID    (BANK_ID),
        .CACHE_SIZE (CACHE_SIZE),
        .LINE_SIZE  (LINE_SIZE),
        .NUM_BANKS  (NUM_BANKS),
        .NUM_WAYS   (NUM_WAYS),
        .WORD_SIZE  (WORD_SIZE),
        .UUID_WIDTH (UUID_WIDTH)
    ) cache_tags (
        .clk        (clk),
        .reset      (tag_reset),

        .req_uuid   (req_uuid_st0),

        .stall      (pipe_stall),

        // read/Fill
        .lookup     (do_lookup_st0),
        .line_addr  (addr_st0),
        .fill       (do_fill_st0),
        .init       (do_init_st0),
        .way_sel    (way_sel_st0),
        .tag_matches(tag_matches_st0)
    );

    assign mshr_id_st0 = is_creq_st0 ? mshr_alloc_id_st0 : replay_id_st0;

    VX_pipe_register #(
        .DATAW  (1 + 1 + 1 + 1 + 1 + `CS_LINE_ADDR_WIDTH + `CS_LINE_WIDTH + WORD_SIZE + WORD_SEL_WIDTH + REQ_SEL_WIDTH + TAG_WIDTH + MSHR_ADDR_WIDTH + MSHR_ADDR_WIDTH + NUM_WAYS + NUM_WAYS + 1),
        .RESETW (1)
    ) pipe_reg1 (
        .clk      (clk),
        .reset    (reset),
        .enable   (~pipe_stall),
        .data_in  ({valid_st0, is_replay_st0, is_fill_st0, is_creq_st0, rw_st0, addr_st0, data_st0, byteen_st0, wsel_st0, req_idx_st0, tag_st0, mshr_id_st0, mshr_prev_st0, tag_matches_st0, way_sel_st0, mshr_pending_st0}),
        .data_out ({valid_st1, is_replay_st1, is_fill_st1, is_creq_st1, rw_st1, addr_st1, data_st1, byteen_st1, wsel_st1, req_idx_st1, tag_st1, mshr_id_st1, mshr_prev_st1, tag_matches_st1, way_sel_st1, mshr_pending_st1})
    );

    // we have a tag hit
    wire is_hit_st1 = (| tag_matches_st1);

    if (UUID_WIDTH != 0) begin
        assign req_uuid_st1 = tag_st1[TAG_WIDTH-1 -: UUID_WIDTH];
    end else begin
        assign req_uuid_st1 = 0;
    end

    wire do_creq_rd_st1   = valid_st1 && is_creq_st1 && ~rw_st1;
    wire do_creq_wr_st1   = valid_st1 && is_creq_st1 && rw_st1;
    wire do_fill_st1      = valid_st1 && is_fill_st1;
    wire do_replay_rd_st1 = valid_st1 && is_replay_st1 && ~rw_st1;
    wire do_replay_wr_st1 = valid_st1 && is_replay_st1 && rw_st1;

    wire do_read_hit_st1  = do_creq_rd_st1 && is_hit_st1;
    wire do_read_miss_st1 = do_creq_rd_st1 && ~is_hit_st1;

    wire do_write_hit_st1 = do_creq_wr_st1 && is_hit_st1;
    wire do_write_miss_st1= do_creq_wr_st1 && ~is_hit_st1;

    `UNUSED_VAR (do_write_miss_st1)

    // ensure mshr replay always get a hit
    `RUNTIME_ASSERT (~(valid_st1 && is_replay_st1) || is_hit_st1, ("runtime error: invalid mshr replay"));

    // detect BRAM's read-during-write hazard
    assign rdw_hazard_st0 = do_fill_st0; // after a fill
    always @(posedge clk) begin
        rdw_hazard_st1 <= (do_creq_rd_st0 && do_write_hit_st1 && (addr_st0 == addr_st1))
                       && ~rdw_hazard_st1; // after a write to same address
    end

    wire [`CS_WORD_WIDTH-1:0] write_data_st1 = data_st1[`CS_WORD_WIDTH-1:0];
    wire [`CS_LINE_WIDTH-1:0] fill_data_st1 = data_st1;

    `RESET_RELAY (data_reset, reset);

    VX_cache_data #(
        .INSTANCE_ID  (INSTANCE_ID),
        .BANK_ID      (BANK_ID),
        .CACHE_SIZE   (CACHE_SIZE),
        .LINE_SIZE    (LINE_SIZE),
        .NUM_BANKS    (NUM_BANKS),
        .NUM_WAYS     (NUM_WAYS),
        .WORD_SIZE    (WORD_SIZE),
        .WRITE_ENABLE (WRITE_ENABLE),
        .UUID_WIDTH   (UUID_WIDTH)
    ) cache_data (
        .clk        (clk),
        .reset      (data_reset),

        .req_uuid   (req_uuid_st1),

        .stall      (pipe_stall),

        .read       (do_read_hit_st1 || do_replay_rd_st1),
        .fill       (do_fill_st1),
        .write      (do_write_hit_st1 || do_replay_wr_st1),
        .way_sel    (way_sel_st1 | tag_matches_st1),
        .line_addr  (addr_st1),
        .wsel       (wsel_st1),
        .byteen     (byteen_st1),
        .fill_data  (fill_data_st1),
        .write_data (write_data_st1),
        .read_data  (read_data_st1)
    );

    wire [MSHR_SIZE-1:0] mshr_matches_st0;
    wire mshr_allocate_st0 = valid_st0 && is_creq_st0 && ~pipe_stall;
    wire mshr_lookup_st0   = mshr_allocate_st0;
    wire mshr_finalize_st1 = valid_st1 && is_creq_st1 && ~pipe_stall;
    wire mshr_release_st1  = is_hit_st1 || (rw_st1 && ~mshr_pending_st1);

    VX_pending_size #(
        .SIZE (MSHR_SIZE)
    ) mshr_pending_size (
        .clk   (clk),
        .reset (reset),
        .incr  (core_req_fire),
        .decr  (replay_fire || (mshr_finalize_st1 && mshr_release_st1)),
        .full  (mshr_alm_full),
        `UNUSED_PIN (size),
        `UNUSED_PIN (empty)
    );

    `RESET_RELAY (mshr_reset, reset);

    VX_cache_mshr #(
        .INSTANCE_ID (INSTANCE_ID),
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
        .lookup_matches (mshr_matches_st0),

        // finalize
        .finalize_valid (mshr_finalize_st1),
        .finalize_release(mshr_release_st1),
        .finalize_pending(mshr_pending_st1),
        .finalize_id    (mshr_id_st1),
        .finalize_prev  (mshr_prev_st1)
    );

    // ignore allocated id from mshr matches
    wire [MSHR_SIZE-1:0] lookup_matches;
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin
        assign lookup_matches[i] = (i != mshr_alloc_id_st0) && mshr_matches_st0[i];
    end
    assign mshr_pending_st0 = (| lookup_matches);

    // schedule core response

    wire crsq_valid, crsq_ready;
    wire [`CS_WORD_WIDTH-1:0] crsq_data;
    wire [REQ_SEL_WIDTH-1:0] crsq_idx;
    wire [TAG_WIDTH-1:0] crsq_tag;

    assign crsq_valid = do_read_hit_st1 || do_replay_rd_st1;
    assign crsq_idx   = req_idx_st1;
    assign crsq_data  = read_data_st1;
    assign crsq_tag   = tag_st1;

    `RESET_RELAY (crsp_reset, reset);

    VX_elastic_buffer #(
        .DATAW   (TAG_WIDTH + `CS_WORD_WIDTH + REQ_SEL_WIDTH),
        .SIZE    (CRSQ_SIZE),
        .OUT_REG (`TO_OUT_BUF_REG(CORE_OUT_BUF))
    ) core_rsp_queue (
        .clk       (clk),
        .reset     (crsp_reset),
        .valid_in  (crsq_valid && ~rdw_hazard_st1),
        .ready_in  (crsq_ready),
        .data_in   ({crsq_tag, crsq_data, crsq_idx}),
        .data_out  ({core_rsp_tag, core_rsp_data, core_rsp_idx}),
        .valid_out (core_rsp_valid),
        .ready_out (core_rsp_ready)
    );

    assign crsq_stall = crsq_valid && ~crsq_ready;

    // schedule memory request

    wire mreq_push, mreq_pop, mreq_empty;
    wire [`CS_WORD_WIDTH-1:0] mreq_data;
    wire [WORD_SIZE-1:0] mreq_byteen;
    wire [WORD_SEL_WIDTH-1:0] mreq_wsel;
    wire [`CS_LINE_ADDR_WIDTH-1:0] mreq_addr;
    wire [MSHR_ADDR_WIDTH-1:0] mreq_id;
    wire mreq_rw;

    assign mreq_push = (do_read_miss_st1 && ~mshr_pending_st1)
                     || do_creq_wr_st1;

    assign mreq_pop = mem_req_valid && mem_req_ready;

    assign mreq_rw   = WRITE_ENABLE && rw_st1;
    assign mreq_addr = addr_st1;
    assign mreq_id   = mshr_id_st1;
    assign mreq_wsel = wsel_st1;
    assign mreq_byteen = byteen_st1;
    assign mreq_data = write_data_st1;

    `RESET_RELAY (mreq_reset, reset);

    VX_fifo_queue #(
        .DATAW    (1 + `CS_LINE_ADDR_WIDTH + MSHR_ADDR_WIDTH + WORD_SIZE + WORD_SEL_WIDTH + `CS_WORD_WIDTH),
        .DEPTH    (MREQ_SIZE),
        .ALM_FULL (MREQ_SIZE-2),
        .OUT_REG  (`TO_OUT_BUF_REG(MEM_OUT_BUF))
    ) mem_req_queue (
        .clk        (clk),
        .reset      (mreq_reset),
        .push       (mreq_push),
        .pop        (mreq_pop),
        .data_in    ({mreq_rw, mreq_addr, mreq_id, mreq_byteen, mreq_wsel, mreq_data}),
        .data_out   ({mem_req_rw, mem_req_addr, mem_req_id, mem_req_byteen, mem_req_wsel, mem_req_data}),
        .empty      (mreq_empty),
        .alm_full   (mreq_alm_full),
        `UNUSED_PIN (full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (size)
    );

    assign mem_req_valid = ~mreq_empty;

///////////////////////////////////////////////////////////////////////////////

`ifdef PERF_ENABLE
    assign perf_read_misses  = do_read_miss_st1;
    assign perf_write_misses = do_write_miss_st1;
    assign perf_mshr_stalls  = mshr_alm_full;
`endif

`ifdef DBG_TRACE_CACHE
    wire crsq_fire = crsq_valid && crsq_ready;
    wire pipeline_stall = (replay_valid || mem_rsp_valid || core_req_valid)
                       && ~(replay_fire || mem_rsp_fire || core_req_fire);
    always @(posedge clk) begin
        if (pipeline_stall) begin
            `TRACE(3, ("%d: *** %s-bank%0d stall: crsq=%b, mreq=%b, mshr=%b\n", $time, INSTANCE_ID, BANK_ID, crsq_stall, mreq_alm_full, mshr_alm_full));
        end
        if (init_enable) begin
            `TRACE(2, ("%d: %s-bank%0d init: addr=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(init_line_sel, BANK_ID)));
        end
        if (mem_rsp_fire) begin
            `TRACE(2, ("%d: %s-bank%0d fill-rsp: addr=0x%0h, mshr_id=%0d, data=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(mem_rsp_addr, BANK_ID), mem_rsp_id, mem_rsp_data));
        end
        if (replay_fire) begin
            `TRACE(2, ("%d: %s-bank%0d mshr-pop: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(replay_addr, BANK_ID), replay_tag, replay_idx, req_uuid_sel));
        end
        if (core_req_fire) begin
            if (core_req_rw)
                `TRACE(2, ("%d: %s-bank%0d core-wr-req: addr=0x%0h, tag=0x%0h, req_idx=%0d, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(core_req_addr, BANK_ID), core_req_tag, core_req_idx, core_req_byteen, core_req_data, req_uuid_sel));
            else
                `TRACE(2, ("%d: %s-bank%0d core-rd-req: addr=0x%0h, tag=0x%0h, req_idx=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(core_req_addr, BANK_ID), core_req_tag, core_req_idx, req_uuid_sel));
        end
        if (crsq_fire) begin
            `TRACE(2, ("%d: %s-bank%0d core-rd-rsp: addr=0x%0h, tag=0x%0h, req_idx=%0d, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(addr_st1, BANK_ID), crsq_tag, crsq_idx, crsq_data, req_uuid_st1));
        end
        if (mreq_push) begin
            if (do_creq_wr_st1)
                `TRACE(2, ("%d: %s-bank%0d writethrough: addr=0x%0h, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(mreq_addr, BANK_ID), mreq_byteen, mreq_data, req_uuid_st1));
            else
                `TRACE(2, ("%d: %s-bank%0d fill-req: addr=0x%0h, mshr_id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(mreq_addr, BANK_ID), mreq_id, req_uuid_st1));
        end
    end
`endif

endmodule
