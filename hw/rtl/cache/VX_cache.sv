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

module VX_cache import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID   = "",

    parameter NUM_REQS              = 4,      // Number of Word requests per cycle

    parameter MEM_PORTS             = 1,      // Number of memory ports

    parameter CACHE_SIZE            = 32768,  // Size of cache in bytes
    parameter LINE_SIZE             = 64,     // Size of line inside a bank in bytes
    parameter NUM_BANKS             = 4,      // Number of banks
    parameter NUM_WAYS              = 4,      // Number of associative ways
    parameter WORD_SIZE             = 16,     // Size of a word in bytes
    parameter SECTOR_SIZE           = LINE_SIZE, // Size of a sector in bytes (fill/eviction/mem-request granule); = LINE_SIZE => 1 sector

    parameter CRSQ_SIZE             = 0,      // Core Response Queue Size
    parameter MSHR_SIZE             = 16,     // Miss Reserv Queue Knob
    parameter MSHR_LINES            = 0,      // MSHR distinct line entries (0 = address per request)
    parameter MRSQ_SIZE             = 4,      // Memory Response Queue Size
    parameter MREQ_SIZE             = 0,      // Memory Request Queue Size (0 = derived minimum)

    parameter LATENCY               = 2,      // Bank pipeline depth (2 = classic lookup+commit; larger defers the data array)

    parameter WRITE_ENABLE          = 1,      // Enable cache writeable

    parameter WRITEBACK             = 0,      // Enable cache writeback

    parameter DIRTY_BYTES           = 0,      // Enable dirty bytes on writeback

    parameter LAZY_FETCH            = 0,      // Write misses allocate without a fill (write-validate)

    parameter WRITE_AROUND          = 0,      // Write misses flagged no-allocate go to memory (write-back only)

    parameter REPL_POLICY           = `CS_REPL_FIFO, // Replacement policy

    parameter TAG_WIDTH             = UUID_WIDTH + 1, // core request tag size

    parameter CORE_OUT_BUF          = 3,      // core response output buffer (bank-level buffering derived internally)
    parameter MEM_OUT_BUF           = 3,

    parameter IS_LLC                = 0,      // last-level cache: banks own AMO commit + RVA reservation tracking
    parameter SHARED_LLC            = 0,      // the LLC serves every memory client alone (point of coherence)

    parameter FLUSH_FORWARD         = 1,      // a flush also flushes the next level

    parameter BANK_HASH             = 0,      // hashed bank select (cache_bank_hash)

    parameter READ_MERGE            = 0,      // merge same-word reads within groups of this many adjacent ports (0: off)

    parameter AMO_ENABLE            = 0       // synthesize atomic-op logic in the banks
 ) (
`ifdef PERF_ENABLE
    output cache_perf_t     cache_perf,
`endif
    input wire clk,
    input wire reset,
    VX_mem_bus_if.slave     core_bus_if [NUM_REQS],
    VX_mem_bus_if.master    mem_bus_if [MEM_PORTS],
    // nothing held between this level's inputs and its memory side, on either path
    input wire              nc_req_empty
);
    `STATIC_ASSERT(NUM_BANKS == (1 << `CLOG2(NUM_BANKS)), ("invalid parameter: number of banks must be power of 2"))
    `STATIC_ASSERT(WRITE_ENABLE || !WRITEBACK, ("invalid parameter: writeback requires write enable"))
    `STATIC_ASSERT(WRITEBACK || !DIRTY_BYTES, ("invalid parameter: dirty bytes require writeback"))
    `STATIC_ASSERT(!LAZY_FETCH || (WRITEBACK && DIRTY_BYTES && IS_LLC), ("invalid parameter: lazy fetch requires a write-back LLC with dirty bytes"))
    `STATIC_ASSERT(!WRITE_AROUND || WRITEBACK, ("invalid parameter: write-around requires a write-back cache"))
    `STATIC_ASSERT(NUM_BANKS >= MEM_PORTS, ("invalid parameter: number of banks must be greater or equal to number of memory ports"))
    `STATIC_ASSERT(!BANK_HASH || NUM_BANKS <= 256, ("invalid parameter: bank hash supports up to 256 banks"))
    localparam REQ_SEL_WIDTH   = `UP(`CS_REQ_SEL_BITS);
    localparam WORD_SEL_WIDTH  = `UP(`CS_WORD_SEL_BITS);
    localparam MSHR_ADDR_WIDTH = `LOG2UP(MSHR_SIZE);
    localparam MEM_TAG_WIDTH   = `CACHE_MEM_TAG_WIDTH(MSHR_SIZE, NUM_BANKS, MEM_PORTS, UUID_WIDTH);
    localparam WORDS_PER_LINE  = LINE_SIZE / WORD_SIZE;
    localparam WORD_WIDTH      = WORD_SIZE * 8;
    localparam WORD_SEL_BITS   = `CLOG2(WORDS_PER_LINE);
    localparam BANK_SEL_BITS   = `CLOG2(NUM_BANKS);
    localparam BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS);
    localparam LINE_ADDR_WIDTH = (`CS_WORD_ADDR_WIDTH - BANK_SEL_BITS - WORD_SEL_BITS);
    // Read merging carries the merged ports' mask (one bit per port of the
    // leader's group) below the request tag through the bank.
    localparam MERGE_GROUP     = (READ_MERGE != 0) ? `MIN(READ_MERGE, NUM_REQS) : 1;
    localparam DO_READ_MERGE   = (MERGE_GROUP > 1);
    localparam BANK_TAG_WIDTH  = TAG_WIDTH + (DO_READ_MERGE ? MERGE_GROUP : 0);
    localparam CORE_REQ_DATAW  = LINE_ADDR_WIDTH + 1 + WORD_SEL_WIDTH + WORD_SIZE + WORD_WIDTH + BANK_TAG_WIDTH + `UP(MEM_ATTR_WIDTH);
    localparam CORE_RSP_DATAW  = WORD_WIDTH + TAG_WIDTH;
    localparam BANK_MEM_TAG_WIDTH = UUID_WIDTH + MSHR_ADDR_WIDTH;
    localparam MEM_REQ_DATAW   = (`UP(BANK_SEL_BITS) + `CS_LINE_SECTOR_ADDR_WIDTH + 1 + SECTOR_SIZE + `CS_SECTOR_WIDTH + BANK_MEM_TAG_WIDTH + `UP(MEM_ATTR_WIDTH));
    localparam MEM_RSP_DATAW   = `CS_SECTOR_WIDTH + MEM_TAG_WIDTH;
    localparam MEM_PORTS_SEL_BITS = `CLOG2(MEM_PORTS);
    localparam MEM_PORTS_SEL_WIDTH = `UP(MEM_PORTS_SEL_BITS);
    localparam MEM_ARB_SEL_BITS = `CLOG2(`CDIV(NUM_BANKS, MEM_PORTS));
    localparam REQ_XBAR_BUF    = (NUM_REQS > 2) ? 2 : 0;
    // With more than two ports, a request stage sits between the ports and the
    // bank crossbar: the flush hold and read merging act on the port heads and
    // the bank arbitration on the stage, two short handshake loops instead of
    // one through both.
    localparam CORE_REQ_STAGE  = (NUM_REQS > 2);
    // The response side mirrors it: with read merging, a response is released
    // only once every port it targets has taken it, so a stage with its own
    // registered ready sits between the bank response queues and the merged
    // response arbiters, keeping that release off the queues' registers.
    localparam CORE_RSP_STAGE  = DO_READ_MERGE && (NUM_REQS > 2);
    localparam CORE_RSP_BUF_ENABLE = (NUM_BANKS != 1) || (NUM_REQS != 1);
    localparam MEM_REQ_BUF_ENABLE = (NUM_BANKS != 1);
    `STATIC_ASSERT(!DO_READ_MERGE || (`IS_POW2(MERGE_GROUP) && `IS_DIVISBLE(NUM_REQS, MERGE_GROUP)), ("invalid parameter: read merge group"))
`ifdef PERF_ENABLE
    wire [NUM_BANKS-1:0] perf_read_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_write_miss_per_bank;
    wire [NUM_BANKS-1:0] perf_evictions_per_bank;
    wire [NUM_BANKS-1:0] perf_mshr_stall_per_bank;
`endif
    // the inputs' flush tokens and gating (VX_cache_req_port, VX_cache_init)
    wire [NUM_REQS-1:0]                flush_req_mask;
    wire [NUM_REQS-1:0]                flush_full_mask;
    wire [NUM_REQS-1:0]                flush_l1_mask;
    wire [NUM_REQS-1:0]                flush_io_mask;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0] flush_req_tag;
    wire [NUM_REQS-1:0]                flush_rsp_pending;
    wire                               flush_hold;
    wire [NUM_REQS-1:0]                flush_dbg_stall;
    wire                               flush_pending_load;
    wire                               flush_pending_owed;
    wire [NUM_REQS-1:0]                flush_rsp_take;
    wire [NUM_BANKS-1:0] per_bank_flush_begin;
    wire                 flush_skip;
    wire [`UP(UUID_WIDTH)-1:0] flush_uuid;
    wire [NUM_BANKS-1:0] per_bank_flush_end;
    wire                 core_req_stage_empty;
    wire                     mem_req_empty;
    wire [MEM_PORTS-1:0]     mem_flush_valid;
    wire                     mem_flush_l1_only;
    wire                     mem_flush_io;
    wire [MEM_PORTS-1:0]     mem_flush_ready;
    wire                     mem_flush_rsp_phase;
    wire [MEM_PORTS-1:0]     mem_flush_rsp_valid;
    wire                     flush_rsp_valid,   flush_rsp_valid_w;
    wire [REQ_SEL_WIDTH-1:0] flush_rsp_idx,     flush_rsp_idx_w;
    wire [NUM_REQS-1:0]      flush_rsp_onehot,  flush_rsp_onehot_w;
    wire [TAG_WIDTH-1:0]     flush_rsp_tag,     flush_rsp_tag_w;
    wire                     flush_rsp_ready,   flush_rsp_ready_w;
    VX_cache_init #(
        .NUM_REQS         (NUM_REQS),
        .NUM_BANKS        (NUM_BANKS),
        .MEM_PORTS        (MEM_PORTS),
        .FLUSH_FORWARD    (FLUSH_FORWARD),
        .SHARED_LLC       (SHARED_LLC),
        .TAG_WIDTH        (TAG_WIDTH),
        .REQ_STAGE        (CORE_REQ_STAGE),
        .REQ_SEL_WIDTH    (REQ_SEL_WIDTH)
    ) cache_init (
        .clk                 (clk),
        .reset               (reset),
        .flush_req_mask      (flush_req_mask),
        .flush_full_mask     (flush_full_mask),
        .flush_l1_mask       (flush_l1_mask),
        .flush_io_mask       (flush_io_mask),
        .req_tag             (flush_req_tag),
        .rsp_pending         (flush_rsp_pending),
        .flush_hold          (flush_hold),
        .dbg_stall           (flush_dbg_stall),
        .pending_load        (flush_pending_load),
        .pending_owed        (flush_pending_owed),
        .rsp_take            (flush_rsp_take),
        .req_stage_empty     (core_req_stage_empty),
        .flush_begin         (per_bank_flush_begin),
        .flush_skip          (flush_skip),
        .flush_uuid          (flush_uuid),
        .flush_end           (per_bank_flush_end),
        .mem_req_empty       (mem_req_empty),
        .nc_req_empty        (nc_req_empty),
        .mem_flush_valid     (mem_flush_valid),
        .mem_flush_l1_only   (mem_flush_l1_only),
        .mem_flush_io        (mem_flush_io),
        .mem_flush_ready     (mem_flush_ready),
        .mem_flush_rsp_phase (mem_flush_rsp_phase),
        .mem_flush_rsp_valid (mem_flush_rsp_valid),
        .flush_rsp_valid     (flush_rsp_valid_w),
        .flush_rsp_idx       (flush_rsp_idx_w),
        .flush_rsp_onehot    (flush_rsp_onehot_w),
        .flush_rsp_tag       (flush_rsp_tag_w),
        .flush_rsp_ready     (flush_rsp_ready_w)
    );
    VX_mem_bus_if #(
        .DATA_SIZE (SECTOR_SIZE),
        .TAG_WIDTH (MEM_TAG_WIDTH)
    ) mem_bus_tmp_if[MEM_PORTS]();
    wire [MEM_PORTS-1:0]                    mem_rsp_queue_valid;
    wire [MEM_PORTS-1:0][MEM_RSP_DATAW-1:0] mem_rsp_queue_data;
    wire [MEM_PORTS-1:0]                    mem_rsp_queue_ready;
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue
        VX_elastic_buffer #(
            .DATAW   (MEM_RSP_DATAW),
            .SIZE    (MRSQ_SIZE),
            .OUT_REG (MRSQ_SIZE > 2)
        ) mem_rsp_queue (
            .clk        (clk),
            .reset      (reset),
            .valid_in   (mem_bus_tmp_if[i].rsp_valid),
            .data_in    (mem_bus_tmp_if[i].rsp_data),
            .ready_in   (mem_bus_tmp_if[i].rsp_ready),
            .valid_out  (mem_rsp_queue_valid[i]),
            .data_out   (mem_rsp_queue_data[i]),
            .ready_out  (mem_rsp_queue_ready[i])
        );
    end
    wire [MEM_PORTS-1:0][MEM_RSP_DATAW-MEM_ARB_SEL_BITS-1:0] mem_rsp_queue_data_s;
    wire [MEM_PORTS-1:0][BANK_SEL_WIDTH-1:0] mem_rsp_queue_sel;
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue_data_s
        wire [BANK_MEM_TAG_WIDTH-1:0] mem_rsp_tag_s = mem_rsp_queue_data[i][MEM_TAG_WIDTH-1:MEM_ARB_SEL_BITS];
        wire [`CS_SECTOR_WIDTH-1:0] mem_rsp_data_s = mem_rsp_queue_data[i][MEM_RSP_DATAW-1:MEM_TAG_WIDTH];
        assign mem_rsp_queue_data_s[i] = {mem_rsp_data_s, mem_rsp_tag_s};
    end
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_rsp_queue_sel
        if (NUM_BANKS > 1) begin : g_multibanks
            if (NUM_BANKS != MEM_PORTS) begin : g_arb_sel
                VX_bits_concat #(
                    .L (MEM_ARB_SEL_BITS),
                    .R (MEM_PORTS_SEL_BITS)
                ) mem_rsp_sel_concat (
                    .left_in  (mem_rsp_queue_data[i][MEM_ARB_SEL_BITS-1:0]),
                    .right_in (MEM_PORTS_SEL_WIDTH'(i)),
                    .data_out (mem_rsp_queue_sel[i])
                );
            end else begin : g_no_arb_sel
                assign mem_rsp_queue_sel[i] = MEM_PORTS_SEL_WIDTH'(i);
            end
        end else begin : g_singlebank
            assign mem_rsp_queue_sel[i] = 0;
        end
    end
    // While the flush tokens are out the cache is quiescent (inputs blocked,
    // MSHRs empty), so every memory response is a token's.
    wire [MEM_PORTS-1:0] mem_rsp_xbar_valid;
    wire [MEM_PORTS-1:0] mem_rsp_xbar_ready;
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_flush_rsp
        assign mem_flush_rsp_valid[i]  = mem_flush_rsp_phase && mem_rsp_queue_valid[i];
        assign mem_rsp_xbar_valid[i]   = mem_rsp_queue_valid[i] && ~mem_flush_rsp_phase;
        assign mem_rsp_queue_ready[i]  = mem_flush_rsp_phase || mem_rsp_xbar_ready[i];
    end
    wire [NUM_BANKS-1:0] per_bank_mem_rsp_valid;
    wire [NUM_BANKS-1:0][MEM_RSP_DATAW-MEM_ARB_SEL_BITS-1:0] per_bank_mem_rsp_pdata;
    wire [NUM_BANKS-1:0] per_bank_mem_rsp_ready;
    VX_stream_omega #(
        .NUM_INPUTS  (MEM_PORTS),
        .NUM_OUTPUTS (NUM_BANKS),
        .DATAW       (MEM_RSP_DATAW-MEM_ARB_SEL_BITS),
        .ARBITER     ("R"),
        // Registered at the input handshake, so the wide fill payload does not
        // load on the bank's late ready.
        .OUT_BUF     (2)
    ) mem_rsp_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_rsp_xbar_valid),
        .data_in   (mem_rsp_queue_data_s),
        .sel_in    (mem_rsp_queue_sel),
        .ready_in  (mem_rsp_xbar_ready),
        .valid_out (per_bank_mem_rsp_valid),
        .data_out  (per_bank_mem_rsp_pdata),
        `UNUSED_PIN (sel_out),
        .ready_out (per_bank_mem_rsp_ready),
        `UNUSED_PIN (collisions),
        `UNUSED_PIN (empty)
    );
    wire [NUM_BANKS-1:0][`CS_SECTOR_WIDTH-1:0] per_bank_mem_rsp_data;
    wire [NUM_BANKS-1:0][BANK_MEM_TAG_WIDTH-1:0] per_bank_mem_rsp_tag;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_mem_rsp_data
        assign {
            per_bank_mem_rsp_data[i],
            per_bank_mem_rsp_tag[i]
        } = per_bank_mem_rsp_pdata[i];
    end
    wire [NUM_BANKS-1:0]                        per_bank_core_req_valid;
    wire [NUM_BANKS-1:0][`CS_LINE_ADDR_WIDTH-1:0] per_bank_core_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_rw;
    wire [NUM_BANKS-1:0][WORD_SEL_WIDTH-1:0]    per_bank_core_req_wsel;
    wire [NUM_BANKS-1:0][WORD_SIZE-1:0]         per_bank_core_req_byteen;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_req_data;
    wire [NUM_BANKS-1:0][BANK_TAG_WIDTH-1:0]    per_bank_core_req_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_req_idx;
    wire [NUM_BANKS-1:0][`UP(MEM_ATTR_WIDTH)-1:0]  per_bank_core_req_attr;
    wire [NUM_BANKS-1:0]                        per_bank_core_req_ready;
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    per_bank_core_rsp_data;
    wire [NUM_BANKS-1:0][BANK_TAG_WIDTH-1:0]    per_bank_core_rsp_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     per_bank_core_rsp_idx;
    wire [NUM_BANKS-1:0][NUM_REQS-1:0]          per_bank_core_rsp_targets;
    wire [NUM_BANKS-1:0]                        per_bank_core_rsp_ready;
    // the banks' own response outputs, ahead of the response stage
    wire [NUM_BANKS-1:0]                        bank_core_rsp_valid;
    wire [NUM_BANKS-1:0][`CS_WORD_WIDTH-1:0]    bank_core_rsp_data;
    wire [NUM_BANKS-1:0][BANK_TAG_WIDTH-1:0]    bank_core_rsp_tag;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]     bank_core_rsp_idx;
    wire [NUM_BANKS-1:0][NUM_REQS-1:0]          bank_core_rsp_targets;
    wire [NUM_BANKS-1:0]                        bank_core_rsp_ready;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_valid;
    wire [NUM_BANKS-1:0][`CS_LINE_SECTOR_ADDR_WIDTH-1:0] per_bank_mem_req_addr;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_rw;
    wire [NUM_BANKS-1:0][SECTOR_SIZE-1:0]       per_bank_mem_req_byteen;
    wire [NUM_BANKS-1:0][`CS_SECTOR_WIDTH-1:0]  per_bank_mem_req_data;
    wire [NUM_BANKS-1:0][BANK_MEM_TAG_WIDTH-1:0] per_bank_mem_req_tag;
    wire [NUM_BANKS-1:0][`UP(MEM_ATTR_WIDTH)-1:0]  per_bank_mem_req_attr;
    wire [NUM_BANKS-1:0]                        per_bank_mem_req_ready;
    // Each input up to the bank arbiters is one VX_cache_req_port: its flush
    // token's kind and answer state, its admission, its bank select, the
    // request stage and its one-hot request to the banks. What combines the
    // inputs stays here: read merging and the bank arbiters.
    wire [NUM_REQS-1:0]                      core_req_valid;
    wire [NUM_REQS-1:0]                      core_req_rw;
    wire [NUM_REQS-1:0][`CS_WORD_ADDR_WIDTH-1:0] core_req_addr;
    wire [NUM_REQS-1:0][TAG_WIDTH-1:0]       core_req_tag;
    wire [NUM_REQS-1:0][`UP(MEM_ATTR_WIDTH)-1:0] core_req_attr;
    wire [NUM_REQS-1:0]                      core_req_stage_ready;
    wire [NUM_REQS-1:0]                      core_req_follower;
    wire [NUM_REQS-1:0]                      core_req_merge_ready;
    wire [NUM_REQS-1:0][`UP(MERGE_GROUP > 1 ? MERGE_GROUP : 0)-1:0] core_req_merge_mask;
    wire [NUM_REQS-1:0]                      core_req_stage_valid;
    wire [NUM_REQS-1:0][NUM_BANKS-1:0]       core_req_bank_valid;
    wire [NUM_REQS-1:0][CORE_REQ_DATAW-1:0]  core_req_bank_data;
    wire [NUM_REQS-1:0][NUM_BANKS-1:0]       core_req_bank_ready;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req
        assign core_req_rw[i]   = core_bus_if[i].req_data.rw;
        assign core_req_addr[i] = core_bus_if[i].req_data.addr;
        assign core_req_tag[i]  = core_bus_if[i].req_data.tag;
        assign core_req_attr[i] = `UP(MEM_ATTR_WIDTH)'(core_bus_if[i].req_data.attr);
        assign flush_req_tag[i] = core_req_tag[i];
    end
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_req_ports
        VX_cache_req_port #(
            .NUM_BANKS     (NUM_BANKS),
            .WORD_SIZE     (WORD_SIZE),
            .WORD_SEL_BITS (WORD_SEL_BITS),
            .ADDR_WIDTH    (`CS_WORD_ADDR_WIDTH),
            .TAG_WIDTH     (TAG_WIDTH),
            .MERGE_GROUP   (MERGE_GROUP),
            .BANK_HASH     (BANK_HASH),
            .SHARED_LLC    (SHARED_LLC),
            .REQ_STAGE     (CORE_REQ_STAGE),
            .SEL_READY     (NUM_REQS == 1)
        ) req_port (
            .clk            (clk),
            .reset          (reset),
            .in_valid       (core_bus_if[i].req_valid),
            .in_rw          (core_req_rw[i]),
            .in_addr        (core_req_addr[i]),
            .in_data        (core_bus_if[i].req_data.data),
            .in_byteen      (core_bus_if[i].req_data.byteen),
            .in_attr        (core_req_attr[i]),
            .in_tag         (core_req_tag[i]),
            .in_ready       (core_bus_if[i].req_ready),
            .flush          (flush_req_mask[i]),
            .flush_full     (flush_full_mask[i]),
            .flush_l1       (flush_l1_mask[i]),
            .flush_io       (flush_io_mask[i]),
            .flush_hold     (flush_hold),
            .dbg_stall      (flush_dbg_stall[i]),
            .pending_load   (flush_pending_load),
            .pending_owed   (flush_pending_owed),
            .rsp_take       (flush_rsp_take[i]),
            .pending        (flush_rsp_pending[i]),
            .req_valid      (core_req_valid[i]),
            .req_ready      (core_req_stage_ready[i]),
            .merge_follower (core_req_follower[i]),
            .merge_ready    (core_req_merge_ready[i]),
            .merge_mask     (core_req_merge_mask[i]),
            .stage_valid    (core_req_stage_valid[i]),
            .bank_valid     (core_req_bank_valid[i]),
            .bank_data      (core_req_bank_data[i]),
            .bank_ready     (core_req_bank_ready[i])
        );
    end
    // Read merging: plain reads of the same word with the same tag from ports
    // of one group in the same cycle are served by one bank access. The lowest
    // such port leads; the others stay out of the request crossbar and are
    // accepted in the cycle their leader is, so every port keeps its own
    // request order into the bank.
    if (DO_READ_MERGE) begin : g_read_merge
        // a plain read has no flush, I/O, local, OM or atomic attribute
        localparam [`UP(MEM_ATTR_WIDTH)-1:0] SPECIAL_ATTRS = `UP(MEM_ATTR_WIDTH)'((1 << MEM_ATTR_FLUSH_OFFS)
                                                                                | (1 << MEM_ATTR_IO_OFFS)
                                                                                | (1 << MEM_ATTR_LOCAL_OFFS)
                                                                                | (1 << MEM_ATTR_OM_OFFS)
                                                                                | (1 << MEM_ATTR_AMO_OFFS));
        wire [NUM_REQS-1:0] plain_read;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_plain_read
            assign plain_read[i] = core_req_valid[i] && ~core_req_rw[i] && ((core_req_attr[i] & SPECIAL_ATTRS) == '0);
        end
        // same_word[i][k]: port i reads what a lower port k of its group (port
        // GROUP_BASE + k) reads; the ports of other groups never merge, so
        // only a group's ports are compared (NUM_REQS x MERGE_GROUP, not
        // NUM_REQS x NUM_REQS, blocks).
        wire [NUM_REQS-1:0][MERGE_GROUP-1:0] same_word;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_same_i
            localparam GROUP_BASE = (i / MERGE_GROUP) * MERGE_GROUP;
            for (genvar k = 0; k < MERGE_GROUP; ++k) begin : g_same_k
                localparam j = GROUP_BASE + k;
                if (j < i) begin : g_cmp
                    assign same_word[i][k] = plain_read[i] && plain_read[j]
                                          && (core_req_addr[i] == core_req_addr[j])
                                          && (core_req_tag[i] == core_req_tag[j]);
                end else begin : g_none
                    assign same_word[i][k] = 1'b0;
                end
            end
        end
        // a port's own access is accepted when its request stage takes it
        wire [NUM_REQS-1:0] leader_ready = ~core_req_follower & core_req_stage_ready;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_merge
            localparam GROUP_BASE = (i / MERGE_GROUP) * MERGE_GROUP;
            wire [MERGE_GROUP-1:0] merged_mask;
            wire [MERGE_GROUP-1:0] leader_fire;
            for (genvar k = 0; k < MERGE_GROUP; ++k) begin : g_merged_mask
                assign merged_mask[k] = same_word[GROUP_BASE + k][i - GROUP_BASE];
            end
            for (genvar k = 0; k < MERGE_GROUP; ++k) begin : g_leader_fire
                assign leader_fire[k] = same_word[i][k] && leader_ready[GROUP_BASE + k];
            end
            assign core_req_follower[i]    = (| same_word[i]);
            assign core_req_merge_mask[i]  = merged_mask;
            assign core_req_merge_ready[i] = (| leader_fire);
        end
    end else begin : g_no_read_merge
        `UNUSED_VAR (core_req_valid)
        `UNUSED_VAR (core_req_rw)
        `UNUSED_VAR (core_req_stage_ready)
        assign core_req_follower    = '0;
        assign core_req_merge_ready = '0;
        assign core_req_merge_mask  = '0;
    end
    // a stream buffer holds an entry exactly when its output is valid
    assign core_req_stage_empty = ~(| core_req_stage_valid);
    // The bank arbiters (a crossbar whose inputs, the ports, demultiplex their
    // own requests). Over more than 12 ports the arbiters are a tree anyway;
    // slices of four instead of eight keep its depth and shorten the
    // round-robin each stage entry's pop waits on.
    localparam REQ_ARB_FANOUT = (NUM_REQS > 12) ? 4 : `MAX_FANOUT;
    wire [NUM_BANKS-1:0][CORE_REQ_DATAW-1:0] core_req_data_out;
    for (genvar b = 0; b < NUM_BANKS; ++b) begin : g_core_req_xbar
        wire [NUM_REQS-1:0] arb_valid_in, arb_ready_in;
        for (genvar i = 0; i < NUM_REQS; ++i) begin : g_in
            assign arb_valid_in[i] = core_req_bank_valid[i][b];
            assign core_req_bank_ready[i][b] = arb_ready_in[i];
        end
        VX_stream_arb #(
            .NUM_INPUTS  (NUM_REQS),
            .NUM_OUTPUTS (1),
            .DATAW       (CORE_REQ_DATAW),
            .ARBITER     ("R"),
            .MAX_FANOUT  (REQ_ARB_FANOUT),
            .OUT_BUF     (REQ_XBAR_BUF)
        ) xbar_arb (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (arb_valid_in),
            .data_in   (core_req_bank_data),
            .ready_in  (arb_ready_in),
            .valid_out (per_bank_core_req_valid[b]),
            .data_out  (core_req_data_out[b]),
            .sel_out   (per_bank_core_req_idx[b]),
            .ready_out (per_bank_core_req_ready[b])
        );
    end
`ifdef PERF_ENABLE
    // bank conflicts: a port and a higher one requesting the same bank in a
    // cycle one of them is accepted
    wire [PERF_CTR_BITS-1:0] perf_collisions;
    reg [NUM_REQS-1:0] perf_collision_per_req, perf_collision_per_req_r;
    always @(*) begin
        perf_collision_per_req = '0;
        for (integer i = 0; i < NUM_REQS; ++i) begin
            for (integer j = i + 1; j < NUM_REQS; ++j) begin
                perf_collision_per_req[i] |= (| (core_req_bank_valid[i] & core_req_bank_valid[j]))
                                          && ((| core_req_bank_ready[i]) | (| core_req_bank_ready[j]));
            end
        end
    end
    `BUFFER(perf_collision_per_req_r, perf_collision_per_req);
    wire [`CLOG2(NUM_REQS+1)-1:0] perf_collision_count;
    `POP_COUNT(perf_collision_count, perf_collision_per_req_r);
    reg [PERF_CTR_BITS-1:0] perf_collisions_r;
    always @(posedge clk) begin
        if (reset) begin
            perf_collisions_r <= '0;
        end else begin
            perf_collisions_r <= perf_collisions_r + PERF_CTR_BITS'(perf_collision_count);
        end
    end
    assign perf_collisions = perf_collisions_r;
`endif
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_req_data_out
        assign {
            per_bank_core_req_addr[i],
            per_bank_core_req_rw[i],
            per_bank_core_req_wsel[i],
            per_bank_core_req_byteen[i],
            per_bank_core_req_data[i],
            per_bank_core_req_tag[i],
            per_bank_core_req_attr[i]
        } = core_req_data_out[i];
    end
    for (genvar bank_id = 0; bank_id < NUM_BANKS; ++bank_id) begin : g_banks
        VX_cache_bank #(
            .BANK_ID      (bank_id),
            .BANK_HASH    (BANK_HASH),
            .INSTANCE_ID  (`SFORMATF(("%s-bank%0d", INSTANCE_ID, bank_id))),
            .CACHE_SIZE   (CACHE_SIZE),
            .LINE_SIZE    (LINE_SIZE),
            .NUM_BANKS    (NUM_BANKS),
            .NUM_WAYS     (NUM_WAYS),
            .WORD_SIZE    (WORD_SIZE),
            .SECTOR_SIZE  (SECTOR_SIZE),
            .NUM_REQS     (NUM_REQS),
            .WRITE_ENABLE (WRITE_ENABLE),
            .WRITEBACK    (WRITEBACK),
            .DIRTY_BYTES  (DIRTY_BYTES),
            .LAZY_FETCH   (LAZY_FETCH),
            .WRITE_AROUND (WRITE_AROUND),
            .REPL_POLICY  (REPL_POLICY),
            .CRSQ_SIZE    (CRSQ_SIZE),
            .MSHR_SIZE    (MSHR_SIZE),
            .MSHR_LINES   (MSHR_LINES),
            .MRSQ_SIZE    (MRSQ_SIZE),
            .MREQ_SIZE    (MREQ_SIZE),
            .LATENCY      (LATENCY),
            .MERGE_GROUP  (DO_READ_MERGE ? MERGE_GROUP : 0),
            .TAG_WIDTH    (BANK_TAG_WIDTH),
            .CORE_OUT_BUF (CORE_RSP_BUF_ENABLE ? 3 : 0),
            .MEM_OUT_BUF  (MEM_REQ_BUF_ENABLE ? 3 : 0),
            .IS_LLC       (IS_LLC),
            .AMO_ENABLE   (AMO_ENABLE)
        ) bank (
            .clk                (clk),
            .reset              (reset),
        `ifdef PERF_ENABLE
            .perf_read_miss     (perf_read_miss_per_bank[bank_id]),
            .perf_write_miss    (perf_write_miss_per_bank[bank_id]),
            .perf_evictions     (perf_evictions_per_bank[bank_id]),
            .perf_mshr_stall    (perf_mshr_stall_per_bank[bank_id]),
        `endif
            .core_req_valid     (per_bank_core_req_valid[bank_id]),
            .core_req_addr      (per_bank_core_req_addr[bank_id]),
            .core_req_rw        (per_bank_core_req_rw[bank_id]),
            .core_req_wsel      (per_bank_core_req_wsel[bank_id]),
            .core_req_byteen    (per_bank_core_req_byteen[bank_id]),
            .core_req_data      (per_bank_core_req_data[bank_id]),
            .core_req_tag       (per_bank_core_req_tag[bank_id]),
            .core_req_idx       (per_bank_core_req_idx[bank_id]),
            .core_req_attr      (per_bank_core_req_attr[bank_id]),
            .core_req_ready     (per_bank_core_req_ready[bank_id]),
            .core_rsp_valid     (bank_core_rsp_valid[bank_id]),
            .core_rsp_data      (bank_core_rsp_data[bank_id]),
            .core_rsp_tag       (bank_core_rsp_tag[bank_id]),
            .core_rsp_idx       (bank_core_rsp_idx[bank_id]),
            .core_rsp_targets   (bank_core_rsp_targets[bank_id]),
            .core_rsp_ready     (bank_core_rsp_ready[bank_id]),
            .mem_req_valid      (per_bank_mem_req_valid[bank_id]),
            .mem_req_addr       (per_bank_mem_req_addr[bank_id]),
            .mem_req_rw         (per_bank_mem_req_rw[bank_id]),
            .mem_req_byteen     (per_bank_mem_req_byteen[bank_id]),
            .mem_req_data       (per_bank_mem_req_data[bank_id]),
            .mem_req_tag        (per_bank_mem_req_tag[bank_id]),
            .mem_req_attr      (per_bank_mem_req_attr[bank_id]),
            .mem_req_ready      (per_bank_mem_req_ready[bank_id]),
            .mem_rsp_valid      (per_bank_mem_rsp_valid[bank_id]),
            .mem_rsp_data       (per_bank_mem_rsp_data[bank_id]),
            .mem_rsp_tag        (per_bank_mem_rsp_tag[bank_id]),
            .mem_rsp_ready      (per_bank_mem_rsp_ready[bank_id]),
            .flush_begin        (per_bank_flush_begin[bank_id]),
            .flush_skip         (flush_skip),
            .flush_uuid         (flush_uuid),
            .flush_end          (per_bank_flush_end[bank_id])
        );
    end
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_rsp_stage
        if (CORE_RSP_STAGE) begin : g_stage
            VX_elastic_buffer #(
                .DATAW   (`CS_WORD_WIDTH + BANK_TAG_WIDTH + REQ_SEL_WIDTH + NUM_REQS),
                .SIZE    (2),
                .OUT_REG (0)
            ) rsp_stage (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (bank_core_rsp_valid[i]),
                .ready_in  (bank_core_rsp_ready[i]),
                .data_in   ({bank_core_rsp_data[i], bank_core_rsp_tag[i], bank_core_rsp_idx[i], bank_core_rsp_targets[i]}),
                .data_out  ({per_bank_core_rsp_data[i], per_bank_core_rsp_tag[i], per_bank_core_rsp_idx[i], per_bank_core_rsp_targets[i]}),
                .valid_out (per_bank_core_rsp_valid[i]),
                .ready_out (per_bank_core_rsp_ready[i])
            );
        end else begin : g_direct
            assign per_bank_core_rsp_valid[i]   = bank_core_rsp_valid[i];
            assign per_bank_core_rsp_data[i]    = bank_core_rsp_data[i];
            assign per_bank_core_rsp_tag[i]     = bank_core_rsp_tag[i];
            assign per_bank_core_rsp_idx[i]     = bank_core_rsp_idx[i];
            assign per_bank_core_rsp_targets[i] = bank_core_rsp_targets[i];
            assign bank_core_rsp_ready[i]       = per_bank_core_rsp_ready[i];
        end
    end

    wire [NUM_REQS-1:0]                  core_rsp_valid;
    wire [NUM_REQS-1:0][CORE_RSP_DATAW-1:0] core_rsp_pdata;
    wire [NUM_REQS-1:0]                  core_rsp_ready;
    wire [NUM_BANKS-1:0][CORE_RSP_DATAW-1:0] per_bank_core_rsp_pdata;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_core_rsp_pdata
        assign per_bank_core_rsp_pdata[i] = {per_bank_core_rsp_data[i], per_bank_core_rsp_tag[i][BANK_TAG_WIDTH-1 -: TAG_WIDTH]};
    end
    // A flush token's answer is buffered on its way to the response crossbar:
    // the token is consumed against the buffer's registered ready, not against
    // the crossbar's, and the answer enters the crossbar from registers. The
    // answer of a flush, a rare event, takes a cycle longer.
    VX_elastic_buffer #(
        .DATAW   (NUM_REQS + REQ_SEL_WIDTH + TAG_WIDTH),
        .SIZE    (2),
        .OUT_REG (1)
    ) flush_rsp_buf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (flush_rsp_valid_w),
        .ready_in  (flush_rsp_ready_w),
        .data_in   ({flush_rsp_onehot_w, flush_rsp_idx_w, flush_rsp_tag_w}),
        .data_out  ({flush_rsp_onehot,   flush_rsp_idx,   flush_rsp_tag}),
        .valid_out (flush_rsp_valid),
        .ready_out (flush_rsp_ready)
    );

    // A flush token's answer takes bank 0's lane into the response crossbar
    // when the bank has no response of its own, so it leaves through the same
    // registered outputs as every other response.
    wire [NUM_BANKS-1:0]                     core_rsp_xbar_valid;
    wire [NUM_BANKS-1:0][CORE_RSP_DATAW-1:0] core_rsp_xbar_pdata;
    wire [NUM_BANKS-1:0][REQ_SEL_WIDTH-1:0]  core_rsp_xbar_idx;
    wire [NUM_BANKS-1:0]                     core_rsp_xbar_ready;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_core_rsp_xbar_in
        if (i == 0) begin : g_flush_rsp
            wire flush_rsp_sel = flush_rsp_valid && ~per_bank_core_rsp_valid[0];
            assign core_rsp_xbar_valid[0] = per_bank_core_rsp_valid[0] || flush_rsp_valid;
            assign core_rsp_xbar_pdata[0] = flush_rsp_sel ? {`CS_WORD_WIDTH'(0), flush_rsp_tag} : per_bank_core_rsp_pdata[0];
            assign core_rsp_xbar_idx[0]   = flush_rsp_sel ? flush_rsp_idx : per_bank_core_rsp_idx[0];
            assign per_bank_core_rsp_ready[0] = core_rsp_xbar_ready[0];
            assign flush_rsp_ready = core_rsp_xbar_ready[0] && ~per_bank_core_rsp_valid[0];
        end else begin : g_bank_rsp
            assign core_rsp_xbar_valid[i] = per_bank_core_rsp_valid[i];
            assign core_rsp_xbar_pdata[i] = per_bank_core_rsp_pdata[i];
            assign core_rsp_xbar_idx[i]   = per_bank_core_rsp_idx[i];
            assign per_bank_core_rsp_ready[i] = core_rsp_xbar_ready[i];
        end
    end
    if (DO_READ_MERGE) begin : g_core_rsp_mcast
        // A merged read's response goes to its leader's port and to every
        // merged port: each output takes it when its arbiter grants it, and the
        // bank's response is released once all its targets have taken it.
        // the targets carry the port index
        `UNUSED_VAR (core_rsp_xbar_idx)
        wire [NUM_BANKS-1:0][NUM_REQS-1:0] lane_pending;
        wire [NUM_BANKS-1:0][NUM_REQS-1:0] lane_taken;
        wire [NUM_REQS-1:0][NUM_BANKS-1:0] per_output_valid;
        wire [NUM_REQS-1:0][NUM_BANKS-1:0] per_output_ready;
        for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_lanes
            wire [NUM_REQS-1:0] targets;
            if (i == 0) begin : g_flush_targets
                assign targets = per_bank_core_rsp_valid[0] ? per_bank_core_rsp_targets[0]
                                                           : flush_rsp_onehot;
            end else begin : g_bank_targets
                assign targets = per_bank_core_rsp_targets[i];
            end
            reg [NUM_REQS-1:0] sent_r;
            assign lane_pending[i] = targets & ~sent_r & {NUM_REQS{core_rsp_xbar_valid[i]}};
            for (genvar j = 0; j < NUM_REQS; ++j) begin : g_taken
                assign per_output_valid[j][i] = lane_pending[i][j];
                assign lane_taken[i][j] = lane_pending[i][j] && per_output_ready[j][i];
            end
            assign core_rsp_xbar_ready[i] = core_rsp_xbar_valid[i] && ((lane_pending[i] & ~lane_taken[i]) == '0);
            always @(posedge clk) begin
                if (reset) begin
                    sent_r <= '0;
                end else if (core_rsp_xbar_valid[i]) begin
                    sent_r <= core_rsp_xbar_ready[i] ? '0 : (sent_r | lane_taken[i]);
                end
            end
        end
        for (genvar j = 0; j < NUM_REQS; ++j) begin : g_outputs
            VX_stream_arb #(
                .NUM_INPUTS  (NUM_BANKS),
                .NUM_OUTPUTS (1),
                .DATAW       (CORE_RSP_DATAW),
                .ARBITER     ("R"),
                .MAX_FANOUT  ((NUM_BANKS > 12) ? 4 : `MAX_FANOUT),
                .OUT_BUF     (CORE_OUT_BUF)
            ) core_rsp_arb (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (per_output_valid[j]),
                .data_in   (core_rsp_xbar_pdata),
                .ready_in  (per_output_ready[j]),
                .valid_out (core_rsp_valid[j]),
                .data_out  (core_rsp_pdata[j]),
                `UNUSED_PIN (sel_out),
                .ready_out (core_rsp_ready[j])
            );
        end
    end else begin : g_core_rsp_xbar
        `UNUSED_VAR (per_bank_core_rsp_targets)
        `UNUSED_VAR (flush_rsp_onehot)
        VX_stream_omega #(
            .NUM_INPUTS  (NUM_BANKS),
            .NUM_OUTPUTS (NUM_REQS),
            .DATAW       (CORE_RSP_DATAW),
            .ARBITER     ("R"),
            .OUT_BUF     (CORE_OUT_BUF)
        ) core_rsp_xbar (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (core_rsp_xbar_valid),
            .data_in   (core_rsp_xbar_pdata),
            .sel_in    (core_rsp_xbar_idx),
            .ready_in  (core_rsp_xbar_ready),
            .valid_out (core_rsp_valid),
            .data_out  (core_rsp_pdata),
            `UNUSED_PIN (sel_out),
            .ready_out (core_rsp_ready),
            `UNUSED_PIN (collisions),
            `UNUSED_PIN (empty)
        );
    end
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_rsp
        assign core_bus_if[i].rsp_valid = core_rsp_valid[i];
        assign {core_bus_if[i].rsp_data.data, core_bus_if[i].rsp_data.tag} = core_rsp_pdata[i];
        assign core_rsp_ready[i] = core_bus_if[i].rsp_ready;
    end
    wire [NUM_BANKS-1:0][MEM_REQ_DATAW-1:0] per_bank_mem_req_pdata;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_mem_req_pdata
        // the raw bank bits of the memory address, formed while the bank id is a constant
        wire [`UP(BANK_SEL_BITS)-1:0] raw_bid;
        if (NUM_BANKS > 1 && BANK_HASH != 0) begin : g_hash
            assign raw_bid = BANK_SEL_BITS'(i) ^ BANK_SEL_BITS'(cache_bank_hash(
                32'(per_bank_mem_req_addr[i][`CS_LINE_SECTOR_ADDR_WIDTH-1 -: `CS_LINE_ADDR_WIDTH]), BANK_SEL_BITS));
        end else begin : g_raw
            assign raw_bid = `UP(BANK_SEL_BITS)'(i);
        end
        assign per_bank_mem_req_pdata[i] = {
            raw_bid,
            per_bank_mem_req_addr[i],
            per_bank_mem_req_rw[i],
            per_bank_mem_req_byteen[i],
            per_bank_mem_req_data[i],
            per_bank_mem_req_tag[i],
            per_bank_mem_req_attr[i]
        };
    end
    wire [NUM_BANKS-1:0][MEM_PORTS_SEL_WIDTH-1:0] per_bank_mem_req_bid;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_per_bank_mem_req_bid
        if (MEM_PORTS > 1) begin : g_multiports
            assign per_bank_mem_req_bid[i] = MEM_PORTS_SEL_WIDTH'(i[MEM_PORTS_SEL_BITS-1:0]);
        end else begin : g_singleport
            assign per_bank_mem_req_bid[i] = 0;
        end
    end
    wire [MEM_PORTS-1:0][MEM_REQ_DATAW-1:0] mem_req_pdata;
    wire [MEM_PORTS-1:0]                    mem_req_tmp_valid;
    wire [MEM_PORTS-1:0][`UP(`LOG2UP(NUM_BANKS))-1:0] mem_req_tmp_idx;
    wire [MEM_PORTS-1:0]                    mem_req_tmp_ready;
    // The flush token for port i is a read of the port's first line with the
    // flush attribute; byteen bit 0 carries the L1-only kind and the io
    // attribute the IO kind. It takes bank
    // i's lane into the request crossbar (bank i maps to port i), so the
    // crossbar forms its address and it leaves through the same registered
    // outputs as the write-backs ahead of it.
    wire [NUM_BANKS-1:0]                    mem_req_xbar_valid;
    wire [NUM_BANKS-1:0][MEM_REQ_DATAW-1:0] mem_req_xbar_pdata;
    wire [NUM_BANKS-1:0]                    mem_req_xbar_ready;
    for (genvar i = 0; i < NUM_BANKS; ++i) begin : g_mem_req_xbar_in
        if (i < MEM_PORTS) begin : g_flush_req
            // line 0 hashes to 0, so the raw bank bits are the lane's bank id
            wire [MEM_REQ_DATAW-1:0] mem_flush_pdata = {
                `UP(BANK_SEL_BITS)'(i),
                `CS_LINE_SECTOR_ADDR_WIDTH'(0),
                1'b0,
                SECTOR_SIZE'(mem_flush_l1_only),
                `CS_SECTOR_WIDTH'(0),
                BANK_MEM_TAG_WIDTH'(0),
                (`UP(MEM_ATTR_WIDTH)'(1) << MEM_ATTR_FLUSH_OFFS) | (`UP(MEM_ATTR_WIDTH)'(mem_flush_io) << MEM_ATTR_IO_OFFS)
            };
            assign mem_req_xbar_valid[i] = per_bank_mem_req_valid[i] || mem_flush_valid[i];
            assign mem_req_xbar_pdata[i] = per_bank_mem_req_valid[i] ? per_bank_mem_req_pdata[i] : mem_flush_pdata;
            assign mem_flush_ready[i]    = mem_req_xbar_ready[i] && ~per_bank_mem_req_valid[i];
            `RUNTIME_ASSERT (~(mem_flush_valid[i] && per_bank_mem_req_valid[i]), ("%t: %s flush token collides with a bank %0d request", $time, INSTANCE_ID, i))
        end else begin : g_bank_req
            assign mem_req_xbar_valid[i] = per_bank_mem_req_valid[i];
            assign mem_req_xbar_pdata[i] = per_bank_mem_req_pdata[i];
        end
        assign per_bank_mem_req_ready[i] = mem_req_xbar_ready[i];
    end
    // Write-backs from different banks are not ordered with each other inside
    // the crossbar, so the tokens wait until it has drained.
    wire mem_req_xbar_empty;
    assign mem_req_empty = (FLUSH_FORWARD != 0) ? mem_req_xbar_empty : 1'b1;
    VX_stream_omega #(
        .NUM_INPUTS  (NUM_BANKS),
        .NUM_OUTPUTS (MEM_PORTS),
        .DATAW       (MEM_REQ_DATAW),
        .ARBITER     ("R"),
        .OUT_BUF     (MEM_OUT_BUF)
    ) mem_req_xbar (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (mem_req_xbar_valid),
        .data_in   (mem_req_xbar_pdata),
        .sel_in    (per_bank_mem_req_bid),
        .ready_in  (mem_req_xbar_ready),
        .valid_out (mem_req_tmp_valid),
        .data_out  (mem_req_pdata),
        .sel_out   (mem_req_tmp_idx),
        .ready_out (mem_req_tmp_ready),
        `UNUSED_PIN (collisions),
        .empty     (mem_req_xbar_empty)
    );
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_tmp_if
        wire [`UP(BANK_SEL_BITS)-1:0]   mem_req_raw_bid_w;
        wire                            mem_req_rw_w;
        wire [`CS_LINE_SECTOR_ADDR_WIDTH-1:0] mem_req_addr_w;
        wire [SECTOR_SIZE-1:0]          mem_req_byteen_w;
        wire [`CS_SECTOR_WIDTH-1:0]     mem_req_data_w;
        wire [BANK_MEM_TAG_WIDTH-1:0]   mem_req_tag_w;
        wire [`UP(MEM_ATTR_WIDTH)-1:0]  mem_req_attr_w;
        assign {
            mem_req_raw_bid_w,
            mem_req_addr_w,
            mem_req_rw_w,
            mem_req_byteen_w,
            mem_req_data_w,
            mem_req_tag_w,
            mem_req_attr_w
        } = mem_req_pdata[i];
        assign mem_bus_tmp_if[i].req_valid       = mem_req_tmp_valid[i];
        assign mem_bus_tmp_if[i].req_data.rw     = mem_req_rw_w;
        assign mem_bus_tmp_if[i].req_data.byteen = mem_req_byteen_w;
        assign mem_bus_tmp_if[i].req_data.data   = mem_req_data_w;
        assign mem_bus_tmp_if[i].req_data.attr   = mem_req_attr_w;
        wire [`CS_LINE_ADDR_WIDTH-1:0] mem_req_line_w = mem_req_addr_w[`CS_LINE_SECTOR_ADDR_WIDTH-1 -: `CS_LINE_ADDR_WIDTH];
        wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] mem_req_sec_w;
        if (`CS_SECTOR_SEL_BITS != 0) begin : g_sec_w
            assign mem_req_sec_w = mem_req_addr_w[`CS_SECTOR_SEL_BITS-1:0];
        end else begin : g_no_sec_w
            assign mem_req_sec_w = '0;
        end
        if (NUM_BANKS > 1) begin : g_multibanks
            if (NUM_BANKS != MEM_PORTS) begin : g_arb_sel
                wire [MEM_ARB_SEL_BITS-1:0] mem_req_arb_sel;
                assign mem_req_arb_sel = mem_req_tmp_idx[i][`LOG2UP(NUM_BANKS)-1 -: MEM_ARB_SEL_BITS];
                assign mem_bus_tmp_if[i].req_data.tag  = {mem_req_tag_w, mem_req_arb_sel};
            end else begin : g_no_arb_sel
                `UNUSED_VAR (mem_req_tmp_idx)
                assign mem_bus_tmp_if[i].req_data.tag  = MEM_TAG_WIDTH'(mem_req_tag_w);
            end
            assign mem_bus_tmp_if[i].req_data.addr = (`CS_MEM_SECTOR_ADDR_WIDTH'({mem_req_line_w, mem_req_raw_bid_w}) << `CS_SECTOR_SEL_BITS)
                                                   | `CS_MEM_SECTOR_ADDR_WIDTH'(mem_req_sec_w);
        end else begin : g_singlebank
            `UNUSED_VAR (mem_req_tmp_idx)
            `UNUSED_VAR (mem_req_raw_bid_w)
            `UNUSED_VAR (mem_req_line_w)
            `UNUSED_VAR (mem_req_sec_w)
            assign mem_bus_tmp_if[i].req_data.addr = `CS_MEM_SECTOR_ADDR_WIDTH'(mem_req_addr_w);
            assign mem_bus_tmp_if[i].req_data.tag  = MEM_TAG_WIDTH'(mem_req_tag_w);
        end
        assign mem_req_tmp_ready[i] = mem_bus_tmp_if[i].req_ready;
    end
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_mem_bus_if
        if (WRITE_ENABLE) begin : g_rw
            `ASSIGN_VX_MEM_BUS_IF (mem_bus_if[i], mem_bus_tmp_if[i]);
        end else begin : g_ro
            `ASSIGN_VX_MEM_BUS_RO_IF (mem_bus_if[i], mem_bus_tmp_if[i]);
        end
    end
`ifdef PERF_ENABLE
    wire [NUM_REQS-1:0]  perf_core_reads_per_req;
    wire [NUM_REQS-1:0]  perf_core_writes_per_req;
    wire [NUM_REQS-1:0]  perf_crsp_stall_per_req;
    wire [MEM_PORTS-1:0] perf_mem_stall_per_port;
    // an admitted request is accepted with its stage's or its leader's access
    wire [NUM_REQS-1:0]  core_req_ready = (core_req_follower & core_req_merge_ready)
                                        | (~core_req_follower & core_req_stage_ready);
    // a merged read counts once, as the bank access that serves it
    `BUFFER(perf_core_reads_per_req, core_req_valid & core_req_ready & ~core_req_rw & ~core_req_follower);
    `BUFFER(perf_core_writes_per_req, core_req_valid & core_req_ready & core_req_rw);
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_perf_crsp_stall_per_req
        assign perf_crsp_stall_per_req[i] = core_bus_if[i].rsp_valid && ~core_bus_if[i].rsp_ready;
    end
    for (genvar i = 0; i < MEM_PORTS; ++i) begin : g_perf_mem_stall_per_port
        assign perf_mem_stall_per_port[i] = mem_bus_if[i].req_valid && ~mem_bus_if[i].req_ready;
    end
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_core_reads_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_core_writes_per_cycle;
    wire [`CLOG2(NUM_REQS+1)-1:0]  perf_crsp_stall_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_read_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_write_miss_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_evictions_per_cycle;
    wire [`CLOG2(NUM_BANKS+1)-1:0] perf_mshr_stall_per_cycle;
    wire [`CLOG2(MEM_PORTS+1)-1:0] perf_mem_stall_per_cycle;
    `POP_COUNT(perf_core_reads_per_cycle, perf_core_reads_per_req);
    `POP_COUNT(perf_core_writes_per_cycle, perf_core_writes_per_req);
    `POP_COUNT(perf_read_miss_per_cycle, perf_read_miss_per_bank);
    `POP_COUNT(perf_write_miss_per_cycle, perf_write_miss_per_bank);
    `POP_COUNT(perf_evictions_per_cycle, perf_evictions_per_bank);
    `POP_COUNT(perf_mshr_stall_per_cycle, perf_mshr_stall_per_bank);
    `POP_COUNT(perf_crsp_stall_per_cycle, perf_crsp_stall_per_req);
    `POP_COUNT(perf_mem_stall_per_cycle, perf_mem_stall_per_port);
    reg [PERF_CTR_BITS-1:0] perf_core_reads;
    reg [PERF_CTR_BITS-1:0] perf_core_writes;
    reg [PERF_CTR_BITS-1:0] perf_read_misses;
    reg [PERF_CTR_BITS-1:0] perf_write_misses;
    reg [PERF_CTR_BITS-1:0] perf_evictions;
    reg [PERF_CTR_BITS-1:0] perf_mshr_stalls;
    reg [PERF_CTR_BITS-1:0] perf_mem_stalls;
    reg [PERF_CTR_BITS-1:0] perf_crsp_stalls;
    always @(posedge clk) begin
        if (reset) begin
            perf_core_reads   <= '0;
            perf_core_writes  <= '0;
            perf_read_misses  <= '0;
            perf_write_misses <= '0;
            perf_evictions    <= '0;
            perf_mshr_stalls  <= '0;
            perf_mem_stalls   <= '0;
            perf_crsp_stalls  <= '0;
        end else begin
            perf_core_reads   <= perf_core_reads   + PERF_CTR_BITS'(perf_core_reads_per_cycle);
            perf_core_writes  <= perf_core_writes  + PERF_CTR_BITS'(perf_core_writes_per_cycle);
            perf_read_misses  <= perf_read_misses  + PERF_CTR_BITS'(perf_read_miss_per_cycle);
            perf_write_misses <= perf_write_misses + PERF_CTR_BITS'(perf_write_miss_per_cycle);
            perf_evictions    <= perf_evictions    + PERF_CTR_BITS'(perf_evictions_per_cycle);
            perf_mshr_stalls  <= perf_mshr_stalls  + PERF_CTR_BITS'(perf_mshr_stall_per_cycle);
            perf_mem_stalls   <= perf_mem_stalls   + PERF_CTR_BITS'(perf_mem_stall_per_cycle);
            perf_crsp_stalls  <= perf_crsp_stalls  + PERF_CTR_BITS'(perf_crsp_stall_per_cycle);
        end
    end
    assign cache_perf.reads        = perf_core_reads;
    assign cache_perf.writes       = perf_core_writes;
    assign cache_perf.read_misses  = perf_read_misses;
    assign cache_perf.write_misses = perf_write_misses;
    assign cache_perf.evictions    = perf_evictions;
    assign cache_perf.bank_stalls  = perf_collisions;
    assign cache_perf.mshr_stalls  = perf_mshr_stalls;
    assign cache_perf.mem_stalls   = perf_mem_stalls;
    assign cache_perf.crsp_stalls  = perf_crsp_stalls;
`endif
endmodule
