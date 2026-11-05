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

// One input of the cache, up to the bank arbiters: the kind of flush token it
// shows and whether that token is still to be answered (the flush sequencer,
// VX_cache_init, holds what all inputs share), the input's admission, its bank
// select (hashed with BANK_HASH), the request packed for the banks, the
// request stage, and the one-hot request to the banks with its ready.
// Every input of a cache is the same module with the same parameters; the
// logic that combines inputs (the sequencer's reductions, read merging, the
// bank arbiters) stays in VX_cache.
module VX_cache_req_port import VX_gpu_pkg::*; #(
    parameter NUM_BANKS       = 1,  // Number of banks
    parameter WORD_SIZE       = 4,  // Word size in bytes
    parameter WORD_SEL_BITS   = 0,  // Word select bits of the word address
    parameter ADDR_WIDTH      = 30, // Word address width
    parameter TAG_WIDTH       = 1,  // Request tag width
    parameter MERGE_GROUP     = 1,  // Read-merge group size (1: no read merging)
    parameter BANK_HASH       = 0,  // Hashed bank select (cache_bank_hash)
    parameter SHARED_LLC      = 0,  // The LLC serving every memory client alone (VX_cache_init)
    parameter REQ_STAGE       = 0,  // A request stage before the bank arbiters
    // The banks' ready is their buffers' ready, whatever the request (one
    // input, no arbitration): the selected bank's ready answers. Otherwise a
    // bank's arbiter is ready only to a valid request it grants.
    parameter SEL_READY       = 0,

    parameter BANK_SEL_BITS   = `CLOG2(NUM_BANKS),
    parameter BANK_SEL_WIDTH  = `UP(BANK_SEL_BITS),
    parameter WORD_SEL_WIDTH  = `UP(WORD_SEL_BITS),
    parameter LINE_ADDR_WIDTH = ADDR_WIDTH - BANK_SEL_BITS - WORD_SEL_BITS,
    parameter MERGE_WIDTH     = (MERGE_GROUP > 1) ? MERGE_GROUP : 0,
    parameter BANK_TAG_WIDTH  = TAG_WIDTH + MERGE_WIDTH,
    parameter ATTR_WIDTH      = `UP(MEM_ATTR_WIDTH),
    parameter DATAW           = LINE_ADDR_WIDTH + 1 + WORD_SEL_WIDTH + WORD_SIZE + 8 * WORD_SIZE + BANK_TAG_WIDTH + ATTR_WIDTH
) (
    input wire                      clk,
    input wire                      reset,

    // the requester
    input wire                      in_valid,
    input wire                      in_rw,
    input wire [ADDR_WIDTH-1:0]     in_addr,
    input wire [8*WORD_SIZE-1:0]    in_data,
    input wire [WORD_SIZE-1:0]      in_byteen,
    input wire [ATTR_WIDTH-1:0]     in_attr,
    input wire [TAG_WIDTH-1:0]      in_tag,
    output wire                     in_ready,

    // the flush sequencer
    output wire                     flush,        // a token is visible here
    output wire                     flush_full,
    output wire                     flush_l1,
    output wire                     flush_io,
    input wire                      flush_hold,   // every input closed for a token
    input wire                      dbg_stall,
    input wire                      pending_load, // the visible tokens become pending
    input wire                      pending_owed, // only those a quiesce without a walk serves
    input wire                      rsp_take,     // this input's token is answered now
    output wire                     pending,      // this input's token is still to be answered

    // the admitted request and its stage's ready, for read merging and the counters
    output wire                     req_valid,
    output wire                     req_ready,    // the request stage takes a request now
    input wire                      merge_follower, // served by a lower input's access
    input wire                      merge_ready,    // that access is accepted now
    input wire [`UP(MERGE_WIDTH)-1:0] merge_mask,   // the inputs this one's access serves

    // the banks
    output wire                     stage_valid,  // a request is held for the banks
    output wire [NUM_BANKS-1:0]     bank_valid,
    output wire [DATAW-1:0]         bank_data,
    input wire [NUM_BANKS-1:0]      bank_ready
);
    // the flush token's kind
    assign flush      = in_valid && in_attr[MEM_ATTR_FLUSH_OFFS];
    assign flush_io   = flush && in_attr[MEM_ATTR_IO_OFFS];
    assign flush_l1   = flush && ~flush_io && (| in_byteen);
    assign flush_full = flush && ~flush_l1 && ~flush_io;

    // whether the token is still to be answered
    wire owed = flush_io || ((SHARED_LLC != 0) && flush_l1);
    reg pending_r;
    always @(posedge clk) begin
        if (reset) begin
            pending_r <= 1'b0;
        end else begin
            pending_r <= pending_load ? (pending_owed ? owed : flush) : (pending_r && ~rsp_take);
        end
    end
    assign pending = pending_r;

    // the debug backpressure (VX_cache_init) and read merging exist only
    // where they are configured
`ifdef DBG_FLUSH_STALL
    wire stall = dbg_stall;
`else
    wire stall = 1'b0;
    `UNUSED_VAR (dbg_stall)
`endif
    wire follower, follower_ready;
    if (MERGE_WIDTH != 0) begin : g_merge
        assign follower       = merge_follower;
        assign follower_ready = merge_ready;
    end else begin : g_no_merge
        assign follower       = 1'b0;
        assign follower_ready = 1'b0;
        `UNUSED_VAR (merge_follower)
        `UNUSED_VAR (merge_ready)
    end

    // Normal requests stay blocked while any token is visible. A token is
    // consumed when its answer is sent and never reaches the banks.
    wire input_enable = ~flush_hold && ~flush && ~stall;
    wire stage_in_ready;
    assign req_valid = in_valid && input_enable;
    assign req_ready = stage_in_ready;
    assign in_ready  = rsp_take || ((follower ? follower_ready : stage_in_ready) && input_enable);

    // the bank select and the request for the banks
    wire [LINE_ADDR_WIDTH-1:0] line_addr = in_addr[(BANK_SEL_BITS + WORD_SEL_BITS) +: LINE_ADDR_WIDTH];
    wire [BANK_SEL_WIDTH-1:0] bank_id;
    if (NUM_BANKS > 1) begin : g_multibanks
        wire [BANK_SEL_BITS-1:0] raw_bid = in_addr[WORD_SEL_BITS +: BANK_SEL_BITS];
        if (BANK_HASH != 0) begin : g_hash
            assign bank_id = raw_bid ^ BANK_SEL_BITS'(cache_bank_hash(32'(line_addr), BANK_SEL_BITS));
        end else begin : g_raw
            assign bank_id = raw_bid;
        end
    end else begin : g_singlebank
        assign bank_id = '0;
    end

    wire [WORD_SEL_WIDTH-1:0] wsel;
    if (WORD_SEL_BITS != 0) begin : g_wsel
        assign wsel = in_addr[0 +: WORD_SEL_BITS];
    end else begin : g_no_wsel
        assign wsel = '0;
    end

    wire [BANK_TAG_WIDTH-1:0] bank_tag;
    if (MERGE_WIDTH != 0) begin : g_merge_tag
        assign bank_tag = {in_tag, merge_mask};
    end else begin : g_no_merge_tag
        assign bank_tag = in_tag;
        `UNUSED_VAR (merge_mask)
    end

    wire [DATAW-1:0] req_data = {line_addr, in_rw, wsel, in_byteen, in_data, bank_tag, in_attr};

    // the request stage
    wire                      sel_valid;
    wire [BANK_SEL_WIDTH-1:0] sel_bid;
    wire                      sel_ready;
    if (REQ_STAGE != 0) begin : g_req_stage
        VX_elastic_buffer #(
            .DATAW   (BANK_SEL_WIDTH + DATAW),
            .SIZE    (2),
            .OUT_REG (0)
        ) req_stage (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (req_valid && ~follower),
            .data_in   ({bank_id, req_data}),
            .ready_in  (stage_in_ready),
            .valid_out (sel_valid),
            .data_out  ({sel_bid, bank_data}),
            .ready_out (sel_ready)
        );
    end else begin : g_no_req_stage
        assign sel_valid      = req_valid && ~follower;
        assign sel_bid        = bank_id;
        assign bank_data      = req_data;
        assign stage_in_ready = sel_ready;
    end
    assign stage_valid = sel_valid;

    // the request to its bank
    VX_demux #(
        .DATAW (1),
        .N     (NUM_BANKS)
    ) sel_demux (
        .sel_in   (sel_bid),
        .data_in  (sel_valid),
        .data_out (bank_valid)
    );
    if (SEL_READY != 0 && NUM_BANKS > 1) begin : g_sel_ready
        assign sel_ready = bank_ready[sel_bid];
    end else begin : g_any_ready
        assign sel_ready = (| bank_ready);
    end

endmodule
