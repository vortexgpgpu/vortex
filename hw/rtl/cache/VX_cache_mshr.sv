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

// This is an implementation of a MSHR for pipelined multi-banked cache.
// We allocate a free slot from the MSHR before processing a core request
// and release the slot when we get a cache hit. This ensure that we do not
// enter the cache bank pipeline when the MSHR is full.
// During a memory fill response, we initiate the replay sequence
// and dequeue all pending entries for the given cache line.
//
// Pending core requests stored in the MSHR are sorted by the order of
// arrival and are dequeued in the same order.
// Each entry has a next pointer to the next entry pending for the same cache line.
//
// During the fill request, the MSHR will release the MSHR entry at fill_id
// which represents the first request in the pending list that initiated the memory fill.
//
// The dequeue response directly follows the fill request and will release
// all the subsequent entries linked to fill_id (pending the same cache line).
//
// During the allocation request, the MSHR will allocate the next free slot
// for the incoming core request. We return the allocated slot id as well as
// the slot id of the previous entry for the same cache line. This is used to
// link the new entry to the pending list.
//
// The release request is used to invalidate the allocated MSHR entry if we had a cache hit.
//
// Warning: This MSHR implementation is strongly coupled with the bank pipeline
// and as such changes to either module requires careful evaluation.
//

module VX_cache_mshr #(
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 16,
    // Number of banks
    parameter NUM_BANKS         = 1,
    // Miss Reserv Queue Knob
    parameter MSHR_SIZE         = 4,
    // Request debug identifier
    parameter UUID_WIDTH        = 0,
    // MSHR parameters
    parameter DATA_WIDTH        = 1,
    // Enable cache writeback
    parameter WRITEBACK         = 0,
    // Cache stall on read during write
    RDW_STALL                   = 0,

    parameter MSHR_ADDR_WIDTH   = `LOG2UP(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     deq_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     alc_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     rel_req_uuid,
`IGNORE_UNUSED_END

    // memory fill
    input wire                          fill_valid,
    input wire [MSHR_ADDR_WIDTH-1:0]    fill_id,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] fill_addr,

    // dequeue
    output wire                         dequeue_valid,
    output wire [`CS_LINE_ADDR_WIDTH-1:0] dequeue_addr,
    output wire                         dequeue_rw,
    output wire [DATA_WIDTH-1:0]        dequeue_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   dequeue_id,
    input wire                          dequeue_ready,

    // allocate
    input wire                          allocate_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] allocate_addr,
    input wire                          allocate_rw,
    input wire [DATA_WIDTH-1:0]         allocate_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   allocate_id,
    output wire                         allocate_pending,
    output wire                         allocate_ready,

    // release
    input wire                          release_valid,
    input wire [MSHR_ADDR_WIDTH-1:0]    release_id
);
    `UNUSED_PARAM (BANK_ID)

    reg [`CS_LINE_ADDR_WIDTH-1:0] addr_table [MSHR_SIZE-1:0];
    reg [MSHR_ADDR_WIDTH-1:0] next_index [MSHR_SIZE-1:0];

    reg [MSHR_SIZE-1:0] valid_table, valid_table_n;
    reg [MSHR_SIZE-1:0] next_table, next_table_x, next_table_n;
    reg [MSHR_SIZE-1:0] write_table;

    reg allocate_rdy, allocate_rdy_n;
    reg [MSHR_ADDR_WIDTH-1:0] allocate_id_r, allocate_id_n;

    reg dequeue_val, dequeue_val_n;
    reg [MSHR_ADDR_WIDTH-1:0] dequeue_id_r, dequeue_id_n;

    wire [MSHR_ADDR_WIDTH-1:0] prev_idx;
    reg [MSHR_ADDR_WIDTH-1:0] post_alloc_id, post_alloc_previd;
    reg post_alloc_val;

    wire allocate_fire = allocate_valid && allocate_ready;
    wire dequeue_fire = dequeue_valid && dequeue_ready;

    wire [MSHR_SIZE-1:0] addr_matches;
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin : g_addr_matches
        assign addr_matches[i] = valid_table[i] && (addr_table[i] == allocate_addr);
    end

    VX_lzc #(
        .N (MSHR_SIZE),
        .REVERSE (1)
    ) allocate_sel (
        .data_in   (~valid_table_n),
        .data_out  (allocate_id_n),
        .valid_out (allocate_rdy_n)
    );

    // find matching tail-entry
    VX_priority_encoder #(
        .N (MSHR_SIZE)
    ) prev_sel (
        .data_in (addr_matches & ~next_table_x),
        .index_out (prev_idx),
        `UNUSED_PIN (onehot_out),
        `UNUSED_PIN (valid_out)
    );

    always @(*) begin
        valid_table_n = valid_table;
        next_table_x  = next_table;
        dequeue_val_n = dequeue_val;
        dequeue_id_n  = dequeue_id;

        if (fill_valid) begin
            dequeue_val_n = 1;
            dequeue_id_n = fill_id;
        end

        if (dequeue_fire) begin
            valid_table_n[dequeue_id] = 0;
            if (next_table[dequeue_id]) begin
                dequeue_id_n = next_index[dequeue_id];
            end else if (!RDW_STALL && post_alloc_val && (post_alloc_previd == dequeue_id)) begin
                dequeue_id_n = post_alloc_id;
            end else begin
                dequeue_val_n = 0;
            end
        end

        if (release_valid) begin
            valid_table_n[release_id] = 0;
        end

        if (post_alloc_val) begin
            next_table_x[post_alloc_previd] = 1;
        end

        next_table_n = next_table_x;
        if (allocate_fire) begin
            valid_table_n[allocate_id] = 1;
            next_table_n[allocate_id] = 0;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            valid_table  <= '0;
            allocate_rdy <= 0;
            dequeue_val  <= 0;
            post_alloc_val <= 0;
        end else begin
            valid_table  <= valid_table_n;
            allocate_rdy <= allocate_rdy_n;
            dequeue_val  <= dequeue_val_n;
            post_alloc_val <= allocate_fire && allocate_pending;
        end

        if (allocate_fire) begin
            addr_table[allocate_id] <= allocate_addr;
            write_table[allocate_id] <= allocate_rw;
        end

        if (post_alloc_val) begin
            next_index[post_alloc_previd] <= post_alloc_id;
        end

        dequeue_id_r  <= dequeue_id_n;
        allocate_id_r <= allocate_id_n;
        next_table    <= next_table_n;
        post_alloc_id <= allocate_id;
        post_alloc_previd <= prev_idx;
    end

    `RUNTIME_ASSERT((~allocate_fire || ~valid_table[allocate_id_r]), ("%t: *** %s inuse allocation: addr=0x%0h, id=%0d (#%0d)", $time, INSTANCE_ID,
        `CS_LINE_TO_FULL_ADDR(allocate_addr, BANK_ID), allocate_id_r, alc_req_uuid))

    `RUNTIME_ASSERT((~release_valid || valid_table[release_id]), ("%t: *** %s invalid release: addr=0x%0h, id=%0d (#%0d)", $time, INSTANCE_ID,
        `CS_LINE_TO_FULL_ADDR(addr_table[release_id], BANK_ID), release_id, rel_req_uuid))

    `RUNTIME_ASSERT((~fill_valid || valid_table[fill_id]), ("%t: *** %s invalid fill: addr=0x%0h, id=%0d", $time, INSTANCE_ID,
        `CS_LINE_TO_FULL_ADDR(addr_table[fill_id], BANK_ID), fill_id))

    VX_dp_ram #(
        .DATAW (DATA_WIDTH),
        .SIZE  (MSHR_SIZE),
        .RADDR_REG (1)
    ) entries (
        .clk   (clk),
        .reset (reset),
        .read  (1'b1),
        .write (allocate_valid),
        .wren  (1'b1),
        .waddr (allocate_id_r),
        .wdata (allocate_data),
        .raddr (dequeue_id_n),
        .rdata (dequeue_data)
    );

    assign fill_addr = addr_table[fill_id];

    assign allocate_ready = allocate_rdy;
    assign allocate_id    = allocate_id_r;
    if (WRITEBACK) begin : g_pending_wb
        assign allocate_pending = |addr_matches;
    end else begin : g_pending_wt
        // exclude write requests if writethrough
        assign allocate_pending = |(addr_matches & ~write_table);
    end

    assign dequeue_valid = dequeue_val;
    assign dequeue_addr  = addr_table[dequeue_id_r];
    assign dequeue_rw    = write_table[dequeue_id_r];
    assign dequeue_id    = dequeue_id_r;

`ifdef DBG_TRACE_CACHE
    reg show_table;
    always @(posedge clk) begin
        if (reset) begin
            show_table <= 0;
        end else begin
            show_table <= allocate_fire || post_alloc_val || release_valid || fill_valid || dequeue_fire;
        end
        if (allocate_fire) begin
            `TRACE(3, ("%t: %s allocate: addr=0x%0h, id=%0d, pending=%b, prev=%0d (#%0d)\n", $time, INSTANCE_ID,
                `CS_LINE_TO_FULL_ADDR(allocate_addr, BANK_ID), allocate_id, allocate_pending, prev_idx, alc_req_uuid))
        end
        if (release_valid) begin
            `TRACE(3, ("%t: %s release: id=%0d (#%0d)\n", $time, INSTANCE_ID, release_id, rel_req_uuid))
        end
        if (fill_valid) begin
            `TRACE(3, ("%t: %s fill: addr=0x%0h, id=%0d\n", $time, INSTANCE_ID,
                `CS_LINE_TO_FULL_ADDR(fill_addr, BANK_ID), fill_id))
        end
        if (dequeue_fire) begin
            `TRACE(3, ("%t: %s dequeue: addr=0x%0h, id=%0d (#%0d)\n", $time, INSTANCE_ID,
                `CS_LINE_TO_FULL_ADDR(dequeue_addr, BANK_ID), dequeue_id_r, deq_req_uuid))
        end
        if (show_table) begin
            `TRACE(3, ("%t: %s table", $time, INSTANCE_ID))
            for (integer i = 0; i < MSHR_SIZE; ++i) begin
                if (valid_table[i]) begin
                    `TRACE(3, (" %0d=0x%0h", i, `CS_LINE_TO_FULL_ADDR(addr_table[i], BANK_ID)))
                    if (write_table[i]) begin
                        `TRACE(3, ("(w)"))
                    end else begin
                        `TRACE(3, ("(r)"))
                    end
                    if (next_table[i])  begin
                        `TRACE(3, ("->%0d", next_index[i]))
                    end
                end
            end
            `TRACE(3, ("\n"))
        end
    end
`endif

endmodule
