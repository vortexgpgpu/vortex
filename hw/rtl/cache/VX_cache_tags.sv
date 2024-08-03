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

module VX_cache_tags #(
    parameter `STRING INSTANCE_ID = "",
    parameter BANK_ID       = 0,
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
    // Enable cache writeback
    parameter WRITEBACK     = 0,
    // Request debug identifier
    parameter UUID_WIDTH    = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire [`UP(UUID_WIDTH)-1:0]    req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    // init/fill/lookup
    input wire                          init,
    input wire                          flush,
    input wire                          fill,
    input wire                          write,
    input wire                          lookup,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] line_addr,
    input wire [NUM_WAYS-1:0]           way_sel,
    output wire [NUM_WAYS-1:0]          tag_matches,

    // eviction
    output wire                         evict_dirty,
    output wire [NUM_WAYS-1:0]          evict_way,
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (lookup)

    //                   valid,   dirty,           tag
    localparam TAG_WIDTH = 1 +  WRITEBACK + `CS_TAG_SEL_BITS;

    wire [`CS_LINE_SEL_BITS-1:0] line_sel = line_addr[`CS_LINE_SEL_BITS-1:0];
    wire [`CS_TAG_SEL_BITS-1:0] line_tag = `CS_LINE_ADDR_TAG(line_addr);

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0] read_valid;
    wire [NUM_WAYS-1:0] read_dirty;

    if (NUM_WAYS > 1)  begin
        reg [NUM_WAYS-1:0] evict_way_r;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset) begin
                evict_way_r <= 1;
            end else if (~stall) begin // holding the value on stalls prevents filling different slots twice
                evict_way_r <= {evict_way_r[NUM_WAYS-2:0], evict_way_r[NUM_WAYS-1]};
            end
        end

        assign evict_way = fill ? evict_way_r : way_sel;

        VX_onehot_mux #(
            .DATAW (`CS_TAG_SEL_BITS),
            .N     (NUM_WAYS)
        ) evict_tag_sel (
            .data_in  (read_tag),
            .sel_in   (evict_way),
            .data_out (evict_tag)
        );
    end else begin
        `UNUSED_VAR (stall)
        assign evict_way = 1'b1;
        assign evict_tag = read_tag;
    end

    // fill and flush need to also read in writeback mode
    wire fill_s = fill && (!WRITEBACK || ~stall);
    wire flush_s = flush && (!WRITEBACK || ~stall);

    for (genvar i = 0; i < NUM_WAYS; ++i) begin

        wire do_fill    = fill_s  && evict_way[i];
        wire do_flush   = flush_s && (!WRITEBACK || way_sel[i]); // flush the whole line in writethrough mode
        wire do_write   = WRITEBACK && write && tag_matches[i];

        wire line_read  = (WRITEBACK && (fill_s || flush_s));
        wire line_write = init || do_fill || do_flush || do_write;
        wire line_valid = ~(init || flush);

        wire [TAG_WIDTH-1:0] line_wdata;
        wire [TAG_WIDTH-1:0] line_rdata;

        if (WRITEBACK) begin
            assign line_wdata = {line_valid, write, line_tag};
            assign {read_valid[i], read_dirty[i], read_tag[i]} = line_rdata;
        end else begin
            assign line_wdata = {line_valid, line_tag};
            assign {read_valid[i], read_tag[i]} = line_rdata;
            assign read_dirty[i] = 1'b0;
        end

        VX_sp_ram #(
            .DATAW (TAG_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .NO_RWCHECK (1),
            .RW_ASSERT (1)
        ) tag_store (
            .clk   (clk),
            .reset (reset),
            .read  (line_read),
            .write (line_write),
            .wren  (1'b1),
            .addr  (line_sel),
            .wdata (line_wdata),
            .rdata (line_rdata)
        );
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin
        assign tag_matches[i] = read_valid[i] && (line_tag == read_tag[i]);
    end

    assign evict_dirty = | (read_dirty & evict_way);

`ifdef DBG_TRACE_CACHE
    wire [`CS_LINE_ADDR_WIDTH-1:0] evict_line_addr = {evict_tag, line_sel};
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s fill: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h, dirty=%b, evict_addr=0x%0h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), evict_way, line_sel, line_tag, evict_dirty, `CS_LINE_TO_FULL_ADDR(evict_line_addr, BANK_ID)));
        end
        if (init) begin
            `TRACE(3, ("%d: %s init: addr=0x%0h, blk_addr=%0d\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), line_sel));
        end
        if (flush && ~stall) begin
            `TRACE(3, ("%d: %s flush: addr=0x%0h, way=%b, blk_addr=%0d, dirty=%b\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(evict_line_addr, BANK_ID), way_sel, line_sel, evict_dirty));
        end
        if (lookup && ~stall) begin
            if (tag_matches != 0) begin
                if (write)
                    `TRACE(3, ("%d: %s write-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), tag_matches, line_sel, line_tag, req_uuid));
                else
                    `TRACE(3, ("%d: %s read-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), tag_matches, line_sel, line_tag, req_uuid));
            end else begin
                if (write)
                    `TRACE(3, ("%d: %s write-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), line_sel, line_tag, req_uuid));
                else
                    `TRACE(3, ("%d: %s read-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), line_sel, line_tag, req_uuid));
            end
        end
    end
`endif

endmodule
