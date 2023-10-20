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
    // Request debug identifier
    parameter UUID_WIDTH    = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire [`UP(UUID_WIDTH)-1:0]    req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    // read/fill
    input wire                          lookup,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] line_addr,
    input wire                          fill,    
    input wire                          init,
    output wire [NUM_WAYS-1:0]          way_sel,
    output wire [NUM_WAYS-1:0]          tag_matches
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (lookup)

    localparam TAG_WIDTH = 1 + `CS_TAG_SEL_BITS;

    wire [`CS_LINE_SEL_BITS-1:0] line_sel = line_addr[`CS_LINE_SEL_BITS-1:0];
    wire [`CS_TAG_SEL_BITS-1:0] line_tag = `CS_LINE_TAG_ADDR(line_addr);

    if (NUM_WAYS > 1)  begin
        reg [NUM_WAYS-1:0] repl_way;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset) begin
                repl_way <= 1;
            end else if (~stall) begin // hold the value on stalls prevent filling different slots twice
                repl_way <= {repl_way[NUM_WAYS-2:0], repl_way[NUM_WAYS-1]};
            end
        end        
        for (genvar i = 0; i < NUM_WAYS; ++i) begin
            assign way_sel[i] = fill && repl_way[i];
        end
    end else begin
        `UNUSED_VAR (stall)
        assign way_sel = fill;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin
        wire [`CS_TAG_SEL_BITS-1:0] read_tag;
        wire read_valid;

        VX_sp_ram #(
            .DATAW (TAG_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .NO_RWCHECK (1)
        ) tag_store (
            .clk   (clk),
            .read  (1'b1),
            .write (way_sel[i] || init),
            `UNUSED_PIN (wren),                
            .addr  (line_sel),
            .wdata ({~init, line_tag}), 
            .rdata ({read_valid, read_tag})
        );
        
        assign tag_matches[i] = read_valid && (line_tag == read_tag);
    end
    
`ifdef DBG_TRACE_CACHE_TAG
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s-bank%0d tag-fill: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, line_tag));
        end
        if (init) begin
            `TRACE(3, ("%d: %s-bank%0d tag-init: addr=0x%0h, blk_addr=%0d\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), line_sel));
        end
        if (lookup && ~stall) begin
            if (tag_matches != 0) begin
                `TRACE(3, ("%d: %s-bank%0d tag-hit: addr=0x%0h, way=%b, blk_addr=%0d, tag_id=0x%0h (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, line_tag, req_uuid));
            end else begin
                `TRACE(3, ("%d: %s-bank%0d tag-miss: addr=0x%0h, blk_addr=%0d, tag_id=0x%0h, (#%0d)\n", $time, INSTANCE_ID, BANK_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), line_sel, line_tag, req_uuid));
            end
        end          
    end    
`endif

endmodule
