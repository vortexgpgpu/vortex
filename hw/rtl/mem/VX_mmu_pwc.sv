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

`include "VX_define.vh"

// Page-walk cache: direct-mapped cache of non-leaf page-table entries, keyed
// by {parent table PPN, child VPN slice} and holding the child table PPN.
// The lookup is combinational so a walk can skip a level in its first cycle;
// fills are registered. Entries are only ever invalidated on reset or flush.
module VX_mmu_pwc #(
    parameter KEY_WIDTH   = 30,
    parameter DATA_WIDTH  = 20,
    parameter NUM_ENTRIES = 64
) (
    input wire                  clk,
    input wire                  reset,
    input wire                  flush,

    input  wire [KEY_WIDTH-1:0]  lookup_key,
    output wire                  lookup_hit,
    output wire [DATA_WIDTH-1:0] lookup_data,

    input wire                  fill_valid,
    input wire [KEY_WIDTH-1:0]  fill_key,
    input wire [DATA_WIDTH-1:0] fill_data
);
    `STATIC_ASSERT(`IS_POW2(NUM_ENTRIES), ("NUM_ENTRIES must be a power of 2"))

    localparam IDX_BITS = `CLOG2(NUM_ENTRIES);
    localparam TAG_BITS = KEY_WIDTH - IDX_BITS;

    reg [NUM_ENTRIES-1:0]           valid_r;
    reg [TAG_BITS-1:0]              tags_r [NUM_ENTRIES];
    reg [DATA_WIDTH-1:0]            data_r [NUM_ENTRIES];

    wire [IDX_BITS-1:0] lookup_idx = lookup_key[IDX_BITS-1:0];
    wire [TAG_BITS-1:0] lookup_tag = lookup_key[KEY_WIDTH-1:IDX_BITS];
    wire [IDX_BITS-1:0] fill_idx   = fill_key[IDX_BITS-1:0];
    wire [TAG_BITS-1:0] fill_tag   = fill_key[KEY_WIDTH-1:IDX_BITS];

    assign lookup_hit  = valid_r[lookup_idx] && (tags_r[lookup_idx] == lookup_tag);
    assign lookup_data = data_r[lookup_idx];

    always @(posedge clk) begin
        if (reset || flush) begin
            valid_r <= '0;
        end else if (fill_valid) begin
            valid_r[fill_idx] <= 1'b1;
        end
    end

    always @(posedge clk) begin
        if (fill_valid) begin
            tags_r[fill_idx] <= fill_tag;
            data_r[fill_idx] <= fill_data;
        end
    end

endmodule
