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

`include "VX_platform.vh"

`TRACING_OFF
module VX_allocator #(
    parameter SIZE  = 1,
    parameter ADDRW = `LOG2UP(SIZE)
) (
    input  wire             clk,
    input  wire             reset,

    input  wire             acquire_en,
    output wire [ADDRW-1:0] acquire_addr,

    input  wire             release_en,
    input  wire [ADDRW-1:0] release_addr,

    output wire             empty,
    output wire             full
);
    reg [SIZE-1:0] free_slots, free_slots_n;
    reg [ADDRW-1:0] acquire_addr_r;
    reg empty_r, full_r;
    wire [ADDRW-1:0] free_index;
    wire free_valid;

    // The next slot is the lowest free one after this cycle's release and
    // acquire. Both candidates without the release (with and without this
    // cycle's acquire) are encoded from the registered state alone, and the
    // release is compared with each; the acquire then selects, so the late
    // enables reach the next slot through a select, not through the priority
    // encoder or a compare.
    wire [SIZE-1:0] acquire_mask = SIZE'(1) << acquire_addr_r;
    wire [SIZE-1:0] free_hold    = free_slots;
    wire [SIZE-1:0] free_acq     = free_slots & ~acquire_mask;

    wire [ADDRW-1:0] hold_index, acq_index;
    wire hold_valid, acq_valid;

    VX_priority_encoder #(
        .N (SIZE)
    ) free_slots_sel (
        .data_in   (free_hold),
        .index_out (hold_index),
        .valid_out (hold_valid),
        `UNUSED_PIN (onehot_out)
    );

    VX_priority_encoder #(
        .N (SIZE)
    ) free_slots_acq_sel (
        .data_in   (free_acq),
        .index_out (acq_index),
        .valid_out (acq_valid),
        `UNUSED_PIN (onehot_out)
    );

    // whether the released slot is the lowest, against either candidate
    wire take_rel_acq  = release_en && (~acq_valid  || (release_addr < acq_index));
    wire take_rel_hold = release_en && (~hold_valid || (release_addr < hold_index));

    assign free_index = acquire_en ? (take_rel_acq  ? release_addr : acq_index)
                                   : (take_rel_hold ? release_addr : hold_index);
    assign free_valid = (acquire_en ? acq_valid : hold_valid) || release_en;

    // An acquire takes a free slot, so the set is then never all free.
    wire [SIZE-1:0] free_rel = free_slots | (release_en ? (SIZE'(1) << release_addr) : '0);
    wire empty_n = (& free_rel) && ~acquire_en;

    always @(*) begin
        free_slots_n = free_slots;
        if (release_en) begin
            free_slots_n[release_addr] = 1;
        end
        if (acquire_en) begin
            free_slots_n[acquire_addr_r] = 0;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            acquire_addr_r <= ADDRW'(1'b0);
            free_slots     <= {SIZE{1'b1}};
            empty_r        <= 1'b1;
            full_r         <= 1'b0;
        end else begin
            if (release_en) begin
                `ASSERT(0 == free_slots[release_addr], ("releasing invalid addr %d", release_addr));
            end
            if (acquire_en) begin
                `ASSERT(~full_r, ("allocator is full"));
            end

            if (acquire_en || (release_en && full_r)) begin
                acquire_addr_r <= free_index;
            end

            free_slots <= free_slots_n;
            empty_r    <= empty_n;
            full_r     <= ~free_valid;
        end
    end

    assign acquire_addr = acquire_addr_r;
    assign empty        = empty_r;
    assign full         = full_r;

endmodule
`TRACING_ON
