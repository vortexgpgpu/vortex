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

// this is an implementation of a pipelined multi-banked cache
// we allocate a free slot from the MSHR before processing a core request
// and release the slot when we get a cache hit.
// during a memory fill response we initiate the replay sequence 
// and dequeue all associated pending entries.

// Warning: This MSHR implementation is strongly coupled with the bank pipeline
// and as such changes to either module requires careful evaluation.
// This implementation makes the following assumptions:
// (1) two-cycle pipeline: st0 and st1.
// (2) core request flow: st0: allocate / lookup, st1: finalize.
// (3) the first dequeue after the fill should happen in st0, when the fill is in st1
//     this is enforced inside the bank by "rdw_hazard_st0".

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
    parameter MSHR_ADDR_WIDTH   = `LOG2UP(MSHR_SIZE)
) (
    input wire clk,
    input wire reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     deq_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     lkp_req_uuid,
    input wire[`UP(UUID_WIDTH)-1:0]     fin_req_uuid,
`IGNORE_UNUSED_END

    // allocate
    input wire                          allocate_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] allocate_addr,
    input wire                          allocate_rw,
    input wire [DATA_WIDTH-1:0]         allocate_data,
    output wire [MSHR_ADDR_WIDTH-1:0]   allocate_id,
    output wire [MSHR_ADDR_WIDTH-1:0]   allocate_tail,
    output wire                         allocate_ready,

    // lookup
    input wire                          lookup_valid,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] lookup_addr,
    output wire [MSHR_SIZE-1:0]         lookup_matches,

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

    // finalize
    input wire                          finalize_valid,
    input wire                          finalize_release,
    input wire                          finalize_pending,
    input wire [MSHR_ADDR_WIDTH-1:0]    finalize_id,
    input wire [MSHR_ADDR_WIDTH-1:0]    finalize_tail
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

    wire [MSHR_ADDR_WIDTH-1:0] tail_idx;
    
    wire allocate_fire = allocate_valid && allocate_ready;
    wire dequeue_fire = dequeue_valid && dequeue_ready;

    wire [MSHR_SIZE-1:0] addr_matches;
    for (genvar i = 0; i < MSHR_SIZE; ++i) begin
        assign addr_matches[i] = valid_table[i] && (addr_table[i] == lookup_addr);
    end

    VX_lzc #(
        .N (MSHR_SIZE),
        .REVERSE (1)
    ) allocate_sel (
        .data_in   (~valid_table_n),
        .data_out  (allocate_id_n),
        .valid_out (allocate_rdy_n)
    );

    VX_onehot_encoder #(
        .N (MSHR_SIZE)
    ) tail_sel (
        .data_in (addr_matches & ~next_table_x),
        .data_out (tail_idx),
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
            end else begin
                dequeue_val_n = 0;
            end
        end

        if (finalize_valid) begin
            if (finalize_release) begin
                valid_table_n[finalize_id] = 0;
            end
            if (finalize_pending) begin
                next_table_x[finalize_tail] = 1;
            end
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
        end else begin
            valid_table  <= valid_table_n;
            allocate_rdy <= allocate_rdy_n;
            dequeue_val  <= dequeue_val_n;
        end

        if (allocate_fire) begin
            addr_table[allocate_id]  <= allocate_addr;
            write_table[allocate_id] <= allocate_rw;
        end

        if (finalize_valid && finalize_pending) begin
            next_index[finalize_tail] <= finalize_id;
        end

        dequeue_id_r  <= dequeue_id_n;
        allocate_id_r <= allocate_id_n;
        next_table    <= next_table_n;
    end
    
    `RUNTIME_ASSERT((~allocate_fire || ~valid_table[allocate_id_r]), ("%t: *** %s-bank%0d inuse allocation: addr=0x%0h, id=%0d (#%0d)", $time, INSTANCE_ID, BANK_ID, 
        `CS_LINE_TO_FULL_ADDR(allocate_addr, BANK_ID), allocate_id_r, lkp_req_uuid))

    `RUNTIME_ASSERT((~finalize_valid || valid_table[finalize_id]), ("%t: *** %s-bank%0d invalid release: addr=0x%0h, id=%0d (#%0d)", $time, INSTANCE_ID, BANK_ID, 
        `CS_LINE_TO_FULL_ADDR(addr_table[finalize_id], BANK_ID), finalize_id, fin_req_uuid))
    
    `RUNTIME_ASSERT((~fill_valid || valid_table[fill_id]), ("%t: *** %s-bank%0d invalid fill: addr=0x%0h, id=%0d", $time, INSTANCE_ID, BANK_ID, 
        `CS_LINE_TO_FULL_ADDR(addr_table[fill_id], BANK_ID), fill_id))

    VX_dp_ram #(
        .DATAW  (DATA_WIDTH),
        .SIZE   (MSHR_SIZE),
        .LUTRAM (1)
    ) entries (
        .clk   (clk),
        .read  (1'b1),
        .write (allocate_valid),
        `UNUSED_PIN (wren),               
        .waddr (allocate_id_r),     
        .wdata (allocate_data),
        .raddr (dequeue_id_r),
        .rdata (dequeue_data)
    );

    assign fill_addr = addr_table[fill_id];

    assign allocate_ready = allocate_rdy;
    assign allocate_id    = allocate_id_r;
    assign allocate_tail  = tail_idx;

    assign dequeue_valid  = dequeue_val;
    assign dequeue_addr   = addr_table[dequeue_id_r];
    assign dequeue_rw     = write_table[dequeue_id_r];
    assign dequeue_id     = dequeue_id_r;

    assign lookup_matches = addr_matches & ~write_table;

    `UNUSED_VAR (lookup_valid)

`ifdef DBG_TRACE_CACHE_MSHR        
    reg show_table;
    always @(posedge clk) begin
        if (reset) begin
            show_table <= 0;
        end else begin
            show_table <= allocate_fire || lookup_valid || finalize_valid || fill_valid || dequeue_fire;
        end
        if (allocate_fire)
            `TRACE(3, ("%d: %s-bank%0d mshr-allocate: addr=0x%0h, tail=%0d, id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID,
                `CS_LINE_TO_FULL_ADDR(allocate_addr, BANK_ID), allocate_tail, allocate_id, lkp_req_uuid));
        if (lookup_valid)
            `TRACE(3, ("%d: %s-bank%0d mshr-lookup: addr=0x%0h, matches=%b (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                `CS_LINE_TO_FULL_ADDR(lookup_addr, BANK_ID), lookup_matches, lkp_req_uuid));
        if (finalize_valid)
            `TRACE(3, ("%d: %s-bank%0d mshr-finalize release=%b, pending=%b, tail=%0d, id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                finalize_release, finalize_pending, finalize_tail, finalize_id, fin_req_uuid));
        if (fill_valid)
            `TRACE(3, ("%d: %s-bank%0d mshr-fill: addr=0x%0h, addr=0x%0h, id=%0d\n", $time, INSTANCE_ID, BANK_ID, 
                `CS_LINE_TO_FULL_ADDR(addr_table[fill_id], BANK_ID), `CS_LINE_TO_FULL_ADDR(fill_addr, BANK_ID), fill_id));
        if (dequeue_fire)
            `TRACE(3, ("%d: %s-bank%0d mshr-dequeue: addr=0x%0h, id=%0d (#%0d)\n", $time, INSTANCE_ID, BANK_ID, 
                `CS_LINE_TO_FULL_ADDR(dequeue_addr, BANK_ID), dequeue_id_r, deq_req_uuid));
        if (show_table) begin
            `TRACE(3, ("%d: %s-bank%0d mshr-table", $time, INSTANCE_ID, BANK_ID));
            for (integer i = 0; i < MSHR_SIZE; ++i) begin
                if (valid_table[i]) begin
                    `TRACE(3, (" %0d=0x%0h", i, `CS_LINE_TO_FULL_ADDR(addr_table[i], BANK_ID)));
                    if (write_table[i])
                        `TRACE(3, ("(w)"));
                    else
                        `TRACE(3, ("(r)"));
                    if (next_table[i])
                        `TRACE(3, ("->%0d", next_index[i]));
                end
            end
            `TRACE(3, ("\n"));
        end        
    end
`endif

endmodule
