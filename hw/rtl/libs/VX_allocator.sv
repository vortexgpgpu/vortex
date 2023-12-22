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

    always @(*) begin
        free_slots_n = free_slots;
        if (release_en) begin
            free_slots_n[release_addr] = 1;                
        end
        if (acquire_en) begin
            free_slots_n[acquire_addr_r] = 0;
        end            
    end

    VX_lzc #(
        .N (SIZE),
        .REVERSE (1)
    ) free_slots_sel (
        .data_in   (free_slots_n),
        .data_out  (free_index),
        .valid_out (free_valid)
    );  

    always @(posedge clk) begin
        if (reset) begin
            acquire_addr_r <= ADDRW'(1'b0);
            free_slots     <= {SIZE{1'b1}};
            empty_r        <= 1'b1;
            full_r         <= 1'b0;            
        end else begin
            if (release_en) begin
                `ASSERT(0 == free_slots[release_addr], ("%t: releasing invalid addr %d", $time, release_addr));
            end
            if (acquire_en) begin                
                `ASSERT(~full_r, ("%t: allocator is full", $time));
            end            
            
            if (acquire_en || (release_en && full_r)) begin
                acquire_addr_r <= free_index;
            end

            free_slots <= free_slots_n;           
            empty_r    <= (& free_slots_n);
            full_r     <= ~free_valid;
        end        
    end
        
    assign acquire_addr = acquire_addr_r;
    assign empty        = empty_r;
    assign full         = full_r;
    
endmodule
`TRACING_ON
