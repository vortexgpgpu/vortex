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

module VX_cache_init #(
    // Size of cache in bytes
    parameter CACHE_SIZE    = 1024, 
    // Size of line inside a bank in bytes
    parameter LINE_SIZE     = 16, 
    // Number of banks
    parameter NUM_BANKS     = 1,
    // Number of associative ways
    parameter NUM_WAYS      = 1
) (
    input  wire clk,
    input  wire reset,    
    output wire [`CS_LINE_SEL_BITS-1:0] addr_out,
    output wire valid_out
);
    reg enabled;
    reg [`CS_LINE_SEL_BITS-1:0] line_ctr;

    always @(posedge clk) begin
        if (reset) begin
            enabled  <= 1;
            line_ctr <= '0;
        end else begin
            if (enabled) begin
                if (line_ctr == ((2 ** `CS_LINE_SEL_BITS)-1)) begin
                    enabled <= 0;
                end
                line_ctr <= line_ctr + `CS_LINE_SEL_BITS'(1);           
            end
        end
    end

    assign addr_out  = line_ctr;
    assign valid_out = enabled;

endmodule
