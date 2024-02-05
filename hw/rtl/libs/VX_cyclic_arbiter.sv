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
module VX_cyclic_arbiter #(
    parameter NUM_REQS     = 1,
    parameter LOCK_ENABLE  = 0,
    parameter LOG_NUM_REQS = `LOG2UP(NUM_REQS)
) (
    input  wire                     clk,
    input  wire                     reset,
    input  wire [NUM_REQS-1:0]      requests,    
    output wire [LOG_NUM_REQS-1:0]  grant_index,
    output wire [NUM_REQS-1:0]      grant_onehot,   
    output wire                     grant_valid,
    input  wire                     grant_unlock
);
    if (NUM_REQS == 1)  begin  

        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)      
        
        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin

        localparam IS_POW2 = (1 << LOG_NUM_REQS) == NUM_REQS;

        reg [LOG_NUM_REQS-1:0] grant_index_r;

        always @(posedge clk) begin
            if (reset) begin
                grant_index_r <= '0;
            end else begin                
                if (!IS_POW2 && grant_index_r == LOG_NUM_REQS'(NUM_REQS-1)) begin
                    grant_index_r <= '0;
                end else if (!LOCK_ENABLE || ~grant_valid || grant_unlock) begin
                    grant_index_r <= grant_index_r + LOG_NUM_REQS'(1);
                end
            end
        end

        reg [NUM_REQS-1:0] grant_onehot_r;
        always @(*) begin
            grant_onehot_r = '0;
            grant_onehot_r[grant_index_r] = 1'b1;
        end

        assign grant_index  = grant_index_r;    
        assign grant_onehot = grant_onehot_r;
        assign grant_valid  = requests[grant_index_r];

    end
    
endmodule
`TRACING_ON
