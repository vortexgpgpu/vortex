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
module VX_fair_arbiter #(
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
        `UNUSED_VAR (grant_unlock)

        assign grant_index  = '0;
        assign grant_onehot = requests;
        assign grant_valid  = requests[0];

    end else begin    

        reg [NUM_REQS-1:0] buffer;

        wire [NUM_REQS-1:0] buffer_qual   = buffer & requests;
        wire [NUM_REQS-1:0] requests_qual = (| buffer) ? buffer_qual : requests;
        wire [NUM_REQS-1:0] buffer_n      = requests_qual & ~grant_onehot;

        always @(posedge clk) begin
            if (reset) begin
                buffer <= '0;
            end else if (!LOCK_ENABLE || grant_unlock) begin
                buffer <= buffer_n;
            end
        end
               
        VX_priority_arbiter #(
            .NUM_REQS (NUM_REQS)
        ) priority_arbiter (
            .requests     (requests_qual), 
            .grant_index  (grant_index),
            .grant_onehot (grant_onehot),
            .grant_valid  (grant_valid)
        );

    end
    
endmodule
`TRACING_ON
