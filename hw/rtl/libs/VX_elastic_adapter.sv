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
module VX_elastic_adapter (
    input wire  clk,
    input wire  reset,

    input wire  valid_in,
    output wire ready_in,
        
    input wire  ready_out,
    output wire valid_out,

    input wire  busy,
    output wire strobe
);
    wire push = valid_in && ready_in;
    wire pop = valid_out && ready_out;

    reg loaded;

    always @(posedge clk) begin
        if (reset) begin
            loaded  <= 0;
        end else begin
            if (push) begin
                loaded <= 1;
            end
            if (pop) begin                
                loaded <= 0;
            end
        end
    end

    assign ready_in  = ~loaded;
    assign valid_out = loaded && ~busy;
    assign strobe    = push;

endmodule
`TRACING_ON
