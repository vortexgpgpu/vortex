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
module VX_bypass_buffer #(
    parameter DATAW    = 1,
    parameter PASSTHRU = 0
) ( 
    input  wire             clk,
    input  wire             reset,
    input  wire             valid_in,
    output wire             ready_in,        
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out,
    input  wire             ready_out,
    output wire             valid_out
); 
    if (PASSTHRU != 0) begin
        `UNUSED_VAR (clk)
        `UNUSED_VAR (reset)
        assign ready_in  = ready_out;
        assign valid_out = valid_in;        
        assign data_out  = data_in;
    end else begin
        reg [DATAW-1:0] buffer;
        reg buffer_valid;

        always @(posedge clk) begin
            if (reset) begin
                buffer_valid <= 0;
            end else begin            
                if (ready_out) begin
                    buffer_valid <= 0;
                end
                if (valid_in && ~ready_out) begin
                    `ASSERT(!buffer_valid, ("runtime error"));
                    buffer_valid <= 1;
                end
            end

            if (valid_in && ~ready_out) begin
                buffer <= data_in;
            end
        end

        assign ready_in  = ready_out || !buffer_valid;
        assign data_out  = buffer_valid ? buffer : data_in;
        assign valid_out = valid_in || buffer_valid;
    end

endmodule
`TRACING_ON