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

module VX_tcu_tet_register #(
    parameter DATAW = 1,
    parameter DEPTH = 1
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    input  wire [DATAW-1:0] data_in,
    output wire [DATAW-1:0] data_out
);
    (* shreg_extract = "no" *) reg [DEPTH-1:0][DATAW-1:0] pipe;

    always_ff @(posedge clk) begin
        if (reset) begin
            pipe <= '0;
        end else if (enable) begin
            pipe[0] <= data_in;
            for (int i = 1; i < DEPTH; ++i) begin
                pipe[i] <= pipe[i-1];
            end
        end
    end

    assign data_out = pipe[DEPTH-1];

endmodule
