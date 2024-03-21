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

module VX_operands import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0
) (
    input wire              clk,
    input wire              reset,

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_scoreboard_if.slave  scoreboard_if [`ISSUE_WIDTH],
    VX_operands_if.master   operands_if [`ISSUE_WIDTH]
);

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin        
        
        `RESET_RELAY (slice_reset, reset);

        VX_gpr_slice #(
            .CORE_ID (CORE_ID)
        ) gpr_slice (
            .clk          (clk),
            .reset        (slice_reset),
            .writeback_if (writeback_if[i]),
            .scoreboard_if(scoreboard_if[i]),
            .operands_if  (operands_if[i])
        );
    end

endmodule
