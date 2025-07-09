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
module VX_pipe_register #(
    parameter DATAW  = 1,
    parameter RESETW = 0,
    parameter DEPTH  = 1,
    parameter [`UP(RESETW)-1:0] INIT_VALUE = {`UP(RESETW){1'b0}}
) (
    input wire              clk,
    input wire              reset,
    input wire              enable,
    input wire [DATAW-1:0]  data_in,
    output wire [DATAW-1:0] data_out
);
    VX_shift_register #(
        .DATAW      (DATAW),
        .RESETW     (RESETW),
        .DEPTH      (DEPTH),
        .INIT_VALUE (INIT_VALUE)
    ) g_shift_register (
        .clk       (clk),
        .reset     (reset),
        .enable    (enable),
        .data_in   (data_in),
        .data_out  (data_out)
    );

endmodule
`TRACING_ON
