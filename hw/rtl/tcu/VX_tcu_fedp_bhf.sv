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

module VX_tcu_fedp_bhf #(
    parameter LATENCY = 1,
    parameter N = 2
) (
    input  wire clk,
    input  wire reset,
    input  wire enable,

    input  wire[2:0] fmt_s,
    input  wire[2:0] fmt_d,

    input  wire [N-1:0][`XLEN-1:0] a_row,
    input  wire [N-1:0][`XLEN-1:0] b_col,
    input  wire [`XLEN-1:0] c_val,
    output wire [`XLEN-1:0] d_val
);
    `UNUSED_VAR(reset);
    `UNUSED_VAR(fmt_s);
    `UNUSED_VAR(fmt_d);
    `UNUSED_VAR ({a_row, b_col, c_val});

    // TODO: use Berkeley HardFloat mulRecFNToRaw, addRecFNToRaw, roundRawFNToRecFN
    `UNUSED_PARAM (LATENCY);
    `UNUSED_PARAM (N);
    `UNUSED_VAR (clk);
    `UNUSED_VAR (enable);
    assign d_val = '0;

endmodule
