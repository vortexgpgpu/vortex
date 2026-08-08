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

module VX_kmu_arb_top import VX_gpu_pkg::*; (
    input  wire       clk,
    input  wire       reset,

    input  wire       in_valid,
    output wire       in_ready,

    input  wire [1:0] out_ready,
    output wire [1:0] out_valid,

    output wire       pending
);

    VX_kmu_bus_if bus_in_if[1]();
    VX_kmu_bus_if bus_out_if[2]();

    assign bus_in_if[0].valid = in_valid;
    assign bus_in_if[0].data  = '0;
    assign in_ready = bus_in_if[0].ready;

    for (genvar i = 0; i < 2; ++i) begin : g_outputs
        assign bus_out_if[i].ready = out_ready[i];
        assign out_valid[i] = bus_out_if[i].valid;
        `UNUSED_VAR (bus_out_if[i].data)
    end

    VX_kmu_arb #(
        .NUM_INPUTS  (1),
        .NUM_OUTPUTS (2),
        .OUT_BUF     (3)
    ) dut (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (bus_in_if),
        .bus_out_if (bus_out_if),
        .pending    (pending)
    );

endmodule
