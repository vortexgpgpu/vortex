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

// Integrated clock gate, ASIC only: the PDK's ICG cell (VX_clockgate_asic,
// mapped by hw/syn/libs/icg/<pdk>.v). clk_out follows clk_in while en is high
// and stays low otherwise; en is latched while clk_in is low, so no glitch.

`TRACING_OFF

module VX_clockgate (
    input  wire clk_in,
    input  wire en,
    output wire clk_out
);

`ifdef ASIC
    VX_clockgate_asic icg (
        .clk_in  (clk_in),
        .en      (en),
        .clk_out (clk_out)
    );
`else
    `STATIC_ASSERT(0, ("VX_clockgate is ASIC-only"))
    `UNUSED_VAR (clk_in)
    `UNUSED_VAR (en)
    assign clk_out = clk_in;
`endif

endmodule

`TRACING_ON
