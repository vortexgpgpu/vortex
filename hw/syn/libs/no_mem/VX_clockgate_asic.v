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

// Behavioral model of the ICG cell; synthesis maps it to the PDK cell
// (hw/syn/libs/icg/<pdk>.v).
(* blackbox, keep_hierarchy = "yes" *)
module VX_clockgate_asic (
    input  wire clk_in,
    input  wire en,
    output wire clk_out
);
    reg en_latched;
    always @(*) begin
        if (~clk_in)
            en_latched = en;
    end
    assign clk_out = clk_in & en_latched;
endmodule
