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

(* blackbox, keep_hierarchy = "yes" *)
module VX_dp_ram_asic #(
    parameter DATAW = 1,
    parameter SIZE  = 1,
    parameter WRENW = 1,
    parameter ADDRW = $clog2(SIZE)
) (
    input wire               clk,
    input wire               reset,
    input wire               read,
    input wire               write,
    input wire [WRENW-1:0]   wren,
    input wire [ADDRW-1:0]   waddr,
    input wire [DATAW-1:0]   wdata,
    input wire [ADDRW-1:0]   raddr,
    output wire [DATAW-1:0]  rdata
);
    localparam WSELW = DATAW / WRENW;

    reg [DATAW-1:0] mem [0:SIZE-1];
    reg [DATAW-1:0] rdata_reg;

    always @(posedge clk) begin
        if (reset) begin
            rdata_reg <= '0;
        end else if (read) begin
            rdata_reg <= mem[raddr];
        end
    end

    always @(posedge clk) begin
        if (write) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i]) begin
                    mem[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
            end
        end
    end

    assign rdata = rdata_reg;

endmodule
