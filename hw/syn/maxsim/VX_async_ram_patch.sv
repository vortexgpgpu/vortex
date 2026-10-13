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

// The maxsim flow's VX_async_ram_patch: the RAM it stands for, read
// asynchronously.
//
// On Vivado, hw/rtl/libs/VX_async_ram_patch realizes an asynchronous-read RAM
// on a synchronous BRAM through placeholder cells that a post-synthesis hook
// rewires; with or without RADDR_REG it reads the word at raddr. The maxsim
// flow threads its memories itself and cannot run that hook, so it replaces the
// staged copy of the module with this one: same parameters and ports, same
// contents and reads, and the block RAM request as a synthesis attribute.
// Plain SystemVerilog, as the staged sources are already preprocessed.
module VX_async_ram_patch #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter WRENW       = 1,
    parameter DUAL_PORT   = 0,
    parameter FORCE_BRAM  = 0,
    parameter RADDR_REG   = 0,
    parameter RADDR_RESET = 0,
    parameter WRITE_FIRST = 0,
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0,
    parameter ADDRW       = (((SIZE) > 1) ? $clog2(SIZE) : 1)
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

    // VX_dp_ram and VX_sp_ram instantiate this module only for a RAM that
    // must be block RAM; the compiler makes its read synchronous
    // (seq-retime-firmem-reads), as the attribute asks.
    (* ram_style = "block" *) reg [DATAW-1:0] ram [0:SIZE-1];

    if (INIT_ENABLE != 0) begin : g_init
        if (INIT_FILE != "") begin : g_file
            initial $readmemh(INIT_FILE, ram);
        end else begin : g_value
            initial begin
                for (integer i = 0; i < SIZE; ++i) begin
                    ram[i] = INIT_VALUE;
                end
            end
        end
    end

    always @(posedge clk) begin
        if (write) begin
            for (integer i = 0; i < WRENW; ++i) begin
                if (wren[i]) begin
                    ram[waddr][i * WSELW +: WSELW] <= wdata[i * WSELW +: WSELW];
                end
            end
        end
    end

    assign rdata = ram[raddr];

endmodule
