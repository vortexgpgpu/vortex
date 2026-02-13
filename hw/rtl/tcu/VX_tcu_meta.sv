// Copyright 2019-2023
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

/* verilator lint_off UNUSEDSIGNAL */

module VX_tcu_meta import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter META_BLOCK_WIDTH = 64  // Default: TCU_NT * 2 * I_RATIO
) (
    input wire          clk,
    input wire          reset,

    // Step indices (from VX_tcu_core)
    input wire [3:0]    step_m,
    input wire [3:0]    step_k,

    // Output (combinational)
    output wire [META_BLOCK_WIDTH-1:0] vld_meta_block
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Local parameters
    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam DEPTH = TCU_M_STEPS * HALF_K_STEPS;
    localparam ADDRW = `CLOG2(DEPTH);
    localparam M_STEP_BITS = `CLOG2(TCU_M_STEPS);   // Bits needed for step_m index
    localparam K_STEP_BITS = (HALF_K_STEPS > 1) ? `CLOG2(HALF_K_STEPS) : 0;

    // Read address: {step_m, step_k} — step_k omitted when K_STEP_BITS=0 (B_SPLIT)
    wire [ADDRW-1:0] read_addr;
    if (K_STEP_BITS > 0) begin : g_addr_mk
        assign read_addr = {step_m[M_STEP_BITS-1:0], step_k[K_STEP_BITS-1:0]};
    end else begin : g_addr_m
        assign read_addr = step_m[M_STEP_BITS-1:0];
    end

    // Post-reset init: even addr → 0101, odd addr → 1010
    reg [ADDRW:0] init_counter;
    wire init_active = ~init_counter[ADDRW];
    wire [ADDRW-1:0] init_addr = init_counter[ADDRW-1:0];
    wire [META_BLOCK_WIDTH-1:0] init_data = init_addr[0] ?
        {(META_BLOCK_WIDTH/4){4'b1010}} :
        {(META_BLOCK_WIDTH/4){4'b0101}};

    always_ff @(posedge clk) begin
        if (reset) begin
            init_counter <= 0;
        end else if (init_active) begin
            init_counter <= init_counter + 1;
        end
    end

    // Metadata SRAM (combinational read)
    VX_dp_ram #(
        .DATAW       (META_BLOCK_WIDTH),
        .SIZE        (DEPTH),
        .WRENW       (1),
        .OUT_REG     (0),   // Combinational read: output same cycle as address
        .RDW_MODE    ("R"),
        .INIT_ENABLE (0)
    ) meta_store (
        .clk   (clk),
        .reset (1'b0),
        .read  (1'b1),      // Always enabled (combinational)
        .write (init_active),
        .wren  (1'b1),
        .waddr (init_addr),
        .wdata (init_data),
        .raddr (read_addr),
        .rdata (vld_meta_block)
    );

endmodule

/* verilator lint_on UNUSEDSIGNAL */
