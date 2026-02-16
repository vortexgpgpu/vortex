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
    parameter META_BLOCK_WIDTH = 64,
    parameter PER_WARP_DEPTH = 4
) (
    input wire          clk,
    input wire          reset,

    // Read port (from FEDP path)
    input wire [`LOG2UP(`NUM_WARPS)-1:0] raddr_wid,
    input wire [3:0]    step_m,
    input wire [3:0]    step_k,
    output wire [META_BLOCK_WIDTH-1:0] vld_meta_block,

    // Write port (meta_store instruction)
    input wire          wr_en,
    input wire [`LOG2UP(`NUM_WARPS)-1:0] wr_wid,
    input wire [3:0]    wr_col_idx,
    input wire [PER_WARP_DEPTH-1:0][31:0] wr_data
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // Local parameters
    localparam HALF_K_STEPS = TCU_K_STEPS / 2;
    localparam TOTAL_DEPTH  = `NUM_WARPS * PER_WARP_DEPTH;
    localparam ADDRW        = `CLOG2(TOTAL_DEPTH);
    localparam ADDRW_PW     = `CLOG2(PER_WARP_DEPTH);
    localparam M_STEP_BITS  = `CLOG2(TCU_M_STEPS);
    localparam K_STEP_BITS  = (HALF_K_STEPS > 1) ? `CLOG2(HALF_K_STEPS) : 0;
    localparam NUM_COLS     = META_BLOCK_WIDTH / 32;

    // Metadata register array (per-warp partitioned)
    reg [META_BLOCK_WIDTH-1:0] meta_mem [0:TOTAL_DEPTH-1];

    // Read address: {wid, step_m, step_k} 
    wire [ADDRW_PW-1:0] per_warp_raddr;
    if (K_STEP_BITS > 0) begin : g_addr_mk
        assign per_warp_raddr = {step_m[M_STEP_BITS-1:0], step_k[K_STEP_BITS-1:0]};
    end else begin : g_addr_m
        assign per_warp_raddr = step_m[M_STEP_BITS-1:0];
    end
    wire [ADDRW-1:0] read_addr = {raddr_wid, per_warp_raddr};

    // Combinational read
    assign vld_meta_block = meta_mem[read_addr];

    // Post-reset init counter: fills all warps with alternating patterns
    reg [ADDRW:0] init_counter;
    wire init_active = ~init_counter[ADDRW];
    wire [ADDRW-1:0] init_addr = init_counter[ADDRW-1:0];
    wire [META_BLOCK_WIDTH-1:0] init_data = init_addr[0] ?
        {(META_BLOCK_WIDTH/4){4'b1010}} :
        {(META_BLOCK_WIDTH/4){4'b0101}};

    // Write logic: init or runtime meta_store
    always_ff @(posedge clk) begin
        if (reset) begin
            init_counter <= 0;
        end else if (init_active) begin
            meta_mem[init_addr] <= init_data;
            init_counter <= init_counter + 1;
        end else if (wr_en) begin
            for (int row = 0; row < PER_WARP_DEPTH; row++) begin
                for (int col = 0; col < NUM_COLS; col++) begin
                    if (col == int'(wr_col_idx)) begin
                        meta_mem[{wr_wid, ADDRW_PW'(row)}][col*32 +: 32] <= wr_data[row];
                    end
                end
            end
        end
    end

endmodule

/* verilator lint_on UNUSEDSIGNAL */
