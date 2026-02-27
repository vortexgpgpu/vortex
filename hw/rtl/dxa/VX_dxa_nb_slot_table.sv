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

module VX_dxa_nb_slot_table #(
    parameter MAX_OUTSTANDING = 4,
    parameter MEM_ADDR_WIDTH  = 32,
    parameter GMEM_ADDR_WIDTH = 26,
    parameter GMEM_OFF_BITS   = 6
) (
    input  wire clk,
    input  wire reset,

    input  wire [GMEM_ADDR_WIDTH-1:0] q_line_addr,
    output reg                        q_inflight_found,
    output reg                        q_free_found,
    output reg [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] q_free_slot,

    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] rsp_slot,
    output wire                       rsp_slot_busy,
    output wire [MEM_ADDR_WIDTH-1:0] rsp_smem_byte_addr,
    output wire [GMEM_ADDR_WIDTH-1:0] rsp_gmem_line_addr,
    output wire [GMEM_OFF_BITS-1:0]  rsp_gmem_off,
    output wire                       rsp_is_last,

    input  wire                        alloc_fire,
    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] alloc_slot,
    input  wire [MEM_ADDR_WIDTH-1:0]   alloc_smem_byte_addr,
    input  wire [GMEM_ADDR_WIDTH-1:0]  alloc_gmem_line_addr,
    input  wire [GMEM_OFF_BITS-1:0]    alloc_gmem_off,
    input  wire                        alloc_is_last,

    input  wire                        release_fire,
    input  wire [`UP(`CLOG2(MAX_OUTSTANDING))-1:0] release_slot
);
    localparam RD_SLOT_BITS = `CLOG2(MAX_OUTSTANDING);
    localparam RD_SLOT_W    = `UP(RD_SLOT_BITS);

    `STATIC_ASSERT(`IS_POW2(MAX_OUTSTANDING), ("MAX_OUTSTANDING must be power of 2"))

    reg [MAX_OUTSTANDING-1:0] busy_r;
    reg [MAX_OUTSTANDING-1:0][MEM_ADDR_WIDTH-1:0] smem_byte_addr_r;
    reg [MAX_OUTSTANDING-1:0][GMEM_ADDR_WIDTH-1:0] gmem_line_addr_r;
    reg [MAX_OUTSTANDING-1:0][GMEM_OFF_BITS-1:0] gmem_off_r;
    reg [MAX_OUTSTANDING-1:0] is_last_r;

    wire [MAX_OUTSTANDING-1:0] busy_eff_w;
    for (genvar i = 0; i < MAX_OUTSTANDING; ++i) begin : g_busy_eff
        assign busy_eff_w[i] = busy_r[i] && ~(release_fire && (release_slot == RD_SLOT_W'(i)));
    end

    always @(*) begin
        q_free_found = 1'b0;
        q_free_slot  = '0;
        for (integer i = 0; i < MAX_OUTSTANDING; ++i) begin
            if (~busy_eff_w[i] && ~q_free_found) begin
                q_free_found = 1'b1;
                q_free_slot  = RD_SLOT_W'(i);
            end
        end
    end

    always @(*) begin
        q_inflight_found = 1'b0;
        for (integer i = 0; i < MAX_OUTSTANDING; ++i) begin
            if (busy_eff_w[i] && (gmem_line_addr_r[i] == q_line_addr)) begin
                q_inflight_found = 1'b1;
            end
        end
    end

    assign rsp_slot_busy      = busy_r[rsp_slot];
    assign rsp_smem_byte_addr = smem_byte_addr_r[rsp_slot];
    assign rsp_gmem_line_addr = gmem_line_addr_r[rsp_slot];
    assign rsp_gmem_off       = gmem_off_r[rsp_slot];
    assign rsp_is_last        = is_last_r[rsp_slot];

    always @(posedge clk) begin
        if (reset) begin
            busy_r <= '0;
            smem_byte_addr_r <= '0;
            gmem_line_addr_r <= '0;
            gmem_off_r <= '0;
            is_last_r <= '0;
        end else begin
            if (release_fire) begin
                busy_r[release_slot] <= 1'b0;
            end
            if (alloc_fire) begin
                busy_r[alloc_slot] <= 1'b1;
                smem_byte_addr_r[alloc_slot] <= alloc_smem_byte_addr;
                gmem_line_addr_r[alloc_slot] <= alloc_gmem_line_addr;
                gmem_off_r[alloc_slot] <= alloc_gmem_off;
                is_last_r[alloc_slot] <= alloc_is_last;
            end
        end
    end
endmodule

