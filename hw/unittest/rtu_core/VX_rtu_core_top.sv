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

// Flat-port wrapper around VX_rtu_core for out-of-context synthesis. The RTU bus
// (arm / req / win) and the RTCache port are interfaces, which cannot cross the
// synthesis top, so each channel is presented as one packed vector. Every I/O is
// registered at the boundary so the report measures the core, not the pad path.

module VX_rtu_core_top import VX_gpu_pkg::*, VX_rtu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_LANES = `VX_CFG_NUM_SFU_LANES,
    parameter NUM_WARPS = `VX_CFG_NUM_WARPS,
    parameter NUM_SRCS  = 1,
    parameter TAG_WIDTH = RTU_REQ_ARB1_TAG_WIDTH,
    parameter CACHE_WORD_SIZE  = RTCACHE_WORD_SIZE,
    parameter CACHE_TAG_WIDTH  = RTCACHE_TAG_WIDTH,
    parameter CACHE_ADDR_WIDTH = `VX_CFG_MEM_ADDR_WIDTH - `CLOG2(CACHE_WORD_SIZE),

    // Channel widths (mirror VX_rtu_bus_slice / VX_rtu_bus_if).
    parameter SRC_WIDTH = `UP(`CLOG2(NUM_SRCS)),
    parameter ARM_DATAW = SRC_WIDTH + NW_WIDTH + NUM_LANES
                        + `VX_CFG_MEM_ADDR_WIDTH + 16 + 16 + 32 + TAG_WIDTH,
    parameter REQ_DATAW = 1 + SRC_WIDTH + NW_WIDTH
                        + NUM_LANES * 32 + NUM_LANES * RTU_CB_ACTION_BITS,
    parameter WIN_DATAW = 1 + NW_WIDTH + RTU_SLOT_BITS
                        + NUM_LANES + NUM_LANES * 32 + TAG_WIDTH
) (
    input  wire clk,
    input  wire reset,

    // arm (unit -> RTU): a warp armed a TRACE; its warp-uniform ray half
    input  wire                        arm_valid,
    input  wire [ARM_DATAW-1:0]        arm_data,
    output wire                        arm_ready,

    // req (unit -> RTU): RAY beats, and the warp's CONTINUE
    input  wire                        req_valid,
    input  wire [REQ_DATAW-1:0]        req_data,
    output wire                        req_ready,

    // win (RTU -> unit): one masked hit-window slot write per beat
    output wire                        win_valid,
    output wire [WIN_DATAW-1:0]        win_data,
    input  wire                        win_ready,

    // RTCache port (master): node/leaf line fetch
    output wire                        cache_req_valid,
    output wire                        cache_req_rw,
    output wire [CACHE_WORD_SIZE-1:0]  cache_req_byteen,
    output wire [CACHE_ADDR_WIDTH-1:0] cache_req_addr,
    output wire [CACHE_WORD_SIZE*8-1:0] cache_req_data,
    output wire [CACHE_TAG_WIDTH-1:0]  cache_req_tag,
    input  wire                        cache_req_ready,

    input  wire                        cache_rsp_valid,
    input  wire [CACHE_WORD_SIZE*8-1:0] cache_rsp_data,
    input  wire [CACHE_TAG_WIDTH-1:0]  cache_rsp_tag,
    output wire                        cache_rsp_ready
);
    // ── registered input boundary ─────────────────────────────────────────
    reg arm_valid_r, req_valid_r, win_ready_r, cache_req_ready_r, cache_rsp_valid_r, cache_rsp_ready_out;
    reg [ARM_DATAW-1:0]         arm_data_r;
    reg [REQ_DATAW-1:0]         req_data_r;
    reg [CACHE_WORD_SIZE*8-1:0] cache_rsp_data_r;
    reg [CACHE_TAG_WIDTH-1:0]   cache_rsp_tag_r;
    always @(posedge clk) begin
        arm_valid_r       <= arm_valid;
        arm_data_r        <= arm_data;
        req_valid_r       <= req_valid;
        req_data_r        <= req_data;
        win_ready_r       <= win_ready;
        cache_req_ready_r <= cache_req_ready;
        cache_rsp_valid_r <= cache_rsp_valid;
        cache_rsp_data_r  <= cache_rsp_data;
        cache_rsp_tag_r   <= cache_rsp_tag;
    end

    // ── the RTU bus interface, fed from the registered ports ───────────────
    VX_rtu_bus_if #(
        .NUM_LANES (NUM_LANES),
        .TAG_WIDTH (TAG_WIDTH),
        .SRC_WIDTH (SRC_WIDTH)
    ) rtu_bus_if ();

    assign rtu_bus_if.arm_valid = arm_valid_r;
    assign rtu_bus_if.arm_data  = arm_data_r;   // packed bitcast (equal width)
    assign rtu_bus_if.req_valid = req_valid_r;
    assign rtu_bus_if.req_data  = req_data_r;
    assign rtu_bus_if.win_ready = win_ready_r;

    // ── the RTCache memory interface ───────────────────────────────────────
    VX_mem_bus_if #(
        .DATA_SIZE (CACHE_WORD_SIZE),
        .TAG_WIDTH (CACHE_TAG_WIDTH)
    ) cache_bus_if ();

    assign cache_bus_if.req_ready    = cache_req_ready_r;
    assign cache_bus_if.rsp_valid    = cache_rsp_valid_r;
    assign cache_bus_if.rsp_data.data = cache_rsp_data_r;
    assign cache_bus_if.rsp_data.tag  = cache_rsp_tag_r;

    // ── DUT ────────────────────────────────────────────────────────────────
    VX_rtu_core #(
        .INSTANCE_ID     (INSTANCE_ID),
        .NUM_LANES       (NUM_LANES),
        .NUM_WARPS       (NUM_WARPS),
        .NUM_SRCS        (NUM_SRCS),
        .TAG_WIDTH       (TAG_WIDTH),
        .CACHE_DATA_SIZE (CACHE_WORD_SIZE),
        .CACHE_TAG_WIDTH (CACHE_TAG_WIDTH)
    ) rtu_core (
        .clk          (clk),
        .reset        (reset),
        .rtu_bus_if   (rtu_bus_if),
        .cache_bus_if (cache_bus_if)
    );

    // ── registered output boundary ─────────────────────────────────────────
    reg arm_ready_o, req_ready_o, win_valid_o, cache_req_valid_o, cache_req_rw_o;
    reg [WIN_DATAW-1:0]         win_data_o;
    reg [CACHE_WORD_SIZE-1:0]   cache_req_byteen_o;
    reg [CACHE_ADDR_WIDTH-1:0]  cache_req_addr_o;
    reg [CACHE_WORD_SIZE*8-1:0] cache_req_data_o;
    reg [CACHE_TAG_WIDTH-1:0]   cache_req_tag_o;
    always @(posedge clk) begin
        arm_ready_o        <= rtu_bus_if.arm_ready;
        req_ready_o        <= rtu_bus_if.req_ready;
        win_valid_o        <= rtu_bus_if.win_valid;
        win_data_o         <= rtu_bus_if.win_data;   // packed bitcast
        cache_req_valid_o  <= cache_bus_if.req_valid;
        cache_req_rw_o     <= cache_bus_if.req_data.rw;
        cache_req_byteen_o <= cache_bus_if.req_data.byteen;
        cache_req_addr_o   <= cache_bus_if.req_data.addr;
        cache_req_data_o   <= cache_bus_if.req_data.data;
        cache_req_tag_o    <= cache_bus_if.req_data.tag;
        cache_rsp_ready_out <= cache_bus_if.rsp_ready;
    end

    assign arm_ready       = arm_ready_o;
    assign req_ready       = req_ready_o;
    assign win_valid       = win_valid_o;
    assign win_data        = win_data_o;
    assign cache_req_valid = cache_req_valid_o;
    assign cache_req_rw    = cache_req_rw_o;
    assign cache_req_byteen= cache_req_byteen_o;
    assign cache_req_addr  = cache_req_addr_o;
    assign cache_req_data  = cache_req_data_o;
    assign cache_req_tag   = cache_req_tag_o;
    assign cache_rsp_ready = cache_rsp_ready_out;

    `UNUSED_VAR (cache_bus_if.req_data.attr)

endmodule
