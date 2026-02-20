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

module VX_tma_smem_downsize import VX_gpu_pkg::*; #(
    parameter SRC_DATA_SIZE = TMA_SMEM_WORD_SIZE,
    parameter DST_DATA_SIZE = LSU_WORD_SIZE,
    parameter TAG_WIDTH = LMEM_TAG_WIDTH,
    parameter SRC_ADDR_WIDTH = (`MEM_ADDR_WIDTH - `CLOG2(SRC_DATA_SIZE)),
    parameter DST_ADDR_WIDTH = (`MEM_ADDR_WIDTH - `CLOG2(DST_DATA_SIZE))
) (
    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave src_bus_if,
    VX_mem_bus_if.master dst_bus_if
);
    localparam DST_DATAW = DST_DATA_SIZE * 8;
    localparam RATIO = (SRC_DATA_SIZE >= DST_DATA_SIZE) ? (SRC_DATA_SIZE / DST_DATA_SIZE) : 1;
    localparam RATIO_BITS = `CLOG2(RATIO);
    localparam RATIO_SEL_W = `UP(RATIO_BITS);

    `STATIC_ASSERT(SRC_DATA_SIZE >= DST_DATA_SIZE, ("invalid parameter"))
    `STATIC_ASSERT(0 == (SRC_DATA_SIZE % DST_DATA_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(DST_ADDR_WIDTH >= SRC_ADDR_WIDTH, ("invalid parameter"))

    wire [RATIO-1:0][DST_DATAW-1:0] src_req_data_lanes = src_bus_if.req_data.data;
    wire [RATIO-1:0][DST_DATA_SIZE-1:0] src_req_byteen_lanes = src_bus_if.req_data.byteen;

    wire [RATIO-1:0] req_lane_active;
    for (genvar i = 0; i < RATIO; ++i) begin : g_req_lane_active
        assign req_lane_active[i] = |src_req_byteen_lanes[i];
    end

    reg [RATIO_SEL_W-1:0] req_lane_idx;
    always @(*) begin
        req_lane_idx = '0;
        for (integer i = 0; i < RATIO; ++i) begin
            if (req_lane_active[i]) begin
                req_lane_idx = RATIO_SEL_W'(i);
            end
        end
    end

    if (RATIO > 1) begin : g_req_lane_assert
        wire [RATIO-1:0] req_lane_multi_hot = req_lane_active & (req_lane_active - RATIO'(1));
        `RUNTIME_ASSERT(~(src_bus_if.req_valid && (|req_lane_multi_hot)),
            ("%t: %s: multi-lane downsize request is unsupported", $time, `STRINGIFY(`SCOPE)))
    end

    wire req_fire = src_bus_if.req_valid && src_bus_if.req_ready;
    wire [DST_ADDR_WIDTH-1:0] dst_req_addr;
    if (RATIO_BITS != 0) begin : g_dst_addr_ratio
        assign dst_req_addr = DST_ADDR_WIDTH'({src_bus_if.req_data.addr, req_lane_idx[RATIO_BITS-1:0]});
    end else begin : g_dst_addr_noratio
        assign dst_req_addr = DST_ADDR_WIDTH'(src_bus_if.req_data.addr);
    end

    reg [RATIO_SEL_W-1:0] rsp_lane_idx_r;
    reg rsp_lane_idx_valid_r;

    always @(posedge clk) begin
        if (reset) begin
            rsp_lane_idx_r <= '0;
            rsp_lane_idx_valid_r <= 1'b0;
        end else begin
            if (req_fire && ~src_bus_if.req_data.rw) begin
                rsp_lane_idx_r <= req_lane_idx;
                rsp_lane_idx_valid_r <= 1'b1;
            end
            if (dst_bus_if.rsp_valid && dst_bus_if.rsp_ready) begin
                rsp_lane_idx_valid_r <= 1'b0;
            end
        end
    end

    `RUNTIME_ASSERT(~(req_fire && ~src_bus_if.req_data.rw && rsp_lane_idx_valid_r),
        ("%t: %s: multi-outstanding read is unsupported", $time, `STRINGIFY(`SCOPE)))

    wire [RATIO_SEL_W-1:0] rsp_lane_idx = rsp_lane_idx_valid_r ? rsp_lane_idx_r : '0;
    reg [RATIO-1:0][DST_DATAW-1:0] src_rsp_data_lanes;
    always @(*) begin
        src_rsp_data_lanes = '0;
        src_rsp_data_lanes[rsp_lane_idx] = dst_bus_if.rsp_data.data;
    end

    assign dst_bus_if.req_valid = src_bus_if.req_valid;
    assign dst_bus_if.req_data.rw = src_bus_if.req_data.rw;
    assign dst_bus_if.req_data.addr = dst_req_addr;
    assign dst_bus_if.req_data.byteen = src_req_byteen_lanes[req_lane_idx];
    assign dst_bus_if.req_data.data = src_req_data_lanes[req_lane_idx];
    assign dst_bus_if.req_data.flags = src_bus_if.req_data.flags;
    assign dst_bus_if.req_data.tag = TAG_WIDTH'(src_bus_if.req_data.tag);
    assign src_bus_if.req_ready = dst_bus_if.req_ready;

    assign src_bus_if.rsp_valid = dst_bus_if.rsp_valid;
    assign src_bus_if.rsp_data.data = src_rsp_data_lanes;
    assign src_bus_if.rsp_data.tag = TAG_WIDTH'(dst_bus_if.rsp_data.tag);
    assign dst_bus_if.rsp_ready = src_bus_if.rsp_ready;

endmodule
