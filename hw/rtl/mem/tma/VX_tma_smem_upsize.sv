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

module VX_tma_smem_upsize import VX_gpu_pkg::*; #(
    parameter SRC_DATA_SIZE = LSU_WORD_SIZE,
    parameter DST_DATA_SIZE = TMA_SMEM_WORD_SIZE,
    parameter TAG_WIDTH = LMEM_TAG_WIDTH,
    parameter SRC_ADDR_WIDTH = (`MEM_ADDR_WIDTH - `CLOG2(SRC_DATA_SIZE)),
    parameter DST_ADDR_WIDTH = (`MEM_ADDR_WIDTH - `CLOG2(DST_DATA_SIZE))
) (
    input wire clk,
    input wire reset,

    VX_mem_bus_if.slave src_bus_if,
    VX_mem_bus_if.master dst_bus_if
);
    localparam SRC_DATAW = SRC_DATA_SIZE * 8;
    localparam RATIO = (DST_DATA_SIZE >= SRC_DATA_SIZE) ? (DST_DATA_SIZE / SRC_DATA_SIZE) : 1;
    localparam RATIO_BITS = `CLOG2(RATIO);
    localparam RATIO_SEL_W = `UP(RATIO_BITS);

    `STATIC_ASSERT(DST_DATA_SIZE >= SRC_DATA_SIZE, ("invalid parameter"))
    `STATIC_ASSERT(0 == (DST_DATA_SIZE % SRC_DATA_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(SRC_ADDR_WIDTH >= RATIO_BITS, ("invalid parameter"))

    wire [RATIO_SEL_W-1:0] req_idx = RATIO_SEL_W'(src_bus_if.req_data.addr);
    wire [SRC_ADDR_WIDTH-RATIO_BITS-1:0] req_addr_hi = src_bus_if.req_data.addr[SRC_ADDR_WIDTH-1:RATIO_BITS];

    wire req_fire = src_bus_if.req_valid && src_bus_if.req_ready;
    wire dst_rsp_fire = dst_bus_if.rsp_valid && dst_bus_if.rsp_ready;

    reg [RATIO_SEL_W-1:0] rsp_idx_r;
    reg rsp_idx_valid_r;

    always @(posedge clk) begin
        if (reset) begin
            rsp_idx_r <= '0;
            rsp_idx_valid_r <= 1'b0;
        end else begin
            if (req_fire && ~src_bus_if.req_data.rw) begin
                rsp_idx_r <= req_idx;
                rsp_idx_valid_r <= 1'b1;
            end
            if (dst_rsp_fire) begin
                rsp_idx_valid_r <= 1'b0;
            end
        end
    end

    `RUNTIME_ASSERT(~(req_fire && ~src_bus_if.req_data.rw && rsp_idx_valid_r),
        ("%t: %s: multi-outstanding read is unsupported", $time, `STRINGIFY(`SCOPE)))

    reg [RATIO-1:0][SRC_DATAW-1:0] dst_req_data_lanes;
    reg [RATIO-1:0][SRC_DATA_SIZE-1:0] dst_req_byteen_lanes;

    always @(*) begin
        dst_req_data_lanes = '0;
        dst_req_byteen_lanes = '0;
        dst_req_data_lanes[req_idx] = src_bus_if.req_data.data;
        dst_req_byteen_lanes[req_idx] = src_bus_if.req_data.byteen;
    end

    wire [RATIO-1:0][SRC_DATAW-1:0] dst_rsp_data_lanes = dst_bus_if.rsp_data.data;
    wire [RATIO_SEL_W-1:0] rsp_idx = rsp_idx_valid_r ? rsp_idx_r : '0;

    assign dst_bus_if.req_valid = src_bus_if.req_valid;
    assign dst_bus_if.req_data.rw = src_bus_if.req_data.rw;
    assign dst_bus_if.req_data.addr = DST_ADDR_WIDTH'(req_addr_hi);
    assign dst_bus_if.req_data.data = dst_req_data_lanes;
    assign dst_bus_if.req_data.byteen = dst_req_byteen_lanes;
    assign dst_bus_if.req_data.flags = src_bus_if.req_data.flags;
    assign dst_bus_if.req_data.tag = TAG_WIDTH'(src_bus_if.req_data.tag);

    assign src_bus_if.req_ready = dst_bus_if.req_ready;

    assign src_bus_if.rsp_valid = dst_bus_if.rsp_valid;
    assign src_bus_if.rsp_data.data = src_rsp_data_lanes(rsp_idx, dst_rsp_data_lanes);
    assign src_bus_if.rsp_data.tag = TAG_WIDTH'(dst_bus_if.rsp_data.tag);
    assign dst_bus_if.rsp_ready = src_bus_if.rsp_ready;

    function automatic [SRC_DATAW-1:0] src_rsp_data_lanes (
        input [RATIO_SEL_W-1:0] lane_idx,
        input [RATIO-1:0][SRC_DATAW-1:0] lanes
    );
    begin
        src_rsp_data_lanes = lanes[lane_idx];
    end
    endfunction

endmodule
