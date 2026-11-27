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

`include "VX_tex_define.vh"

// One lane's texel words out of a memory response. Each tap reads its source
// tap, either from the lane's own fetch or, when all lanes shared that
// address, from lane 0's; then it is aligned and truncated to the texel
// stride. The source tap, whether it was shared, and lane 0's words are the
// same for every lane, so they are selected once and come in per tap.
// Taps run level-major: tap (k * 4 + j) is the bilinear corner j of level k.
module VX_tex_rsp_qual import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter NUM_TAPS     = TEX_NUM_LEVELS * 4,
    parameter TAP_SEL_BITS = `CLOG2(NUM_TAPS)
) (
    input wire [`TEX_LGSTRIDE_BITS-1:0]         lgstride,
    input wire [NUM_TAPS-1:0][TAP_SEL_BITS-1:0] src_tap,
    input wire [NUM_TAPS-1:0]                   take_shared,
    input wire [NUM_TAPS-1:0][31:0]             shared_data,
    input wire [NUM_TAPS-1:0][1:0]              shared_align,
    input wire [NUM_TAPS-1:0][31:0]             lane_data,
    input wire [NUM_TAPS-1:0][1:0]              lane_align,

    output wire [TEX_NUM_LEVELS-1:0][3:0][31:0] data
);
    `STATIC_ASSERT((NUM_TAPS == TEX_NUM_LEVELS * 4), ("invalid parameter: NUM_TAPS=%0d", NUM_TAPS))

    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
        for (genvar j = 0; j < 4; ++j) begin : g_j
            localparam T = k * 4 + j;
            wire [31:0] src_data  = take_shared[T] ? shared_data[T]  : lane_data[src_tap[T]];
            wire [1:0]  src_align = take_shared[T] ? shared_align[T] : lane_align[src_tap[T]];

            reg [31:0] data_shifted;
            always @(*) begin
                data_shifted[31:16] = src_data[31:16];
                data_shifted[15:0]  = src_align[1] ? src_data[31:16]    : src_data[15:0];
                data_shifted[7:0]   = src_align[0] ? data_shifted[15:8] : data_shifted[7:0];
            end

            reg [31:0] data_stride;
            always @(*) begin
                case (lgstride)
                0:       data_stride = 32'(data_shifted[7:0]);
                1:       data_stride = 32'(data_shifted[15:0]);
                2:       data_stride = data_shifted;
                default: data_stride = 'x;
                endcase
            end

            assign data[k][j] = data_stride;
        end
    end

endmodule
