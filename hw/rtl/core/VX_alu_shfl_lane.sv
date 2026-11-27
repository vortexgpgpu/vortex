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

// One lane of SHFL: the source lane by mode (up, down, butterfly, index)
// within the lane's segment, and its value when in bounds and active. The
// lane's id is an input, so every lane is the same module.
module VX_alu_shfl_lane import VX_gpu_pkg::*; #(
    parameter NUM_LANES = 2,
    parameter LANE_BITS = `CLOG2(NUM_LANES)
) (
    input wire [LANE_BITS-1:0] lane_id,
    input wire [1:0]        op,
    // the lane's packed select operand: {mask, cval, bval} at 6-bit strides
    input wire [`VX_CFG_XLEN-1:0] sel,
    input wire [NUM_LANES-1:0] tmask,
    input wire [NUM_LANES-1:0][`VX_CFG_XLEN-1:0] data,
    // the lane's own value, wired apart so it needs no lane mux
    input wire [`VX_CFG_XLEN-1:0] self_data,
    output wire [`VX_CFG_XLEN-1:0] result
);
    `UNUSED_VAR (sel)

    wire [LANE_BITS-1:0] bval = sel[0 +: LANE_BITS];
    wire [LANE_BITS-1:0] cval = sel[6 +: LANE_BITS];
    wire [LANE_BITS-1:0] mask = sel[12 +: LANE_BITS];
    wire [LANE_BITS-1:0] minLane = (lane_id & mask);
    wire [LANE_BITS-1:0] maxLane = minLane | (cval & ~(mask));

    wire [LANE_BITS:0]   lane_up   = lane_id - bval;
    wire [LANE_BITS:0]   lane_down = lane_id + bval;
    wire [LANE_BITS-1:0] lane_bfly = lane_id ^ bval;
    wire [LANE_BITS-1:0] lane_idx  = minLane | (bval & ~mask);

    // The source lane is read before its bound check resolves, which
    // only picks between that read and the lane's own value.
    reg [LANE_BITS-1:0] lane;
    reg in_bounds;
    always @(*) begin
        case (op)
            INST_SHFL_UP: begin
                lane      = lane_up[LANE_BITS-1:0];
                in_bounds = ($signed(lane_up) >= $signed({1'b0, minLane}));
            end
            INST_SHFL_DOWN: begin
                lane      = lane_down[LANE_BITS-1:0];
                in_bounds = (lane_down <= {1'b0, maxLane});
            end
            INST_SHFL_BFLY: begin
                lane      = lane_bfly;
                in_bounds = (lane_bfly <= maxLane);
            end
            default: begin // INST_SHFL_IDX
                lane      = lane_idx;
                in_bounds = (lane_idx <= maxLane);
            end
        endcase
    end
    wire take_lane = in_bounds && tmask[lane];
    assign result = take_lane ? data[lane] : self_data;

endmodule
