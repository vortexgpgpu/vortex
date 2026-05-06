//!/bin/bash

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

`include "VX_platform.vh"

module VX_tex_sat #(
    parameter IN_W  = 8,
    parameter OUT_W = 4,
    parameter MODEL = 1
) (
    input wire [IN_W-1:0]   data_in,
    output wire [OUT_W-1:0] data_out
);
    `STATIC_ASSERT(((OUT_W+1) < IN_W), ("invalid parameter"))

    if (MODEL == 1) begin : g_model1
        wire [OUT_W-1:0] underflow_mask = {OUT_W{~data_in[IN_W-1]}};
        wire [OUT_W-1:0] overflow_mask = {OUT_W{(| data_in[IN_W-2:OUT_W])}};
        assign data_out = (data_in[OUT_W-1:0] & underflow_mask) | overflow_mask;
    end else begin : g_model0
        assign data_out = data_in[IN_W-1] ? OUT_W'(0) : ((data_in > {OUT_W{1'b1}}) ? {OUT_W{1'b1}} : OUT_W'(data_in));
    end

endmodule
