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

module VX_uop_mux import VX_gpu_pkg::*; #(
    parameter NUM_REQS = 1
) (
    input  wire [NUM_REQS-1:0] valid_in,
    input  ibuffer_t           data_in [NUM_REQS],
    input  wire [NUM_REQS-1:0] done_in,

    output wire                valid_out,
    output ibuffer_t           data_out,
    output wire                done_out
);
    // Priority select: higher index has higher priority.
    ibuffer_t uop_data_r;
    reg       uop_done_r;

    always @(*) begin
        uop_done_r = 'x;
        uop_data_r = 'x;
        for (integer i = 0; i < NUM_REQS; ++i) begin
            if (valid_in[i]) begin
                uop_data_r = data_in[i];
                uop_done_r = done_in[i];
            end
        end
    end

    assign valid_out = (| valid_in);
    assign data_out = uop_data_r;
    assign done_out = uop_done_r;

endmodule
