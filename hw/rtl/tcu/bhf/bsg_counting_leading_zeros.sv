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

module bsg_counting_leading_zeros #(
    parameter width_p = 1,
    parameter lg_width_p = `LOG2UP(width_p)
) (
    input  wire [width_p-1:0]    a_i,
    output wire [lg_width_p-1:0] num_zero_o
);
    VX_lzc #(
        .N (width_p)
    ) lzc (
        .data_in   (a_i),
        .data_out  (num_zero_o),
        `UNUSED_PIN(valid_out)
    );

endmodule
