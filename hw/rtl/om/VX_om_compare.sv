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

`include "VX_om_define.vh"

module VX_om_compare #(
    parameter DATAW = 24
) (
    // Inputs
    input wire [`VX_OM_DEPTH_FUNC_BITS-1:0] func,
    input wire [DATAW-1:0] a,
    input wire [DATAW-1:0] b,

    // Outputs
    output wire result
);
    wire [DATAW:0] sub = (a - b);
    wire equal = (0 == sub);
    wire less  = sub[DATAW];

    reg result_r;

    always @(*) begin
        case (func)
            `VX_OM_DEPTH_FUNC_NEVER    : result_r = 0;
            `VX_OM_DEPTH_FUNC_LESS     : result_r = less;
            `VX_OM_DEPTH_FUNC_EQUAL    : result_r = equal;
            `VX_OM_DEPTH_FUNC_LEQUAL   : result_r = less || equal;
            `VX_OM_DEPTH_FUNC_GREATER  : result_r = ~(less || equal);
            `VX_OM_DEPTH_FUNC_NOTEQUAL : result_r = ~equal;
            `VX_OM_DEPTH_FUNC_GEQUAL   : result_r = ~less;
            `VX_OM_DEPTH_FUNC_ALWAYS   : result_r = 1;
            default                    : result_r = 'x;
        endcase        
    end

    assign result = result_r;

endmodule
