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

`include "VX_tex_define.vh"

module VX_tex_wrap (
    input wire [`TEX_WRAP_BITS-1:0]    wrap_i,
    input wire [`VX_TEX_FXD_BITS-1:0]  coord_i,
    output wire [`VX_TEX_FXD_FRAC-1:0] coord_o
);
    
    reg [`VX_TEX_FXD_FRAC-1:0] coord_r;

    wire [`VX_TEX_FXD_FRAC-1:0] clamp;

    VX_tex_sat #(
        .IN_W  (`VX_TEX_FXD_BITS),
        .OUT_W (`VX_TEX_FXD_FRAC)
    ) sat_fx (
        .data_in  (coord_i),
        .data_out (clamp)
    );

    always @(*) begin
        case (wrap_i)
            `VX_TEX_WRAP_CLAMP:   
                coord_r = clamp;
            `VX_TEX_WRAP_MIRROR: 
                coord_r = coord_i[`VX_TEX_FXD_FRAC-1:0] ^ {`VX_TEX_FXD_FRAC{coord_i[`VX_TEX_FXD_FRAC]}};
            default: //`VX_TEX_WRAP_REPEAT
                coord_r = coord_i[`VX_TEX_FXD_FRAC-1:0];
        endcase
    end

    assign coord_o = coord_r;

endmodule
