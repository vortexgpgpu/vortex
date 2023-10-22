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

module VX_tex_stride (
    input wire [`TEX_FORMAT_BITS-1:0]    format,
    output wire [`TEX_LGSTRIDE_BITS-1:0] log_stride
);  

    reg [`TEX_LGSTRIDE_BITS-1:0] log_stride_r;  

    always @(*) begin
        case (format)
            `VX_TEX_FORMAT_A8R8G8B8: log_stride_r = 2;            
            `VX_TEX_FORMAT_R5G6B5,
            `VX_TEX_FORMAT_A1R5G5B5,
            `VX_TEX_FORMAT_A4R4G4B4,
            `VX_TEX_FORMAT_A8L8:     log_stride_r = 1;            
            `VX_TEX_FORMAT_L8,
            `VX_TEX_FORMAT_A8:       log_stride_r = 0;
            default:                 log_stride_r = 'x;
        endcase
    end

    assign log_stride = log_stride_r;

endmodule
