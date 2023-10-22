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

module VX_tex_format (
    input wire [`TEX_FORMAT_BITS-1:0] format,
    input wire [31:0]               texel_in,    
    output wire [31:0]              texel_out
);  

    reg [31:0] texel_out_r;

    always @(*) begin        
        case (format)
        `VX_TEX_FORMAT_A8R8G8B8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[15:8];
            texel_out_r[23:16] = texel_in[23:16];
            texel_out_r[31:24] = texel_in[31:24];
        end
        `VX_TEX_FORMAT_R5G6B5: begin
            texel_out_r[07:00] = {texel_in[4:0], texel_in[4:2]};
            texel_out_r[15:08] = {texel_in[10:5], texel_in[10:9]};
            texel_out_r[23:16] = {texel_in[15:11], texel_in[15:13]};
            texel_out_r[31:24] = 8'hff;
        end
        `VX_TEX_FORMAT_A1R5G5B5: begin
            texel_out_r[07:00] = {texel_in[4:0], texel_in[4:2]};
            texel_out_r[15:08] = {texel_in[9:5], texel_in[9:7]};
            texel_out_r[23:16] = {texel_in[14:10], texel_in[14:12]};
            texel_out_r[31:24] = {8{texel_in[15]}};
        end
        `VX_TEX_FORMAT_A4R4G4B4: begin
            texel_out_r[07:00] = {2{texel_in[3:0]}};
            texel_out_r[15:08] = {2{texel_in[7:4]}};
            texel_out_r[23:16] = {2{texel_in[11:8]}};
            texel_out_r[31:24] = {2{texel_in[15:12]}};
        end
        `VX_TEX_FORMAT_A8L8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[7:0];
            texel_out_r[23:16] = texel_in[7:0];
            texel_out_r[31:24] = texel_in[15:8];
        end
        `VX_TEX_FORMAT_L8: begin
            texel_out_r[07:00] = texel_in[7:0];
            texel_out_r[15:08] = texel_in[7:0];
            texel_out_r[23:16] = texel_in[7:0];
            texel_out_r[31:24] = 8'hff;
        end
        //`TEX_FORMAT_A8
        default: begin
            texel_out_r[07:00] = 8'hff;
            texel_out_r[15:08] = 8'hff;
            texel_out_r[23:16] = 8'hff;
            texel_out_r[31:24] = texel_in[7:0];
        end
        endcase
    end

    assign texel_out = texel_out_r;

endmodule
