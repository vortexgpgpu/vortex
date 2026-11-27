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

// One lane's bilinear tap coordinates: the low and high texel coordinate of
// each axis at each mip level, wrapped, with the per-axis border flags, and
// the shifts that scale them to the level. Everything here depends on the
// lane's own mip level, and the level's distance from the base size
// (logdims - miplevel) is shared by the half-texel offset, the coordinate
// shift and the row pitch.
module VX_tex_clamp_lane import VX_gpu_pkg::*, VX_tex_pkg::*; #(
    parameter SHIFT_BITS = `CLOG2(`TEX_FXD_FRAC+1),
    parameter PITCH_BITS = `MAX(TEX_LOD_BITS, `TEX_LGSTRIDE_BITS) + 1
) (
    input wire                                   bilinear,
    input wire [1:0][TEX_WRAP_BITS-1:0]          wraps,
    input wire [1:0][TEX_LOD_BITS-1:0]           logdims,
    input wire [`TEX_LGSTRIDE_BITS-1:0]          log_stride,
    input wire [TEX_NUM_LEVELS-1:0][TEX_LOD_BITS-1:0] miplevel,
    input wire [1:0][`VX_TEX_FXD_BITS-1:0]       coords,

    output wire [TEX_NUM_LEVELS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_lo,
    output wire [TEX_NUM_LEVELS-1:0][1:0][`TEX_FXD_FRAC-1:0] clamped_hi,
    output wire [TEX_NUM_LEVELS-1:0][1:0]        border_lo,
    output wire [TEX_NUM_LEVELS-1:0][1:0]        border_hi,
    output wire [TEX_NUM_LEVELS-1:0][1:0][SHIFT_BITS-1:0] dim_shift,
    output wire [TEX_NUM_LEVELS-1:0][PITCH_BITS-1:0] log_pitch
);

    for (genvar k = 0; k < TEX_NUM_LEVELS; ++k) begin : g_k
        for (genvar j = 0; j < 2; ++j) begin  : g_j
            // Half a texel of this level. The shift is floored rather than
            // allowed to go negative: past the end of the mip chain the
            // level is 1x1 and its half-texel is half the whole coordinate
            // range, which is what the software sampler uses. Shifting left
            // first and letting the result truncate would collapse it to
            // zero instead, and the two taps would then agree on whether
            // they left the texture when they should not.
            wire [SHIFT_BITS-1:0] delta_shift = (logdims[j] >= miplevel[k])
                                              ? SHIFT_BITS'(logdims[j] - miplevel[k])
                                              : SHIFT_BITS'(0);
            wire [`TEX_FXD_FRAC-1:0] delta = `TEX_FXD_FRAC'(`TEX_FXD_HALF) >> delta_shift;
            wire [`VX_TEX_FXD_BITS-1:0] coord_lo = bilinear ? (coords[j] - `VX_TEX_FXD_BITS'(delta)) : coords[j];
            wire [`VX_TEX_FXD_BITS-1:0] coord_hi = bilinear ? (coords[j] + `VX_TEX_FXD_BITS'(delta)) : coords[j];

            VX_tex_wrap tex_wrap_lo (
                .wrap_i   (wraps[j]),
                .coord_i  (coord_lo),
                .coord_o  (clamped_lo[k][j]),
                .border_o (border_lo[k][j])
            );

            VX_tex_wrap tex_wrap_hi (
                .wrap_i   (wraps[j]),
                .coord_i  (coord_hi),
                .coord_o  (clamped_hi[k][j]),
                .border_o (border_hi[k][j])
            );
        end

        for (genvar j = 0; j < 2; ++j) begin : g_dim_shift
            assign dim_shift[k][j] = SHIFT_BITS'(`TEX_FXD_FRAC - `TEX_BLEND_FRAC) - (logdims[j] - miplevel[k]);
        end

        assign log_pitch[k] = PITCH_BITS'(logdims[0] - miplevel[k]) + PITCH_BITS'(log_stride);
    end

endmodule
