/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Three bilinear probes of the 5x3 texture under MIRRORED_REPEAT; the host
 * holds the blends they must produce. No jitter: a coordinate that moved would
 * move the weights. */
#version 450

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

void main()
{
   /* Inside the mirrored period [1,2), across columns 3 and 4 on row 0. */
   float per = texture(tex, vec2(1.26, 0.5 / 3.0)).r;
   /* Rows 0 and 1 of column 2, reached from below 0. */
   float row = texture(tex, vec2(2.5 / 5.0, -0.25)).r;
   /* Columns 1 and 2 of row 1, in range. */
   float col = texture(tex, vec2(1.9 / 5.0, 1.5 / 3.0)).r;

   out_color = vec4(per, row, col, 1.0);
}
