/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Green everywhere; a point also writes its gl_PointCoord into red and blue,
 * which the host checks against the pixel's position in the square. */
#version 450

layout(push_constant) uniform PC {
   layout(offset = 72) int use_coord;
} pc;

layout(location = 0) out vec4 out_color;

void main()
{
   vec2 pc2 = (pc.use_coord != 0) ? gl_PointCoord : vec2(0.0);
   out_color = vec4(pc2.x, 1.0, pc2.y, 1.0);
}
