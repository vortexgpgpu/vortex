/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input point becomes one triangle strip of vn cells (8 wide, 4 tall),
// every vertex of its own colour, so the provoking vertex of each triangle --
// and the winding of the odd ones, under back-face culling -- shows.
layout(points) in;
layout(triangle_strip, max_vertices = 14) out;
layout(location = 1) flat in uint vn[];
layout(location = 0) flat out vec4 gcol;
vec4 tint(int k)
{
   int c = k % 15 + 1;
   return vec4(float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1), float((c >> 3) & 1));
}
void main()
{
   int k = 0;
   for (uint c = 0u; c <= vn[0]; c++) {
      for (int r = 0; r < 2; r++) {
         gl_Position = gl_in[0].gl_Position + vec4(8.0 * float(c), 4.0 * float(r), 0.0, 0.0) / 32.0;
         gcol = tint(gl_PrimitiveIDIn * 3 + k++);
         EmitVertex();
      }
   }
}
