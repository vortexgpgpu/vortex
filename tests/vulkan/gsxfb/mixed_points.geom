/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Input point p is drawn on stream 0, which nothing captures, and emits the
// point's own count of points on stream 1, captured into buffer 0 as (p, j):
// both counts are found at run time, and they differ.
layout(points) in;
layout(points, max_vertices = 9) out;
layout(location = 0) flat in uint vn[];
layout(stream = 1, location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c;
void main()
{
   uint p = uint(gl_PrimitiveIDIn);
   gl_Position = gl_in[0].gl_Position;
   gl_PointSize = 1.0;
   EmitStreamVertex(0);
   for (uint j = 0u; j < vn[0]; j++) {
      c = uvec2(p, j);
      EmitStreamVertex(1);
   }
}
