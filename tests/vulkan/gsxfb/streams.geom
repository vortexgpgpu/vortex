/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Input point p emits s + 1 points on vertex stream s, each captured into
// buffer s as (p, 100 s + j). Stream 0's points are also drawn.
layout(points) in;
layout(points, max_vertices = 10) out;
layout(stream = 0, location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c0;
layout(stream = 1, location = 1, xfb_buffer = 1, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c1;
layout(stream = 2, location = 2, xfb_buffer = 2, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c2;
layout(stream = 3, location = 3, xfb_buffer = 3, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c3;
void main()
{
   uint p = uint(gl_PrimitiveIDIn);
   gl_Position = gl_in[0].gl_Position;
   gl_PointSize = 1.0;
   c0 = uvec2(p, 0u);
   EmitStreamVertex(0);
   for (uint j = 0u; j < 2u; j++) {
      c1 = uvec2(p, 100u + j);
      EmitStreamVertex(1);
   }
   for (uint j = 0u; j < 3u; j++) {
      c2 = uvec2(p, 200u + j);
      EmitStreamVertex(2);
   }
   for (uint j = 0u; j < 4u; j++) {
      c3 = uvec2(p, 300u + j);
      EmitStreamVertex(3);
   }
}
