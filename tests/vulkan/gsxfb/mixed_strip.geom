/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Input triangle p is drawn as it came on stream 0, which nothing captures,
// and becomes a strip of COUNT + 2 vertices on stream 1, in the right half of
// the target, which is captured into buffer 0 as (p, k) and never drawn.
// COUNT is 3, or with DYNAMIC the vertex's own count.
layout(triangles) in;
layout(triangle_strip, max_vertices = 12) out;
layout(location = 0) flat in uint vn[];
layout(stream = 1, location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c;
layout(constant_id = 0) const bool DYNAMIC = false;
void main()
{
   uint p = uint(gl_PrimitiveIDIn);
   for (int i = 0; i < 3; i++) {
      gl_Position = gl_in[i].gl_Position;
      EmitStreamVertex(0);
   }
   EndStreamPrimitive(0);
   uint n = DYNAMIC ? vn[0] : 3u;
   for (uint k = 0u; k < n + 2u; k++) {
      gl_Position = vec4(0.2 + 0.1 * float(k / 2u), -0.5 + 0.4 * float(k % 2u), 0.0, 1.0);
      c = uvec2(p, k);
      EmitStreamVertex(1);
   }
}
