/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Input triangle p becomes a strip of COUNT + 2 vertices -- COUNT triangles --
// vertex k captured into buffers 0 and 1 as (p, k). COUNT is 2, or with
// DYNAMIC the point's own count, which the compiler cannot know.
layout(triangles) in;
layout(triangle_strip, max_vertices = 8) out;
layout(location = 0) flat in uint vn[];
layout(location = 0, xfb_buffer = 0, xfb_offset = 0, xfb_stride = 8) flat out uvec2 c;
layout(location = 1, xfb_buffer = 1, xfb_offset = 4, xfb_stride = 12) flat out uint k1;
layout(constant_id = 0) const bool DYNAMIC = false;
void main()
{
   uint p = uint(gl_PrimitiveIDIn);
   uint n = DYNAMIC ? vn[0] : 2u;
   for (uint k = 0u; k < n + 2u; k++) {
      gl_Position = gl_in[0].gl_Position + vec4(0.05 * float(k / 2u), 0.05 * float(k % 2u), 0.0, 0.0);
      c = uvec2(p, k);
      k1 = 1000u * p + k;
      EmitVertex();
   }
}
