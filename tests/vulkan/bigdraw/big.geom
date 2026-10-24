/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Input point i becomes n one-pixel triangles, triangle j over the centre of
// pixel (8 i + j) % 65536: eight of them, or with DYNAMIC (i * 7) % 9, a count
// the compiler cannot know.
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;
layout(location = 0) flat out vec4 vcol;
layout(constant_id = 0) const bool DYNAMIC = false;
void main()
{
   uint i = uint(gl_PrimitiveIDIn);
   uint n = DYNAMIC ? (i * 7u) % 9u : 8u;
   for (uint j = 0u; j < n; j++) {
      uint p = (8u * i + j) % 65536u;
      vec2 c = vec2(float(p % 256u), float(p / 256u)) + 0.1;
      for (uint k = 0u; k < 3u; k++) {
         vec2 v = c + (k == 1u ? vec2(1.0, 0.0) : k == 2u ? vec2(0.0, 1.0) : vec2(0.0));
         gl_Position = vec4(v / 128.0 - 1.0, 0.0, 1.0);
         vcol = vec4(float(i & 255u), float((i >> 8) & 255u), float(j * 32u), 255.0) / 255.0;
         EmitVertex();
      }
      EndPrimitive();
   }
}
