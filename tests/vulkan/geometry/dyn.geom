/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input point becomes some number of overlapping 8x4 quads, each one two
// pixels right and one down of the last and of its own colour, the number
// taken from where `src` says: the vertex (vn), a push constant, or a texel.
// Every invocation counts itself and its quads into the storage buffer.
layout(points) in;
layout(triangle_strip, max_vertices = 24) out;
layout(location = 0) in vec4 vcol[];
layout(location = 1) flat in uint vn[];
layout(location = 0) flat out vec4 gcol;
layout(push_constant) uniform pc_t { uint src; uint n; } pc;
layout(set = 0, binding = 0) buffer cnt_t { uint invocations; uint quads; } cnt;
layout(set = 0, binding = 1) uniform sampler2D counts;
vec4 tint(uint q)
{
   uint c = q % 15u + 1u;
   return vec4(float(c & 1u), float((c >> 1) & 1u), float((c >> 2) & 1u), float((c >> 3) & 1u));
}
void main()
{
   uint n = pc.src == 0u ? vn[0]
          : pc.src == 1u ? pc.n + uint(gl_PrimitiveIDIn % 2)
          : uint(texelFetch(counts, ivec2(gl_PrimitiveIDIn, 0), 0).r * 255.0 + 0.5);
   atomicAdd(cnt.invocations, 1u);
   atomicAdd(cnt.quads, n);
   const vec2 h = vec2(4.0, 2.0) / 32.0;
   for (uint q = 0u; q < n; q++) {
      vec4 o = gl_in[0].gl_Position + vec4(vec2(2.0, 1.0) * float(q) / 32.0, 0.0, 0.0);
      for (int i = 0; i < 4; i++) {
         vec2 s = vec2((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0);
         gl_Position = o + vec4(s * h, 0.0, 0.0);
         gcol = tint(uint(gl_PrimitiveIDIn) * 7u + q);
         EmitVertex();
      }
      EndPrimitive();
   }
}
