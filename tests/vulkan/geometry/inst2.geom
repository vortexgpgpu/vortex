/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// GS instancing, 2 invocations: invocation i of each input point draws a
// 16x8 quad shifted i pixels right and down, in a colour naming the invocation
// and the point, so with no blending the image shows which invocation of which
// point came last.
layout(points, invocations = 2) in;
layout(triangle_strip, max_vertices = 4) out;
layout(location = 0) flat out vec4 gcol;
void main()
{
   uint c = (uint(gl_PrimitiveIDIn) * 5u + uint(gl_InvocationID)) % 15u + 1u;
   vec4 col = vec4(float(c & 1u), float((c >> 1) & 1u), float((c >> 2) & 1u),
                   float((c >> 3) & 1u));
   vec4 o = gl_in[0].gl_Position +
            vec4(float(gl_InvocationID), float(gl_InvocationID), 0.0, 0.0) / 32.0;
   const vec2 h = vec2(8.0, 4.0) / 32.0;
   for (int i = 0; i < 4; i++) {
      vec2 s = vec2((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0);
      gl_Position = o + vec4(s * h, 0.0, 0.0);
      gcol = col;
      EmitVertex();
   }
}
