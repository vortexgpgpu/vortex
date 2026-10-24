/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Four invocations per input point, each emitting a two-triangle strip on
// screen: what the pipeline statistics count.
layout(points, invocations = 4) in;
layout(triangle_strip, max_vertices = 4) out;
void main()
{
   vec4 o = gl_in[0].gl_Position + vec4(0.1 * float(gl_InvocationID), 0.0, 0.0, 0.0);
   for (int i = 0; i < 4; i++) {
      gl_Position = o + vec4(0.05 * float(i & 1), 0.05 * float(i >> 1), 0.0, 0.0);
      EmitVertex();
   }
}
