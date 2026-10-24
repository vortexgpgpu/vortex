/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Each triangle drawn 24 times, invocation i into viewport i / 6 and layer
 * i % 6: gl_ViewportIndex and gl_Layer come from the geometry stage. */
#version 450

layout(triangles, invocations = 24) in;
layout(triangle_strip, max_vertices = 3) out;

void main()
{
   for (int k = 0; k < 3; k++) {
      gl_Position = gl_in[k].gl_Position;
      gl_ViewportIndex = gl_InvocationID / 6;
      gl_Layer = gl_InvocationID % 6;
      EmitVertex();
   }
}
