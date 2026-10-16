/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * strip = 1: a zig-zag strip across the viewport, two vertices per column.
 * strip = 0: horizontal lines, one per vertex pair, on pixel-row centres. */
#version 460

layout(push_constant) uniform PC {
   uint strip;
   uint height;
} pc;

void main()
{
   int v = gl_VertexIndex;
   vec2 p;
   if (pc.strip != 0u) {
      p = vec2(-1.0 + float(v / 2) * 0.5, (v & 1) != 0 ? 1.0 : -1.0);
   } else {
      float row = float(8 + 24 * (v / 2)) + 0.5;
      p = vec2((v & 1) != 0 ? 1.0 : -1.0, row / float(pc.height) * 2.0 - 1.0);
   }
   gl_Position = vec4(p, 0.0, 1.0);
}
