/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each point becomes a quad, 8 pixels across and 4 down, of the point's colour.
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
layout(location = 0) in vec4 vcol[];
layout(location = 0) flat out vec4 gcol;
void main()
{
   const vec2 h = vec2(4.0, 2.0) / 32.0;
   for (int i = 0; i < 4; i++) {
      vec2 s = vec2((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0);
      gl_Position = gl_in[0].gl_Position + vec4(s * h, 0.0, 0.0);
      gcol = vcol[0];
      EmitVertex();
   }
}
