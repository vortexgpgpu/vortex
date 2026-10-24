/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input line as it came, its colours' red and blue swapped.
layout(lines) in;
layout(line_strip, max_vertices = 2) out;
layout(location = 0) in vec4 vcol[];
layout(location = 0) flat out vec4 gcol;
void main()
{
   for (int i = 0; i < 2; i++) {
      gl_Position = gl_in[i].gl_Position;
      gcol = vcol[i].bgra;
      EmitVertex();
   }
}
