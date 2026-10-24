/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input triangle as it came, its vertices' colours flat with red and blue
// swapped -- a draw that skipped this stage would keep them.
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
layout(location = 0) in vec4 vcol[];
layout(location = 0) flat out vec4 gcol;
void main()
{
   for (int i = 0; i < 3; i++) {
      gl_Position = gl_in[i].gl_Position;
      gcol = vcol[i].bgra;
      EmitVertex();
   }
}
