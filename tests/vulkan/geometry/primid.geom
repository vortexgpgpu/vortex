/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */
#version 450
// Each input triangle passed on, with gl_PrimitiveID written as three times
// the input primitive's id plus one.
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
void main()
{
   for (int i = 0; i < 3; i++) {
      gl_Position = gl_in[i].gl_Position;
      gl_PrimitiveID = gl_PrimitiveIDIn * 3 + 1;
      EmitVertex();
   }
}
