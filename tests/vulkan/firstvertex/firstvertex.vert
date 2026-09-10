/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Twelve vertices forming two quads that cover opposite halves of the target.
 * gl_VertexIndex 0-5 is the left half, 6-11 the right, so a draw of six
 * vertices at firstVertex = 6 must fill the right half and leave the left
 * cleared. A gl_VertexIndex that drops firstVertex fills the left half
 * instead. */
#version 450

vec2 corners[6] = vec2[](
   vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
   vec2(0.0, -1.0), vec2(1.0,  1.0), vec2(0.0, 1.0)
);

void main()
{
   vec2 p = corners[gl_VertexIndex % 6];
   /* indices 0-5 occupy x in [-1,0]; 6-11 the mirrored x in [0,1] */
   if (gl_VertexIndex < 6) {
      p.x = p.x - 1.0;
   }
   gl_Position = vec4(p, 0.25, 1.0);
}
