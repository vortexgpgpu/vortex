/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two full-screen quads either side of the pass' depth clear: gl_VertexIndex
 * 0-5 is the near one, 6-11 the far one. Both cover the whole target, so a
 * pixel's colour says which quad the depth test admitted. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   float z = (gl_VertexIndex < 6) ? 0.25 : 0.75;
   gl_Position = vec4(positions[gl_VertexIndex % 6], z, 1.0);
}
