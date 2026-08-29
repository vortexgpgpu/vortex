/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A full-viewport triangle built from gl_VertexIndex. The geometry is
 * incidental; the test is about what the fragment colour becomes in memory. */
#version 450

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ?  0.9 : -0.9,
                 (gl_VertexIndex == 2) ?  0.9 : -0.9);
   gl_Position = vec4(p, 0.0, 1.0);
}
