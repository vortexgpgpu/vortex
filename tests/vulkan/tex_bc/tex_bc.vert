/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle covering the whole target, from gl_VertexIndex. */
#version 450

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
}
