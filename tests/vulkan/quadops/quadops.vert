/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle over the whole target; the scissor cuts it to [0, 33) on both
 * axes, so its edges cross no quad but at the scissor's right and bottom. */
#version 450

void main()
{
   vec2 p = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0, (gl_VertexIndex == 2) ? 3.0 : -1.0);
   gl_Position = vec4(p, 0.0, 1.0);
}
