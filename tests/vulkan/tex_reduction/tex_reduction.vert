/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad. The fragment stage samples at constant coordinates. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
