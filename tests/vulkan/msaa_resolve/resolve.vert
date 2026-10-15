/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Oversized full-screen quad: every sample of every pixel is inside it, so
 * every sample of a source is written by the fill. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.5, -1.5), vec2( 1.5, -1.5), vec2( 1.5,  1.5),
   vec2(-1.5, -1.5), vec2( 1.5,  1.5), vec2(-1.5,  1.5)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
