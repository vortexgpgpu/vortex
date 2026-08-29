/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Oversized full-screen quad. Every sample of every pixel is covered, so
 * the resolve of a solid fill has nothing partial in it and any pixel that
 * comes back between the clear and the fill is a defect rather than an edge. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.5, -1.5), vec2( 1.5, -1.5), vec2( 1.5,  1.5),
   vec2(-1.5, -1.5), vec2( 1.5,  1.5), vec2(-1.5,  1.5)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
