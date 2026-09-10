/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two draws share this shader. Vertices 0-2 are the marking triangle over
 * the lower-left half; vertices 3-8 are the full-screen quad the second
 * pass draws, admitted only where the first pass marked the stencil. */
#version 450

vec2 positions[9] = vec2[](
   /* marking triangle: lower-left half */
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
   /* full-screen quad */
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
