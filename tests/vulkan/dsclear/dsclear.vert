/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One full-screen quad at a fixed depth the pass' depth clear admits, so the
 * stencil test alone decides which pixels survive. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.25, 1.0);
}
