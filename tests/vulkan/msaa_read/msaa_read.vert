/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Oversized full-screen quad, used by both passes. Covering every sample of
 * every pixel is what makes the scene uniform: the answer is then one colour
 * for the whole target, and a target that comes back with more than one colour
 * in it is a defect and not a coverage fraction. */
#version 450

vec2 positions[6] = vec2[](
   vec2(-1.5, -1.5), vec2( 1.5, -1.5), vec2( 1.5,  1.5),
   vec2(-1.5, -1.5), vec2( 1.5,  1.5), vec2(-1.5,  1.5)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
