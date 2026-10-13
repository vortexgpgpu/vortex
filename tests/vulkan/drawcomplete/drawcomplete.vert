/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Hardcoded positions, picked by vertex index, at the depth the push constant
 * gives:
 *   0-6    the restart pass: a triangle on the left (0-2), a vertex the
 *          restart leaves without a triangle (3), and a triangle on the
 *          right (4-6);
 *   7-12   a quad over the left half of the target;
 *   13-18  a quad over the whole target. */
#version 450

layout(push_constant) uniform pc_t { float z; } pc;

vec2 positions[19] = vec2[](
   vec2(-0.9, -0.9), vec2(-0.4, -0.9), vec2(-0.65, 0.9),
   vec2( 0.0,  0.9),
   vec2( 0.4, -0.9), vec2( 0.9, -0.9), vec2( 0.65, 0.9),
   vec2(-1.0, -1.0), vec2( 0.0, -1.0), vec2( 0.0,  1.0),
   vec2(-1.0, -1.0), vec2( 0.0,  1.0), vec2(-1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], pc.z, 1.0);
}
