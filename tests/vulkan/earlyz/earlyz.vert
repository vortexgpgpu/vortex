/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad at a caller-chosen depth. Every draw in this test covers the
 * whole target, so which colour survives is decided by the depth test alone and
 * never by coverage. */
#version 450

layout(push_constant) uniform PC {
   float z;        /* plane depth for this draw */
   float zwrite;   /* depth the fragment stage writes, where it writes one */
   vec4  color;
} pc;

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], pc.z, 1.0);
}
