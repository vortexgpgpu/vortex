/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad at the push-constant depth: coverage is never the
 * variable, so every outcome is a depth or stencil decision. */
#version 450

layout(push_constant) uniform PC {
   float z;
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
