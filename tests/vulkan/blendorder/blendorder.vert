/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One full-screen quad per instance, so every pixel receives one fragment per
 * instance and the instances must merge in draw order. */
#version 450

layout(location = 0) flat out int v_layer;

vec2 positions[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
   v_layer = gl_InstanceIndex;
}
