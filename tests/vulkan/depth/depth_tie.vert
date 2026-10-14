/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Full-screen quad whose depth and colour come from the instance index (see
 * run_pairs in main.c): even instances are blue, odd ones red.
 *   0, 1   z = 0.25 and 0.15 of a 16-bit step further: one D16 value (16384)
 *   2, 3   z = 0.01 and a fifth of a 24-bit step further
 *   4, 5   window depths -0.5 and 1.5 in a [-1, 2] viewport */
#version 450

layout(location = 0) out vec3 v_color;

vec2 corners[6] = vec2[](
   vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
   vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

float depths[6] = float[](
   0.25, 0.25 + 0.15 / 65535.0,
   0.01, 0.01 + 0.2 / 16777216.0,
   1.0 / 6.0, 5.0 / 6.0
);

void main()
{
   gl_Position = vec4(corners[gl_VertexIndex], depths[gl_InstanceIndex], 1.0);
   v_color = (gl_InstanceIndex & 1) == 0 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
}
