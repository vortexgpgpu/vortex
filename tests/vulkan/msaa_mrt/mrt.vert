/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Two primitives in window coordinates of the 16x16 target, with their depth:
 *
 *   vertices 0-5  the flat quad, oversized, at depth 0.5 (primitive 0)
 *   vertices 6-8  the sloped triangle, covering the whole target, on the plane
 *                 z = Z0 + x/64 + y/128 (primitive 1)
 *
 * Every coefficient and vertex depth is a dyadic fraction, so the plane is
 * exact in float. Z0 puts the plane's crossing of 0.5 on the line
 * 2x + y = 24.0625, which no sample position (multiples of 1/8) lies on: the
 * nearest is 1/16 away, a depth margin of 1/2048. */
#version 450

layout(location = 0) flat out int v_prim;

const vec2 EXTENT = vec2(16.0, 16.0);

vec3 positions[9] = vec3[](
   vec3(-8.0, -8.0, 0.5), vec3(24.0, -8.0, 0.5), vec3(24.0, 24.0, 0.5),
   vec3(-8.0, -8.0, 0.5), vec3(24.0, 24.0, 0.5), vec3(-8.0, 24.0, 0.5),
   vec3(-8.0, -8.0, 0.12451171875),
   vec3(40.0, -8.0, 0.87451171875),
   vec3(-8.0, 40.0, 0.49951171875)
);

void main()
{
   vec3 p = positions[gl_VertexIndex];
   v_prim = gl_VertexIndex < 6 ? 0 : 1;
   gl_Position = vec4(p.xy / EXTENT * 2.0 - 1.0, p.z, 1.0);
}
