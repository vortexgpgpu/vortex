/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * A quad covering exactly the 16x16 target at depth z = 1/8 + x/32 + y/64 of
 * window position (x, y): every vertex stays inside [0, 1], so nothing is
 * clipped, and the depth of a sample is the plane at that sample's position.
 * Keep in step with plane_depth() in main.c. */
#version 450

const vec2 EXTENT = vec2(16.0, 16.0);

vec2 corners[6] = vec2[](
   vec2( 0.0,  0.0), vec2(16.0,  0.0), vec2(16.0, 16.0),
   vec2( 0.0,  0.0), vec2(16.0, 16.0), vec2( 0.0, 16.0)
);

void main()
{
   vec2 win = corners[gl_VertexIndex];
   float z = 0.125 + win.x / 32.0 + win.y / 64.0;
   gl_Position = vec4(win / EXTENT * 2.0 - 1.0, z, 1.0);
}
