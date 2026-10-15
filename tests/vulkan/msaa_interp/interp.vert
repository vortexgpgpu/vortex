/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * One triangle, given in window coordinates of the 8x8 target, whose only edge
 * near the target is the line x + y = 24, far outside it: every pixel is fully
 * covered by this one primitive, so it gets exactly one pixel-rate invocation.
 * The window position is also passed on, without perspective, so an input
 * evaluated anywhere returns the point it was evaluated at. */
#version 450

layout(location = 0) noperspective out vec2 v_win;

const vec2 EXTENT = vec2(8.0, 8.0);

vec2 positions[3] = vec2[](
   vec2(-8.0, -8.0), vec2(32.0, -8.0), vec2(-8.0, 32.0)
);

void main()
{
   vec2 win = positions[gl_VertexIndex];
   v_win = win;
   gl_Position = vec4(win / EXTENT * 2.0 - 1.0, 0.0, 1.0);
}
