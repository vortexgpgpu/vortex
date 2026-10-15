/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Geometry given in window coordinates of the 8x8 target, which the shader
 * also passes on as an input: its value at any point is then that point's own
 * window position, so where the fragment shader evaluated it can be read off
 * the result.
 *
 * Vertices 0-5 are an oversized full-screen quad. Vertices 6-8 are a triangle
 * whose only edge inside the target is the vertical line x = 4.5: column 4 is
 * covered at samples 0 and 2 (x offsets 0.375 and 0.125) and not at samples 1
 * and 3 (0.875 and 0.625), columns 0-3 are covered and 5-7 are not. */
#version 450

layout(location = 0) out vec2 v_win;

const vec2 EXTENT = vec2(8.0, 8.0);

vec2 positions[9] = vec2[](
   vec2(-4.0, -4.0), vec2(12.0, -4.0), vec2(12.0, 12.0),
   vec2(-4.0, -4.0), vec2(12.0, 12.0), vec2(-4.0, 12.0),
   vec2( 4.5, -8.0), vec2( 4.5, 16.0), vec2(-12.0, 4.0)
);

void main()
{
   vec2 win = positions[gl_VertexIndex];
   v_win = win;
   gl_Position = vec4(win / EXTENT * 2.0 - 1.0, 0.0, 1.0);
}
