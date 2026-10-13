/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Nine hardcoded positions: vertices 0-2 are a triangle in the left half of the
 * target, 3-5 one in the right half, 6-8 a thin one in the gap between them.
 * Which is drawn depends only on firstVertex, so a draw that ignores it lands
 * on the wrong part of the image rather than merely covering a different
 * number of pixels. */
#version 450

vec2 positions[9] = vec2[](
   /* left */
   vec2(-0.9, -0.9), vec2(-0.1, -0.9), vec2(-0.5,  0.9),
   /* right */
   vec2( 0.1, -0.9), vec2( 0.9, -0.9), vec2( 0.5,  0.9),
   /* gap */
   vec2(-0.08, -0.9), vec2( 0.08, -0.9), vec2( 0.0,  0.9)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
