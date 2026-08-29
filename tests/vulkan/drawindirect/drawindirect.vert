/* Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 *
 * Six hardcoded positions: vertices 0-2 are a triangle in the left half of the
 * target, 3-5 one in the right half. Which of the two is drawn depends only on
 * firstVertex, so a draw that ignores it lands on the wrong side of the image
 * rather than merely covering a different number of pixels. */
#version 450

vec2 positions[6] = vec2[](
   /* left */
   vec2(-0.9, -0.9), vec2(-0.1, -0.9), vec2(-0.5,  0.9),
   /* right */
   vec2( 0.1, -0.9), vec2( 0.9, -0.9), vec2( 0.5,  0.9)
);

void main()
{
   gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
